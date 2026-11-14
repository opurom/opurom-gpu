//! Headless Vulkan-style command recorder for OPUROM display lists.
//!
//! This module never loads a Vulkan driver and never submits a command buffer.
//! It translates the same validated, packed display-list stream that Metal
//! uploads into a typed command stream, then encodes that stream for headless
//! inspection. Rect primitives become two-triangle draws, particle fields
//! become point draws, per-command clips become scissor commands, framebuffer
//! staging becomes `CopyBufferToImage`, and Metal compute passes become
//! compute dispatches. A standalone triangle or image source command does not
//! exist in the current OPUROM display-list schema; those primitives are
//! represented here by `TriangleList` draws and the buffer-to-image copy.
//!
//! Stream framing, all integers little-endian:
//!
//! ```text
//! magic[8] = "OPRVKCB0" | version:u32 | width:u32 | height:u32
//! background:rgba8 | command_count:u64
//! command_count * { tag:u8 | payload_len:u64 | payload[payload_len] }
//! checksum:u64 = FNV-1a 64 over every preceding byte
//! ```
//!
//! Command tags are 1=BindPipeline, 2=SetScissor, 3=Draw,
//! 4=CopyBufferToImage and 5=Dispatch. Every payload has a fixed field order;
//! `decode_stream` rejects an unknown tag, an unexpected payload length,
//! trailing bytes and a bad checksum.

use std::convert::{TryFrom, TryInto};
use std::fmt;

use crate::display::{
    parse_display_list, validate_effect, Command, DisplayList, Pass, COMMAND_WORDS, COORD_BIAS,
    COORD_LIMIT, GRADIENT, GRADIENT_HORIZONTAL, KERNEL_BLUR_H, KERNEL_BLUR_V, KERNEL_EFFECT,
    KERNEL_GAUSSIAN_H, KERNEL_GAUSSIAN_V, KERNEL_MASK, KERNEL_SNAPSHOT, PARAM_WORDS, PARTICLES,
    PARTICLE_MAX_COUNT, PARTICLE_MAX_RADIUS, PARTICLE_MAX_SEED, ROUNDED, SOLID,
};
use crate::{display_byte_count, fnv1a, GpuError, KERNEL_BATCH};

pub use crate::display::MAX_COMMANDS;

pub const VULKAN_MAGIC: [u8; 8] = *b"OPRVKCB0";
pub const VULKAN_STREAM_VERSION: u32 = 1;
/// One display-list command expands to at most four typed Vulkan commands.
pub const MAX_VULKAN_COMMANDS: usize = 4 * MAX_COMMANDS + 16;
const MAX_PAYLOAD_BYTES: usize = 256;

const TAG_BIND_PIPELINE: u8 = 1;
const TAG_SET_SCISSOR: u8 = 2;
const TAG_DRAW: u8 = 3;
const TAG_COPY_BUFFER_TO_IMAGE: u8 = 4;
const TAG_DISPATCH: u8 = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VkError {
    InvalidViewport(String),
    OutOfRangeRect {
        command: usize,
        rect: [i64; 4],
    },
    InvalidClip {
        command: usize,
        clip: [i64; 4],
    },
    InvalidRadius {
        command: usize,
        radius: i64,
    },
    UnboundedCount {
        command: usize,
        count: i64,
        limit: i64,
    },
    InvalidEffect {
        command: usize,
        message: String,
    },
    MalformedDisplayList(String),
    InvalidEncoding(String),
}

impl From<GpuError> for VkError {
    fn from(error: GpuError) -> Self {
        Self::MalformedDisplayList(error.to_string())
    }
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidViewport(message) => write!(f, "InvalidViewport: {message}"),
            Self::OutOfRangeRect { command, rect } => {
                write!(f, "OutOfRangeRect: command {command} rect {rect:?}")
            }
            Self::InvalidClip { command, clip } => {
                write!(f, "InvalidClip: command {command} clip {clip:?}")
            }
            Self::InvalidRadius { command, radius } => {
                write!(f, "InvalidRadius: command {command} radius {radius}")
            }
            Self::UnboundedCount {
                command,
                count,
                limit,
            } => write!(
                f,
                "UnboundedCount: command {command} count {count} exceeds {limit}"
            ),
            Self::InvalidEffect { command, message } => {
                write!(f, "InvalidEffect: command {command} {message}")
            }
            Self::MalformedDisplayList(message) => {
                write!(f, "MalformedDisplayList: {message}")
            }
            Self::InvalidEncoding(message) => write!(f, "InvalidEncoding: {message}"),
        }
    }
}

impl std::error::Error for VkError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkRect2D {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkPrimitive {
    TriangleList,
    PointList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkEffect {
    Backdrop,
    Mask,
    Shadow,
    Glow,
    Glass,
    Liquid,
    BlurHorizontal,
    BlurVertical,
    GaussianHorizontal,
    GaussianVertical,
}

impl VkEffect {
    const fn tag(self) -> u8 {
        match self {
            Self::Backdrop => 0,
            Self::Mask => 1,
            Self::Shadow => 2,
            Self::Glow => 3,
            Self::Glass => 4,
            Self::Liquid => 5,
            Self::BlurHorizontal => 6,
            Self::BlurVertical => 7,
            Self::GaussianHorizontal => 8,
            Self::GaussianVertical => 9,
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => Self::Backdrop,
            1 => Self::Mask,
            2 => Self::Shadow,
            3 => Self::Glow,
            4 => Self::Glass,
            5 => Self::Liquid,
            6 => Self::BlurHorizontal,
            7 => Self::BlurVertical,
            8 => Self::GaussianHorizontal,
            9 => Self::GaussianVertical,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkPipeline {
    ClearImage,
    Solid,
    Rounded,
    Gradient,
    GradientHorizontal,
    Particles,
    Effect(VkEffect),
}

impl VkPipeline {
    /// Map a packed Metal regular-command kind to its Vulkan pipeline.
    pub fn from_display_kind(kind: u32) -> Option<Self> {
        Some(match kind {
            SOLID => Self::Solid,
            ROUNDED => Self::Rounded,
            GRADIENT => Self::Gradient,
            GRADIENT_HORIZONTAL => Self::GradientHorizontal,
            PARTICLES => Self::Particles,
            _ => return None,
        })
    }

    /// Return the packed Metal regular-command kind, if this is a draw pipeline.
    pub fn display_kind(self) -> Option<u32> {
        Some(match self {
            Self::Solid => SOLID,
            Self::Rounded => ROUNDED,
            Self::Gradient => GRADIENT,
            Self::GradientHorizontal => GRADIENT_HORIZONTAL,
            Self::Particles => PARTICLES,
            Self::ClearImage | Self::Effect(_) => return None,
        })
    }

    const fn tag(self) -> u8 {
        match self {
            Self::ClearImage => 0,
            Self::Solid => 1,
            Self::Rounded => 2,
            Self::Gradient => 3,
            Self::GradientHorizontal => 4,
            Self::Particles => 5,
            Self::Effect(_) => 6,
        }
    }
}

impl VkPrimitive {
    const fn tag(self) -> u8 {
        match self {
            Self::TriangleList => 0,
            Self::PointList => 1,
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        Some(match tag {
            0 => Self::TriangleList,
            1 => Self::PointList,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkDraw {
    pub pipeline: VkPipeline,
    pub primitive: VkPrimitive,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub rect: VkRect2D,
    pub scissor: VkRect2D,
    pub radius: u32,
    pub color: [u8; 4],
    pub second_color: Option<[u8; 4]>,
    pub alpha: u8,
    pub count: u32,
    pub seed: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkCopyBufferToImage {
    pub image_id: u32,
    pub buffer_offset: u64,
    pub bytes_per_row: u32,
    pub bytes_per_image: u32,
    pub extent: [u32; 2],
    pub color: [u8; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkDispatch {
    pub effect: VkEffect,
    pub kernel: u32,
    pub group_count: [u32; 3],
    pub params: [u32; PARAM_WORDS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkCommand {
    BindPipeline { pipeline: VkPipeline },
    SetScissor { rect: VkRect2D },
    Draw(VkDraw),
    CopyBufferToImage(VkCopyBufferToImage),
    Dispatch(VkDispatch),
}

impl VkCommand {
    const fn tag(self) -> u8 {
        match self {
            Self::BindPipeline { .. } => TAG_BIND_PIPELINE,
            Self::SetScissor { .. } => TAG_SET_SCISSOR,
            Self::Draw(_) => TAG_DRAW,
            Self::CopyBufferToImage(_) => TAG_COPY_BUFFER_TO_IMAGE,
            Self::Dispatch(_) => TAG_DISPATCH,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanBackend {
    width: u32,
    height: u32,
    background: [u8; 4],
    commands: Vec<VkCommand>,
}

impl Default for VulkanBackend {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            background: [0, 0, 0, 255],
            commands: Vec::new(),
        }
    }
}

impl VulkanBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn viewport(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn background(&self) -> [u8; 4] {
        self.background
    }

    pub fn commands(&self) -> &[VkCommand] {
        &self.commands
    }

    /// Parse a display-list JSON document, validate it, and record commands.
    pub fn record_json(&mut self, json: &str) -> Result<Vec<VkCommand>, VkError> {
        let list = parse_display_list(json).map_err(VkError::from)?;
        self.record(&list)
    }

    /// Translate a validated display list into a deterministic command stream.
    pub fn record(&mut self, list: &DisplayList) -> Result<Vec<VkCommand>, VkError> {
        let (packed, passes) = validate_stream(list)?;
        let byte_count = display_byte_count(list.width, list.height)
            .map_err(|error| VkError::InvalidViewport(error.to_string()))?;
        let bytes_per_row = list
            .width
            .checked_mul(4)
            .ok_or_else(|| VkError::InvalidViewport("RGBA8 row byte count overflow".to_owned()))?;
        let bytes_per_image = u32::try_from(byte_count)
            .map_err(|_| VkError::InvalidViewport("RGBA8 image byte count overflow".to_owned()))?;
        let group_count = [
            list.width.saturating_add(7) / 8,
            list.height.saturating_add(7) / 8,
            1,
        ];
        let mut commands = vec![
            VkCommand::BindPipeline {
                pipeline: VkPipeline::ClearImage,
            },
            VkCommand::CopyBufferToImage(VkCopyBufferToImage {
                image_id: 0,
                buffer_offset: 0,
                bytes_per_row,
                bytes_per_image,
                extent: [list.width, list.height],
                color: list.background,
            }),
        ];

        for pass in &passes {
            if pass.kernel == KERNEL_BATCH {
                let start = usize::try_from(pass.params[5]).map_err(|_| {
                    VkError::MalformedDisplayList("batch start does not fit usize".to_owned())
                })?;
                let count = usize::try_from(pass.params[6]).map_err(|_| {
                    VkError::MalformedDisplayList("batch count does not fit usize".to_owned())
                })?;
                let end = start.checked_add(count).ok_or_else(|| {
                    VkError::MalformedDisplayList("batch word offset overflow".to_owned())
                })?;
                end.checked_mul(COMMAND_WORDS).ok_or_else(|| {
                    VkError::MalformedDisplayList("batch byte offset overflow".to_owned())
                })?;
                for word_index in start..end {
                    let byte_start = word_index.checked_mul(COMMAND_WORDS).ok_or_else(|| {
                        VkError::MalformedDisplayList("command byte offset overflow".to_owned())
                    })?;
                    let byte_end = byte_start.checked_add(COMMAND_WORDS).ok_or_else(|| {
                        VkError::MalformedDisplayList("command byte offset overflow".to_owned())
                    })?;
                    let words = packed.get(byte_start..byte_end).ok_or_else(|| {
                        VkError::MalformedDisplayList(
                            "batch points outside the packed command stream".to_owned(),
                        )
                    })?;
                    let draw = draw_from_words(words, word_index)?;
                    debug_assert_eq!(VkPipeline::from_display_kind(words[0]), Some(draw.pipeline));
                    commands.push(VkCommand::BindPipeline {
                        pipeline: draw.pipeline,
                    });
                    commands.push(VkCommand::SetScissor { rect: draw.scissor });
                    commands.push(VkCommand::Draw(draw));
                }
            } else {
                let effect = effect_from_pass(pass)?;
                commands.push(VkCommand::BindPipeline {
                    pipeline: VkPipeline::Effect(effect),
                });
                commands.push(VkCommand::Dispatch(VkDispatch {
                    effect,
                    kernel: dispatch_kernel(effect),
                    group_count,
                    params: pass.params,
                }));
            }
        }

        self.width = list.width;
        self.height = list.height;
        self.background = list.background;
        self.commands = commands.clone();
        Ok(commands)
    }

    /// Encode the recorded command stream using the framing documented above.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&VULKAN_MAGIC);
        put_u32(&mut out, VULKAN_STREAM_VERSION);
        put_u32(&mut out, self.width);
        put_u32(&mut out, self.height);
        put_rgba(&mut out, self.background);
        put_u64(&mut out, self.commands.len() as u64);
        for command in &self.commands {
            put_u8(&mut out, command.tag());
            let payload = encode_payload(*command);
            put_u64(&mut out, payload.len() as u64);
            out.extend_from_slice(&payload);
        }
        let checksum = fnv1a(&out);
        put_u64(&mut out, checksum);
        out
    }

    /// Decode and verify a stream produced by `encode`.
    pub fn decode(bytes: &[u8]) -> Result<Self, VkError> {
        let split = bytes.len().checked_sub(8).ok_or_else(|| {
            VkError::InvalidEncoding("the stream is shorter than a checksum".to_owned())
        })?;
        let (body, checksum_bytes) = bytes.split_at(split);
        let checksum_array: [u8; 8] = checksum_bytes
            .try_into()
            .map_err(|_| VkError::InvalidEncoding("bad checksum length".to_owned()))?;
        let expected_checksum = u64::from_le_bytes(checksum_array);
        if fnv1a(body) != expected_checksum {
            return Err(VkError::InvalidEncoding("checksum mismatch".to_owned()));
        }

        let mut reader = Reader::new(body);
        let magic = reader.take(8)?;
        if magic != &VULKAN_MAGIC[..] {
            return Err(VkError::InvalidEncoding("bad magic".to_owned()));
        }
        let version = reader.u32()?;
        if version != VULKAN_STREAM_VERSION {
            return Err(VkError::InvalidEncoding(format!(
                "unsupported stream version {version}"
            )));
        }
        let width = reader.u32()?;
        let height = reader.u32()?;
        display_byte_count(width, height)
            .map_err(|error| VkError::InvalidEncoding(error.to_string()))?;
        let background = reader.rgba()?;
        if background[3] != 255 {
            return Err(VkError::InvalidEncoding(
                "background alpha must be 255".to_owned(),
            ));
        }
        let command_count = reader.u64()?;
        if command_count > MAX_VULKAN_COMMANDS as u64 {
            return Err(VkError::InvalidEncoding(
                "command count exceeds the bounded display-list stream".to_owned(),
            ));
        }
        let command_count = usize::try_from(command_count)
            .map_err(|_| VkError::InvalidEncoding("command count does not fit usize".to_owned()))?;
        let mut commands = Vec::new();
        for _ in 0..command_count {
            let tag = reader.u8()?;
            let payload_len = reader.u64()?;
            let payload_len = usize::try_from(payload_len).map_err(|_| {
                VkError::InvalidEncoding("payload length does not fit usize".to_owned())
            })?;
            if payload_len > MAX_PAYLOAD_BYTES {
                return Err(VkError::InvalidEncoding(
                    "payload length exceeds the typed command schema".to_owned(),
                ));
            }
            let payload = reader.take(payload_len)?;
            let mut payload_reader = Reader::new(payload);
            let command = decode_command(tag, &mut payload_reader)?;
            if !payload_reader.is_empty() {
                return Err(VkError::InvalidEncoding(
                    "command payload has trailing bytes".to_owned(),
                ));
            }
            commands.push(command);
        }
        if !reader.is_empty() {
            return Err(VkError::InvalidEncoding(
                "stream has trailing bytes".to_owned(),
            ));
        }
        Ok(Self {
            width,
            height,
            background,
            commands,
        })
    }
}

/// Record one display list with a fresh headless backend.
pub fn record_stream(list: &DisplayList) -> Result<Vec<VkCommand>, VkError> {
    let mut backend = VulkanBackend::new();
    backend.record(list)
}

/// Record and encode one display list with a fresh headless backend.
pub fn encode_stream(list: &DisplayList) -> Result<Vec<u8>, VkError> {
    let mut backend = VulkanBackend::new();
    backend.record(list)?;
    Ok(backend.encode())
}

/// Decode a typed command stream into a backend instance.
pub fn decode_stream(bytes: &[u8]) -> Result<VulkanBackend, VkError> {
    VulkanBackend::decode(bytes)
}

fn validate_stream(list: &DisplayList) -> Result<(Vec<u32>, Vec<Pass>), VkError> {
    display_byte_count(list.width, list.height)
        .map_err(|error| VkError::InvalidViewport(error.to_string()))?;
    if list.background[3] != 255 {
        return Err(VkError::MalformedDisplayList(
            "background alpha must be 255".to_owned(),
        ));
    }
    let command_count = i64::try_from(list.commands.len()).unwrap_or(i64::MAX);
    let command_limit = i64::try_from(MAX_COMMANDS).unwrap_or(i64::MAX);
    if list.commands.len() > MAX_COMMANDS {
        return Err(VkError::UnboundedCount {
            command: list.commands.len(),
            count: command_count,
            limit: command_limit,
        });
    }
    for (index, command) in list.commands.iter().enumerate() {
        match *command {
            Command::Solid { rect, clip, .. } => {
                validate_regular_geometry(index, rect, clip, 0, list.width, list.height)?;
            }
            Command::Rounded {
                rect, clip, radius, ..
            }
            | Command::Gradient {
                rect, clip, radius, ..
            }
            | Command::GradientHorizontal {
                rect, clip, radius, ..
            } => validate_regular_geometry(index, rect, clip, radius, list.width, list.height)?,
            Command::Particles {
                rect,
                clip,
                count,
                seed,
                radius,
                ..
            } => {
                validate_regular_geometry(index, rect, clip, 0, list.width, list.height)?;
                validate_particles(index, count, seed, radius)?;
            }
            _ => {}
        }
        validate_effect_fields(index, command, list.width, list.height)?;
    }
    // Use the exact Metal/CPU packing path as the final trust-boundary check.
    list.build_passes().map_err(VkError::from)
}

fn validate_regular_geometry(
    index: usize,
    rect: [i64; 4],
    clip: [i64; 4],
    radius: i64,
    width: u32,
    height: u32,
) -> Result<(), VkError> {
    if rect.iter().any(|value| !in_coord_range(*value))
        || radius < -COORD_LIMIT
        || radius > COORD_LIMIT
        || rect[2] <= 0
        || rect[3] <= 0
    {
        return Err(VkError::OutOfRangeRect {
            command: index,
            rect,
        });
    }
    if radius < 0 || radius > rect[2].min(rect[3]) / 2 {
        return Err(VkError::InvalidRadius {
            command: index,
            radius,
        });
    }
    if clip.iter().any(|value| !in_coord_range(*value))
        || clip[0] < 0
        || clip[1] < 0
        || clip[2] < 0
        || clip[3] < 0
        || clip[0].saturating_add(clip[2]) > i64::from(width)
        || clip[1].saturating_add(clip[3]) > i64::from(height)
    {
        return Err(VkError::InvalidClip {
            command: index,
            clip,
        });
    }
    Ok(())
}

fn validate_particles(index: usize, count: i64, seed: i64, radius: i64) -> Result<(), VkError> {
    if !(0..=PARTICLE_MAX_COUNT).contains(&count) {
        return Err(VkError::UnboundedCount {
            command: index,
            count,
            limit: PARTICLE_MAX_COUNT,
        });
    }
    if !(0..=PARTICLE_MAX_SEED).contains(&seed) {
        return Err(VkError::MalformedDisplayList(format!(
            "particle seed {seed} must be 0..={PARTICLE_MAX_SEED}"
        )));
    }
    if !(0..=PARTICLE_MAX_RADIUS).contains(&radius) {
        return Err(VkError::InvalidRadius {
            command: index,
            radius,
        });
    }
    Ok(())
}

fn validate_effect_fields(
    index: usize,
    command: &Command,
    width: u32,
    height: u32,
) -> Result<(), VkError> {
    let mut rect = [0i64; 4];
    let mut clip = [0i64; 4];
    let mut bounds = [0i64; 4];
    let mut shape_radius = 0i64;
    let mut radius = 0i64;
    let mut has_geometry = false;
    let mut has_bounds = false;
    match *command {
        Command::EffectBackdrop => {}
        Command::EffectGaussianBlur {
            radius: effect_radius,
        }
        | Command::EffectBlur {
            radius: effect_radius,
        } => radius = effect_radius,
        Command::EffectMask {
            rect: r,
            clip: c,
            bounds: b,
            shape_radius: sr,
            ..
        } => {
            rect = r;
            clip = c;
            bounds = b;
            shape_radius = sr;
            has_geometry = true;
            has_bounds = true;
        }
        Command::EffectShadow {
            bounds: b,
            radius: r,
            ..
        }
        | Command::EffectGlow {
            bounds: b,
            radius: r,
            ..
        } => {
            bounds = b;
            radius = r;
            has_bounds = true;
        }
        Command::EffectGlass {
            rect: r,
            clip: c,
            bounds: b,
            shape_radius: sr,
            radius: rad,
            ..
        } => {
            rect = r;
            clip = c;
            bounds = b;
            shape_radius = sr;
            radius = rad;
            has_geometry = true;
            has_bounds = true;
        }
        Command::EffectLiquid {
            rect: r,
            clip: c,
            bounds: b,
            shape_radius: sr,
            ..
        } => {
            rect = r;
            clip = c;
            bounds = b;
            shape_radius = sr;
            has_geometry = true;
            has_bounds = true;
        }
        _ => {}
    }
    if radius < 0 {
        return Err(VkError::InvalidRadius {
            command: index,
            radius,
        });
    }
    if shape_radius < 0 {
        return Err(VkError::InvalidRadius {
            command: index,
            radius: shape_radius,
        });
    }
    if has_geometry {
        check_effect_coords(index, rect)?;
        if rect[2] < 0 || rect[3] < 0 {
            return Err(VkError::OutOfRangeRect {
                command: index,
                rect,
            });
        }
        check_clip(index, clip, width, height)?;
    }
    if has_bounds {
        check_clip(index, bounds, width, height)?;
    }
    validate_effect(command, width, height).map_err(|error| VkError::InvalidEffect {
        command: index,
        message: error.to_string(),
    })
}

fn check_effect_coords(index: usize, rect: [i64; 4]) -> Result<(), VkError> {
    if rect.iter().any(|value| !in_coord_range(*value)) {
        return Err(VkError::OutOfRangeRect {
            command: index,
            rect,
        });
    }
    Ok(())
}

fn check_clip(index: usize, values: [i64; 4], width: u32, height: u32) -> Result<(), VkError> {
    if values.iter().any(|value| !in_coord_range(*value))
        || values[0] < 0
        || values[1] < 0
        || values[2] < 0
        || values[3] < 0
        || values[0].saturating_add(values[2]) > i64::from(width)
        || values[1].saturating_add(values[3]) > i64::from(height)
    {
        return Err(VkError::InvalidClip {
            command: index,
            clip: values,
        });
    }
    Ok(())
}

fn in_coord_range(value: i64) -> bool {
    (-COORD_LIMIT..=COORD_LIMIT).contains(&value)
}

fn dispatch_kernel(effect: VkEffect) -> u32 {
    match effect {
        VkEffect::Backdrop => KERNEL_SNAPSHOT,
        VkEffect::Mask => KERNEL_MASK,
        VkEffect::Shadow | VkEffect::Glow | VkEffect::Glass | VkEffect::Liquid => KERNEL_EFFECT,
        VkEffect::BlurHorizontal => KERNEL_BLUR_H,
        VkEffect::BlurVertical => KERNEL_BLUR_V,
        VkEffect::GaussianHorizontal => KERNEL_GAUSSIAN_H,
        VkEffect::GaussianVertical => KERNEL_GAUSSIAN_V,
    }
}

fn effect_from_pass(pass: &Pass) -> Result<VkEffect, VkError> {
    let effect = match pass.kernel {
        KERNEL_SNAPSHOT => VkEffect::Backdrop,
        KERNEL_MASK => VkEffect::Mask,
        KERNEL_EFFECT => match pass.params[2] {
            0 => VkEffect::Shadow,
            1 => VkEffect::Glow,
            2 => VkEffect::Glass,
            3 => VkEffect::Liquid,
            other => {
                return Err(VkError::MalformedDisplayList(format!(
                    "unknown local effect kind {other}"
                )))
            }
        },
        KERNEL_BLUR_H => VkEffect::BlurHorizontal,
        KERNEL_BLUR_V => VkEffect::BlurVertical,
        KERNEL_GAUSSIAN_H => VkEffect::GaussianHorizontal,
        KERNEL_GAUSSIAN_V => VkEffect::GaussianVertical,
        other => {
            return Err(VkError::MalformedDisplayList(format!(
                "unknown Metal pass kernel {other}"
            )))
        }
    };
    Ok(effect)
}

fn draw_from_words(words: &[u32], command: usize) -> Result<VkDraw, VkError> {
    if words.len() != COMMAND_WORDS {
        return Err(VkError::MalformedDisplayList(format!(
            "command {command} has {} words, expected {COMMAND_WORDS}",
            words.len()
        )));
    }
    let pipeline = VkPipeline::from_display_kind(words[0]).ok_or_else(|| {
        VkError::MalformedDisplayList(format!(
            "command {command} has unknown packed kind {}",
            words[0]
        ))
    })?;
    let coord = |word: u32| i64::from(word) - COORD_BIAS;
    let rect_raw = [
        coord(words[1]),
        coord(words[2]),
        coord(words[3]),
        coord(words[4]),
    ];
    let clip_raw = [
        coord(words[5]),
        coord(words[6]),
        coord(words[7]),
        coord(words[8]),
    ];
    let rect = vk_rect(rect_raw, command, false)?;
    let scissor = vk_rect(clip_raw, command, true)?;
    let corner_radius = coord(words[9]);
    let color = words[10].to_le_bytes();
    let (primitive, vertex_count, radius, second_color, alpha, count, seed) = match pipeline {
        VkPipeline::Gradient | VkPipeline::GradientHorizontal => {
            let alpha_u8 = u8::try_from(words[12]).map_err(|_| {
                VkError::MalformedDisplayList(format!(
                    "command {command} gradient alpha exceeds 255"
                ))
            })?;
            (
                VkPrimitive::TriangleList,
                6,
                u32::try_from(corner_radius).map_err(|_| VkError::InvalidRadius {
                    command,
                    radius: corner_radius,
                })?,
                Some(words[11].to_le_bytes()),
                alpha_u8,
                0,
                0,
            )
        }
        VkPipeline::Particles => (
            VkPrimitive::PointList,
            words[11],
            words[13],
            None,
            color[3],
            words[11],
            words[12],
        ),
        _ => (
            VkPrimitive::TriangleList,
            6,
            u32::try_from(corner_radius).map_err(|_| VkError::InvalidRadius {
                command,
                radius: corner_radius,
            })?,
            None,
            color[3],
            0,
            0,
        ),
    };
    Ok(VkDraw {
        pipeline,
        primitive,
        vertex_count,
        instance_count: 1,
        rect,
        scissor,
        radius,
        color,
        second_color,
        alpha,
        count,
        seed,
    })
}

fn vk_rect(values: [i64; 4], command: usize, is_clip: bool) -> Result<VkRect2D, VkError> {
    let conversion_error = || {
        if is_clip {
            VkError::InvalidClip {
                command,
                clip: values,
            }
        } else {
            VkError::OutOfRangeRect {
                command,
                rect: values,
            }
        }
    };
    let x = i32::try_from(values[0]).map_err(|_| conversion_error())?;
    let y = i32::try_from(values[1]).map_err(|_| conversion_error())?;
    let width = u32::try_from(values[2]).map_err(|_| conversion_error())?;
    let height = u32::try_from(values[3]).map_err(|_| conversion_error())?;
    Ok(VkRect2D {
        x,
        y,
        width,
        height,
    })
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_rgba(out: &mut Vec<u8>, color: [u8; 4]) {
    out.extend_from_slice(&color);
}

fn put_rect(out: &mut Vec<u8>, rect: VkRect2D) {
    put_i32(out, rect.x);
    put_i32(out, rect.y);
    put_u32(out, rect.width);
    put_u32(out, rect.height);
}

fn put_pipeline(out: &mut Vec<u8>, pipeline: VkPipeline) {
    put_u8(out, pipeline.tag());
    if let VkPipeline::Effect(effect) = pipeline {
        put_u8(out, effect.tag());
    }
}

fn encode_payload(command: VkCommand) -> Vec<u8> {
    let mut out = Vec::new();
    match command {
        VkCommand::BindPipeline { pipeline } => put_pipeline(&mut out, pipeline),
        VkCommand::SetScissor { rect } => put_rect(&mut out, rect),
        VkCommand::Draw(draw) => {
            put_pipeline(&mut out, draw.pipeline);
            put_u8(&mut out, draw.primitive.tag());
            put_u32(&mut out, draw.vertex_count);
            put_u32(&mut out, draw.instance_count);
            put_rect(&mut out, draw.rect);
            put_rect(&mut out, draw.scissor);
            put_u32(&mut out, draw.radius);
            put_rgba(&mut out, draw.color);
            match draw.second_color {
                Some(color) => {
                    put_u8(&mut out, 1);
                    put_rgba(&mut out, color);
                }
                None => put_u8(&mut out, 0),
            }
            put_u8(&mut out, draw.alpha);
            put_u32(&mut out, draw.count);
            put_u32(&mut out, draw.seed);
        }
        VkCommand::CopyBufferToImage(copy) => {
            put_u32(&mut out, copy.image_id);
            put_u64(&mut out, copy.buffer_offset);
            put_u32(&mut out, copy.bytes_per_row);
            put_u32(&mut out, copy.bytes_per_image);
            put_u32(&mut out, copy.extent[0]);
            put_u32(&mut out, copy.extent[1]);
            put_rgba(&mut out, copy.color);
        }
        VkCommand::Dispatch(dispatch) => {
            put_u8(&mut out, dispatch.effect.tag());
            put_u32(&mut out, dispatch.kernel);
            for group in dispatch.group_count {
                put_u32(&mut out, group);
            }
            for param in dispatch.params {
                put_u32(&mut out, param);
            }
        }
    }
    out
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], VkError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| VkError::InvalidEncoding("stream byte offset overflow".to_owned()))?;
        let bytes = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| VkError::InvalidEncoding("unexpected end of stream".to_owned()))?;
        self.pos = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, VkError> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| VkError::InvalidEncoding("expected a byte".to_owned()))
    }

    fn u32(&mut self) -> Result<u32, VkError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| VkError::InvalidEncoding("bad u32 field".to_owned()))?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, VkError> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| VkError::InvalidEncoding("bad u64 field".to_owned()))?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn i32(&mut self) -> Result<i32, VkError> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| VkError::InvalidEncoding("bad i32 field".to_owned()))?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn rgba(&mut self) -> Result<[u8; 4], VkError> {
        self.take(4)?
            .try_into()
            .map_err(|_| VkError::InvalidEncoding("bad RGBA8 field".to_owned()))
    }

    fn rect(&mut self) -> Result<VkRect2D, VkError> {
        Ok(VkRect2D {
            x: self.i32()?,
            y: self.i32()?,
            width: self.u32()?,
            height: self.u32()?,
        })
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }
}

fn decode_pipeline(reader: &mut Reader<'_>) -> Result<VkPipeline, VkError> {
    let pipeline = match reader.u8()? {
        0 => VkPipeline::ClearImage,
        1 => VkPipeline::Solid,
        2 => VkPipeline::Rounded,
        3 => VkPipeline::Gradient,
        4 => VkPipeline::GradientHorizontal,
        5 => VkPipeline::Particles,
        6 => {
            let tag = reader.u8()?;
            let effect = VkEffect::from_tag(tag).ok_or_else(|| {
                VkError::InvalidEncoding(format!("unknown effect pipeline tag {tag}"))
            })?;
            VkPipeline::Effect(effect)
        }
        tag => {
            return Err(VkError::InvalidEncoding(format!(
                "unknown pipeline tag {tag}"
            )))
        }
    };
    Ok(pipeline)
}

fn decode_optional_rgba(reader: &mut Reader<'_>) -> Result<Option<[u8; 4]>, VkError> {
    match reader.u8()? {
        0 => Ok(None),
        1 => Ok(Some(reader.rgba()?)),
        tag => Err(VkError::InvalidEncoding(format!(
            "unknown optional colour tag {tag}"
        ))),
    }
}

fn decode_command(tag: u8, reader: &mut Reader<'_>) -> Result<VkCommand, VkError> {
    let command = match tag {
        TAG_BIND_PIPELINE => VkCommand::BindPipeline {
            pipeline: decode_pipeline(reader)?,
        },
        TAG_SET_SCISSOR => VkCommand::SetScissor {
            rect: reader.rect()?,
        },
        TAG_DRAW => {
            let pipeline = decode_pipeline(reader)?;
            let primitive_tag = reader.u8()?;
            let primitive = VkPrimitive::from_tag(primitive_tag).ok_or_else(|| {
                VkError::InvalidEncoding(format!("unknown primitive tag {primitive_tag}"))
            })?;
            VkCommand::Draw(VkDraw {
                pipeline,
                primitive,
                vertex_count: reader.u32()?,
                instance_count: reader.u32()?,
                rect: reader.rect()?,
                scissor: reader.rect()?,
                radius: reader.u32()?,
                color: reader.rgba()?,
                second_color: decode_optional_rgba(reader)?,
                alpha: reader.u8()?,
                count: reader.u32()?,
                seed: reader.u32()?,
            })
        }
        TAG_COPY_BUFFER_TO_IMAGE => VkCommand::CopyBufferToImage(VkCopyBufferToImage {
            image_id: reader.u32()?,
            buffer_offset: reader.u64()?,
            bytes_per_row: reader.u32()?,
            bytes_per_image: reader.u32()?,
            extent: [reader.u32()?, reader.u32()?],
            color: reader.rgba()?,
        }),
        TAG_DISPATCH => {
            let effect_tag = reader.u8()?;
            let effect = VkEffect::from_tag(effect_tag).ok_or_else(|| {
                VkError::InvalidEncoding(format!("unknown dispatch effect tag {effect_tag}"))
            })?;
            let kernel = reader.u32()?;
            let mut group_count = [0u32; 3];
            for group in &mut group_count {
                *group = reader.u32()?;
            }
            let mut params = [0u32; PARAM_WORDS];
            for param in &mut params {
                *param = reader.u32()?;
            }
            VkCommand::Dispatch(VkDispatch {
                effect,
                kernel,
                group_count,
                params,
            })
        }
        other => {
            return Err(VkError::InvalidEncoding(format!(
                "unknown command tag {other}"
            )))
        }
    };
    Ok(command)
}
