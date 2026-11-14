//! Standard-library parser, packer and CPU unit oracle for display lists and
//! effect commands.
//!
//! This module never substitutes for Metal. `raster` returns MetalUnavailable
//! off macOS; the CPU interpreter below covers regular commands only and is for
//! parser/kernel unit tests, not the cross-repo equality proof.

use crate::{GpuError, MAX_DIMENSION};
use std::str;

pub const MAX_COMMANDS: usize = 1_000_000;
pub const COMMAND_WORDS: usize = 16;
/// Signed coordinates are encoded as `value + COORD_BIAS` for MSL's `int`.
pub const COORD_BIAS: i64 = 1 << 30;
pub const COORD_LIMIT: i64 = 500_000_000;

/// Effect parameter bounds, shared with opurom-render's effect schema.
pub const EFFECT_MAX_RADIUS: i64 = 32;
pub const EFFECT_MAX_OFFSET: i64 = 64;
pub const EFFECT_UNIT: i64 = 1024;

/// Bounds for the separable Gaussian blur command. Its binomial weights are
/// packed into words 4..4+2r of the 28-word param block; r <= 8 keeps the
/// kernel compact with a small power-of-two weight sum.
pub const GAUSSIAN_MIN_RADIUS: i64 = 1;
pub const GAUSSIAN_MAX_RADIUS: i64 = 8;

/// Particle parameter bounds, shared with opurom-render's particle schema.
pub const PARTICLE_MAX_COUNT: i64 = 512;
pub const PARTICLE_MAX_RADIUS: i64 = 32;
pub const PARTICLE_MAX_SEED: i64 = u32::MAX as i64;

/// Fixed-size parameter block passed to every Metal compute pass.
pub const PARAM_WORDS: usize = 28;
pub const KERNEL_BATCH: u32 = 0;
pub const KERNEL_SNAPSHOT: u32 = 1;
pub const KERNEL_MASK: u32 = 2;
pub const KERNEL_EFFECT: u32 = 3;
pub const KERNEL_BLUR_H: u32 = 4;
pub const KERNEL_BLUR_V: u32 = 5;
pub const KERNEL_GAUSSIAN_H: u32 = 6;
pub const KERNEL_GAUSSIAN_V: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    pub kernel: u32,
    pub params: [u32; PARAM_WORDS],
}

const UNIT: i64 = 1024;
pub(crate) const SOLID: u32 = 0;
pub(crate) const ROUNDED: u32 = 1;
pub(crate) const GRADIENT: u32 = 2;
pub(crate) const GRADIENT_HORIZONTAL: u32 = 3;
pub(crate) const PARTICLES: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Solid {
        rect: [i64; 4],
        clip: [i64; 4],
        color: [u8; 4],
    },
    Rounded {
        rect: [i64; 4],
        clip: [i64; 4],
        radius: i64,
        color: [u8; 4],
    },
    Gradient {
        rect: [i64; 4],
        clip: [i64; 4],
        radius: i64,
        top: [u8; 4],
        bottom: [u8; 4],
        alpha: u8,
    },
    GradientHorizontal {
        rect: [i64; 4],
        clip: [i64; 4],
        radius: i64,
        top: [u8; 4],
        bottom: [u8; 4],
        alpha: u8,
    },
    /// Deterministic particle field: `count` integer dots scattered over the
    /// rect by a seeded pure integer hash, blended in dot order.
    Particles {
        rect: [i64; 4],
        clip: [i64; 4],
        count: i64,
        seed: i64,
        radius: i64,
        color: [u8; 4],
    },
    /// Snapshot the frame before a node's ordinary draw commands.
    EffectBackdrop,
    /// Snapshot the node's silhouette before any local effect runs.
    EffectMask {
        rect: [i64; 4],
        clip: [i64; 4],
        bounds: [i64; 4],
        shape_radius: i64,
        shape: bool,
    },
    EffectShadow {
        bounds: [i64; 4],
        radius: i64,
        dx: i64,
        dy: i64,
        alpha: i64,
        color: [u8; 4],
    },
    EffectGlow {
        bounds: [i64; 4],
        radius: i64,
        alpha: i64,
        color: [u8; 4],
    },
    EffectGlass {
        rect: [i64; 4],
        clip: [i64; 4],
        bounds: [i64; 4],
        shape_radius: i64,
        radius: i64,
        alpha: i64,
        color: [u8; 4],
    },
    EffectLiquid {
        rect: [i64; 4],
        clip: [i64; 4],
        bounds: [i64; 4],
        shape_radius: i64,
        denom: i64,
        alpha: i64,
        color: [u8; 4],
    },
    /// Whole-frame box blur, applied after every node in layout order.
    EffectBlur { radius: i64 },
    /// Whole-frame separable Gaussian (binomial) blur over a bounded radius.
    /// Flutter's `BoxShadow.toPaint` uses a Gaussian `MaskFilter.blur`; this
    /// bell-shaped kernel avoids the ringing of the uniform box blur.
    EffectGaussianBlur { radius: i64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayList {
    pub width: u32,
    pub height: u32,
    pub background: [u8; 4],
    pub commands: Vec<Command>,
}

impl Command {
    /// Regular commands only; effect commands carry no raster geometry.
    fn geometry(&self) -> Option<([i64; 4], [i64; 4], i64)> {
        Some(match *self {
            Command::Solid { rect, clip, .. } => (rect, clip, 0),
            Command::Rounded {
                rect, clip, radius, ..
            }
            | Command::Gradient {
                rect, clip, radius, ..
            }
            | Command::GradientHorizontal {
                rect, clip, radius, ..
            } => (rect, clip, radius),
            Command::Particles { rect, clip, .. } => (rect, clip, 0),
            _ => return None,
        })
    }

    fn sample(&self, x: i64, y: i64) -> Option<[u8; 4]> {
        // Particles blend dot by dot in `blend_at`; `sample` covers the four
        // single-source rect kinds.
        if matches!(self, Command::Particles { .. }) {
            return None;
        }
        let (rect, clip, radius) = self.geometry()?;
        if x < rect[0] || y < rect[1] || x >= rect[0] + rect[2] || y >= rect[1] + rect[3] {
            return None;
        }
        if x < clip[0] || y < clip[1] || x >= clip[0] + clip[2] || y >= clip[1] + clip[3] {
            return None;
        }
        if radius > 0 && outside_round(rect, radius, x, y) {
            return None;
        }
        match *self {
            Command::Solid { color, .. } | Command::Rounded { color, .. } => Some(color),
            Command::Gradient {
                rect,
                top,
                bottom,
                alpha,
                ..
            }
            | Command::GradientHorizontal {
                rect,
                top,
                bottom,
                alpha,
                ..
            } => {
                let (last, local) = if matches!(*self, Command::GradientHorizontal { .. }) {
                    (rect[2] - 1, x - rect[0])
                } else {
                    (rect[3] - 1, y - rect[1])
                };
                let t = if last <= 0 {
                    UNIT
                } else {
                    (local * UNIT) / last
                };
                Some([
                    mix_channel(top[0], bottom[0], t),
                    mix_channel(top[1], bottom[1], t),
                    mix_channel(top[2], bottom[2], t),
                    alpha,
                ])
            }
            _ => unreachable!("effect commands are not per-pixel samples"),
        }
    }

    /// Blend one command into `pixel`. Regular commands sample a single source
    /// colour; particles blend every covering dot in index order, which is the
    /// same order the MSL kernel uses.
    fn blend_at(&self, pixel: &mut [u8], x: i64, y: i64) {
        if let Command::Particles {
            rect,
            clip,
            count,
            seed,
            radius,
            color,
        } = self
        {
            if x < rect[0] || y < rect[1] || x >= rect[0] + rect[2] || y >= rect[1] + rect[3] {
                return;
            }
            if x < clip[0] || y < clip[1] || x >= clip[0] + clip[2] || y >= clip[1] + clip[3] {
                return;
            }
            let rad = (*radius).max(0);
            let limit = rad * rad;
            for index in 0..*count {
                let (cx, cy) = particle_center(*rect, *seed, index as u32);
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= limit {
                    blend(pixel, *color);
                }
            }
            return;
        }
        if let Some(src) = self.sample(x, y) {
            blend(pixel, src);
        }
    }

    fn pack_regular(&self) -> Result<[u32; COMMAND_WORDS], GpuError> {
        let (rect, clip, radius) = self.geometry().expect("regular command");
        let mut packed = [0u32; COMMAND_WORDS];
        packed[0] = match self {
            Command::Solid { .. } => SOLID,
            Command::Rounded { .. } => ROUNDED,
            Command::Gradient { .. } => GRADIENT,
            Command::GradientHorizontal { .. } => GRADIENT_HORIZONTAL,
            Command::Particles { .. } => PARTICLES,
            _ => unreachable!("regular command"),
        };
        for (offset, value) in rect.iter().enumerate() {
            packed[1 + offset] = encode_coord(*value, "rect")?;
        }
        for (offset, value) in clip.iter().enumerate() {
            packed[5 + offset] = encode_coord(*value, "clip")?;
        }
        packed[9] = encode_coord(radius, "radius")?;
        match *self {
            Command::Solid { color, .. } | Command::Rounded { color, .. } => {
                packed[10] = pack_rgba(color);
            }
            Command::Gradient {
                top, bottom, alpha, ..
            }
            | Command::GradientHorizontal {
                top, bottom, alpha, ..
            } => {
                packed[10] = pack_rgba(top);
                packed[11] = pack_rgba(bottom);
                packed[12] = u32::from(alpha);
            }
            Command::Particles {
                color,
                count,
                seed,
                radius,
                ..
            } => {
                packed[10] = pack_rgba(color);
                // Words 11..=13 are raw integers, not biased coordinates; the
                // MSL kernel reads them directly.
                packed[11] = count as u32;
                packed[12] = seed as u32;
                packed[13] = radius as u32;
            }
            _ => unreachable!("regular command"),
        }
        Ok(packed)
    }
}

fn base_params(width: u32, height: u32) -> [u32; PARAM_WORDS] {
    let mut params = [0u32; PARAM_WORDS];
    params[0] = width;
    params[1] = height;
    params
}

fn put4(params: &mut [u32; PARAM_WORDS], offset: usize, values: [i64; 4]) -> Result<(), GpuError> {
    for (index, value) in values.iter().enumerate() {
        params[offset + index] = encode_coord(*value, "effect rect")?;
    }
    Ok(())
}

/// One or two Metal passes for a single effect command.
fn effect_passes(command: &Command, width: u32, height: u32) -> Result<Vec<Pass>, GpuError> {
    let one = |kernel, params| Ok(vec![Pass { kernel, params }]);
    let mut p = base_params(width, height);
    match *command {
        Command::EffectBackdrop => one(KERNEL_SNAPSHOT, p),
        Command::EffectGaussianBlur { radius } => {
            let (sum, weights) = gaussian_weights(radius);
            p[2] = radius as u32;
            p[3] = sum as u32;
            for j in 0..weights.len() {
                p[4 + j] = weights[j] as u32;
            }
            Ok(vec![
                Pass {
                    kernel: KERNEL_GAUSSIAN_H,
                    params: p,
                },
                Pass {
                    kernel: KERNEL_GAUSSIAN_V,
                    params: p,
                },
            ])
        }
        Command::EffectBlur { radius } => {
            p[2] = radius as u32;
            Ok(vec![
                Pass {
                    kernel: KERNEL_BLUR_H,
                    params: p,
                },
                Pass {
                    kernel: KERNEL_BLUR_V,
                    params: p,
                },
            ])
        }
        Command::EffectMask {
            rect,
            clip,
            bounds,
            shape_radius,
            shape,
        } => {
            p[11] = shape_radius as u32;
            p[25] = u32::from(shape);
            put4(&mut p, 13, bounds)?;
            put4(&mut p, 17, rect)?;
            put4(&mut p, 21, clip)?;
            one(KERNEL_MASK, p)
        }
        Command::EffectShadow {
            bounds,
            radius,
            dx,
            dy,
            alpha,
            color,
        } => {
            p[2] = 0;
            p[3] = pack_rgba(color);
            p[7] = alpha as u32;
            p[8] = radius as u32;
            p[9] = encode_coord(dx, "dx")?;
            p[10] = encode_coord(dy, "dy")?;
            put4(&mut p, 13, bounds)?;
            one(KERNEL_EFFECT, p)
        }
        Command::EffectGlow {
            bounds,
            radius,
            alpha,
            color,
        } => {
            p[2] = 1;
            p[3] = pack_rgba(color);
            p[7] = alpha as u32;
            p[8] = radius as u32;
            put4(&mut p, 13, bounds)?;
            one(KERNEL_EFFECT, p)
        }
        Command::EffectGlass {
            rect,
            clip,
            bounds,
            shape_radius,
            radius,
            alpha,
            color,
        } => {
            p[2] = 2;
            p[3] = pack_rgba(color);
            p[7] = alpha as u32;
            p[8] = radius as u32;
            p[11] = shape_radius as u32;
            put4(&mut p, 13, bounds)?;
            put4(&mut p, 17, rect)?;
            put4(&mut p, 21, clip)?;
            one(KERNEL_EFFECT, p)
        }
        Command::EffectLiquid {
            rect,
            clip,
            bounds,
            shape_radius,
            denom,
            alpha,
            color,
        } => {
            p[2] = 3;
            p[3] = pack_rgba(color);
            p[7] = alpha as u32;
            p[11] = shape_radius as u32;
            p[12] = denom as u32;
            put4(&mut p, 13, bounds)?;
            put4(&mut p, 17, rect)?;
            put4(&mut p, 21, clip)?;
            one(KERNEL_EFFECT, p)
        }
        _ => unreachable!("regular commands are batches"),
    }
}

impl DisplayList {
    /// Public fields bypass the parser, so recheck bounds on every execution path.
    fn validate(&self) -> Result<(), GpuError> {
        for command in &self.commands {
            if let Some((rect, clip, radius)) = command.geometry() {
                validate_geometry(rect, clip, radius, self.width, self.height)?;
                validate_gradient_stops(command)?;
            } else {
                validate_effect(command, self.width, self.height)?;
            }
            if let Command::Particles {
                count,
                seed,
                radius,
                ..
            } = command
            {
                validate_particles(*count, *seed, *radius)?;
            }
        }
        Ok(())
    }

    /// Pack regular commands and build the ordered Metal pass list.
    pub fn build_passes(&self) -> Result<(Vec<u32>, Vec<Pass>), GpuError> {
        self.validate()?;
        let mut words = Vec::with_capacity(self.commands.len() * COMMAND_WORDS);
        let background = pack_rgba(self.background);
        let mut clear = base_params(self.width, self.height);
        clear[2] = 1;
        clear[4] = background;
        let mut passes = vec![Pass {
            kernel: KERNEL_BATCH,
            params: clear,
        }];
        let mut batch_start = 0u32;
        let mut batch_count = 0u32;
        let batch_pass = |start: u32, count: u32| Pass {
            kernel: KERNEL_BATCH,
            params: {
                let mut p = base_params(self.width, self.height);
                p[4] = background;
                p[5] = start;
                p[6] = count;
                p
            },
        };
        for command in &self.commands {
            if command.geometry().is_some() {
                if words.len() / COMMAND_WORDS >= MAX_COMMANDS {
                    return Err(GpuError::InvalidDisplayList(
                        "too many regular commands".to_owned(),
                    ));
                }
                words.extend_from_slice(&command.pack_regular()?);
                batch_count += 1;
            } else {
                if batch_count > 0 {
                    passes.push(batch_pass(batch_start, batch_count));
                    batch_count = 0;
                }
                batch_start = words.len() as u32;
                passes.extend(effect_passes(command, self.width, self.height)?);
            }
        }
        if batch_count > 0 {
            passes.push(batch_pass(batch_start, batch_count));
        }
        Ok((words, passes))
    }

    pub fn pack(&self) -> Result<Vec<u32>, GpuError> {
        Ok(self.build_passes()?.0)
    }
}

pub(crate) fn pack_rgba(color: [u8; 4]) -> u32 {
    u32::from(color[0])
        | (u32::from(color[1]) << 8)
        | (u32::from(color[2]) << 16)
        | (u32::from(color[3]) << 24)
}

fn encode_coord(value: i64, what: &str) -> Result<u32, GpuError> {
    if !(-COORD_LIMIT..=COORD_LIMIT).contains(&value) {
        return Err(GpuError::InvalidDisplayList(format!(
            "{what} coordinate {value} exceeds the integer range"
        )));
    }
    Ok((value + COORD_BIAS) as u32)
}

fn mix_channel(from: u8, to: u8, t: i64) -> u8 {
    let value = ((i64::from(to) - i64::from(from)) * t + UNIT / 2) / UNIT + i64::from(from);
    value.clamp(0, 255) as u8
}

fn outside_round(rect: [i64; 4], rad: i64, x: i64, y: i64) -> bool {
    if rad <= 0 {
        return false;
    }
    let cr = rad - 1;
    let limit = cr * cr;
    let dx = x - rect[0];
    let cx = if dx < rad {
        cr - dx
    } else if dx >= rect[2] - rad {
        dx - (rect[2] - rad)
    } else {
        -1
    };
    let dy = y - rect[1];
    let cy = if dy < rad {
        cr - dy
    } else if dy >= rect[3] - rad {
        dy - (rect[3] - rad)
    } else {
        -1
    };
    cx >= 0 && cy >= 0 && cx * cx + cy * cy > limit
}

fn blend(pixel: &mut [u8], src: [u8; 4]) {
    if src[3] == 0 {
        return;
    }
    if src[3] == 255 {
        pixel[0] = src[0];
        pixel[1] = src[1];
        pixel[2] = src[2];
        pixel[3] = 255;
        return;
    }
    let a = u32::from(src[3]);
    let inv = 255 - a;
    for k in 0..3 {
        let dst = u32::from(pixel[k]);
        pixel[k] = ((u32::from(src[k]) * a + dst * inv) / 255) as u8;
    }
    pixel[3] = 255;
}

/// Pure-Rust unit oracle for regular commands only. Effect commands are
/// executed by Metal; the real `raster` never calls this on a non-Metal host.
pub fn render_cpu_commands(list: &DisplayList) -> Result<Vec<u8>, GpuError> {
    let byte_count = crate::display_byte_count(list.width, list.height)?;
    list.validate()?;
    let mut pixels = Vec::with_capacity(byte_count);
    for _ in 0..(u64::from(list.width) * u64::from(list.height)) {
        pixels.extend_from_slice(&[
            list.background[0],
            list.background[1],
            list.background[2],
            255,
        ]);
    }
    for y in 0..list.height {
        for x in 0..list.width {
            let offset = (y as usize * list.width as usize + x as usize) * 4;
            for command in &list.commands {
                command.blend_at(&mut pixels[offset..offset + 4], i64::from(x), i64::from(y));
            }
        }
    }
    Ok(pixels)
}

/// Binomial row `C(2r, 0..=2r)`: an integer, deterministic discrete-Gaussian
/// kernel whose weights sum to `2^(2r)`. Both the CPU reference and the packed
/// Metal params derive from this one function, so their weights are identical.
fn gaussian_weights(radius: i64) -> (i64, Vec<i64>) {
    let n = 2 * radius;
    let mut weights = vec![1i64; (n + 1) as usize];
    for k in 1..=n as usize {
        weights[k] = weights[k - 1] * (n - k as i64 + 1) / k as i64;
    }
    let sum: i64 = weights.iter().sum();
    (sum, weights)
}

/// Integer CPU reference for the two separable Metal Gaussian passes. The
/// horizontal then vertical 1D convolutions use the same binomial weights,
/// clamped edges and `(sum + W/2)/W` half-up rounding as `gaussian_horizontal`
/// and `gaussian_vertical`, so the returned RGBA bytes match the GPU readback.
pub fn gaussian_blur_cpu(
    pixels: &[u8],
    width: u32,
    height: u32,
    radius: i64,
) -> Result<Vec<u8>, GpuError> {
    effect_range(
        radius,
        GAUSSIAN_MIN_RADIUS,
        GAUSSIAN_MAX_RADIUS,
        "gaussian radius",
    )?;
    let byte_count = crate::display_byte_count(width, height)?;
    if pixels.len() != byte_count {
        return Err(GpuError::InvalidParameters(format!(
            "RGBA length {} does not match {byte_count}",
            pixels.len()
        )));
    }
    let (sum, weights) = gaussian_weights(radius);
    let (w, h, r) = (width as i64, height as i64, radius);
    let taps = 2 * r;
    let mut horizontal = vec![0u8; byte_count];
    for y in 0..h {
        for x in 0..w {
            for k in 0..4i64 {
                let mut acc = 0i64;
                for j in 0..=taps {
                    let sx = (x + j - r).clamp(0, w - 1);
                    let idx = ((y * w + sx) * 4 + k) as usize;
                    acc += i64::from(pixels[idx]) * weights[j as usize];
                }
                let out = (acc + sum / 2) / sum;
                horizontal[((y * w + x) * 4 + k) as usize] = out.clamp(0, 255) as u8;
            }
        }
    }
    let mut blurred = vec![0u8; byte_count];
    for y in 0..h {
        for x in 0..w {
            for k in 0..4i64 {
                let mut acc = 0i64;
                for j in 0..=taps {
                    let sy = (y + j - r).clamp(0, h - 1);
                    let idx = ((sy * w + x) * 4 + k) as usize;
                    acc += i64::from(horizontal[idx]) * weights[j as usize];
                }
                let out = (acc + sum / 2) / sum;
                blurred[((y * w + x) * 4 + k) as usize] = out.clamp(0, 255) as u8;
            }
        }
    }
    Ok(blurred)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn error(&self, message: impl Into<String>) -> GpuError {
        GpuError::InvalidDisplayList(format!("{} at byte {}", message.into(), self.pos))
    }

    fn whitespace(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Result<u8, GpuError> {
        self.whitespace();
        self.bytes
            .get(self.pos)
            .copied()
            .ok_or_else(|| self.error("unexpected end of JSON"))
    }

    fn expect(&mut self, expected: u8, what: &str) -> Result<(), GpuError> {
        let got = self.peek()?;
        if got != expected {
            return Err(self.error(format!(
                "expected '{}' {what}, found '{}'",
                expected as char, got as char
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn string(&mut self) -> Result<String, GpuError> {
        self.expect(b'"', "in string")?;
        let start = self.pos;
        while let Some(&byte) = self.bytes.get(self.pos) {
            if byte == b'"' {
                let text = str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|_| self.error("string is not UTF-8"))?
                    .to_owned();
                self.pos += 1;
                return Ok(text);
            }
            if byte == b'\\' || byte.is_ascii_control() {
                return Err(self.error("unsupported string escape"));
            }
            self.pos += 1;
        }
        Err(self.error("unterminated string"))
    }

    fn key(&mut self, expected: &str) -> Result<(), GpuError> {
        let got = self.string()?;
        if got != expected {
            return Err(self.error(format!("expected key {expected:?}, found {got:?}")));
        }
        self.expect(b':', &format!("after key {expected:?}"))
    }

    fn integer(&mut self) -> Result<i64, GpuError> {
        self.whitespace();
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        let digits = self.pos;
        while self.bytes.get(self.pos).is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == digits {
            return Err(self.error("expected an integer"));
        }
        str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error("integer is not UTF-8"))?
            .parse::<i64>()
            .map_err(|_| self.error("integer is out of i64 range"))
    }

    fn byte(&mut self, what: &str) -> Result<u8, GpuError> {
        let value = self.integer()?;
        if !(0..=255).contains(&value) {
            return Err(self.error(format!("{what} must be 0..=255")));
        }
        Ok(value as u8)
    }

    fn dimension(&mut self, what: &str) -> Result<u32, GpuError> {
        let value = self.integer()?;
        if !(1..=i64::from(MAX_DIMENSION)).contains(&value) {
            return Err(self.error(format!("{what} must be 1..={MAX_DIMENSION}")));
        }
        Ok(value as u32)
    }

    fn i64_array(&mut self, what: &str) -> Result<[i64; 4], GpuError> {
        self.expect(b'[', &format!("in {what}"))?;
        let mut values = [0i64; 4];
        for (index, slot) in values.iter_mut().enumerate() {
            if index > 0 {
                self.expect(b',', &format!("in {what}"))?;
            }
            *slot = self.integer()?;
        }
        self.expect(b']', &format!("after {what}"))?;
        Ok(values)
    }

    fn u8_array(&mut self, what: &str) -> Result<[u8; 4], GpuError> {
        self.expect(b'[', &format!("in {what}"))?;
        let mut values = [0u8; 4];
        for (index, slot) in values.iter_mut().enumerate() {
            if index > 0 {
                self.expect(b',', &format!("in {what}"))?;
            }
            *slot = self.byte(what)?;
        }
        self.expect(b']', &format!("after {what}"))?;
        Ok(values)
    }

    fn finish(&mut self) -> Result<(), GpuError> {
        self.whitespace();
        if self.pos != self.bytes.len() {
            return Err(self.error("unexpected trailing data"));
        }
        Ok(())
    }
}

fn validate_geometry(
    rect: [i64; 4],
    clip: [i64; 4],
    radius: i64,
    width: u32,
    height: u32,
) -> Result<(), GpuError> {
    for value in rect.iter().chain(clip.iter()).chain([radius].iter()) {
        if !(-COORD_LIMIT..=COORD_LIMIT).contains(value) {
            return Err(GpuError::InvalidDisplayList(format!(
                "coordinate {value} exceeds the integer range"
            )));
        }
    }
    if rect[2] <= 0 || rect[3] <= 0 {
        return Err(GpuError::InvalidDisplayList(
            "command rect must have positive width and height".to_owned(),
        ));
    }
    if radius < 0 || radius > rect[2].min(rect[3]) / 2 {
        return Err(GpuError::InvalidDisplayList(
            "radius exceeds the rounded-rect bounds".to_owned(),
        ));
    }
    if clip[0] < 0 || clip[1] < 0 || clip[2] < 0 || clip[3] < 0 {
        return Err(GpuError::InvalidDisplayList(
            "clip must be normalized to the viewport".to_owned(),
        ));
    }
    if clip[0].saturating_add(clip[2]) > i64::from(width)
        || clip[1].saturating_add(clip[3]) > i64::from(height)
    {
        return Err(GpuError::InvalidDisplayList(
            "clip extends outside the viewport".to_owned(),
        ));
    }
    Ok(())
}

/// Mirrors the parser's stop invariant for directly-constructed gradients:
/// both stops must carry the same alpha as the alpha field.
fn validate_gradient_stops(command: &Command) -> Result<(), GpuError> {
    let (top, bottom, alpha) = match *command {
        Command::Gradient {
            top, bottom, alpha, ..
        }
        | Command::GradientHorizontal {
            top, bottom, alpha, ..
        } => (top, bottom, alpha),
        _ => return Ok(()),
    };
    if top[3] != alpha || bottom[3] != alpha {
        return Err(GpuError::InvalidDisplayList(
            "gradient stop alpha must match the alpha field".to_owned(),
        ));
    }
    Ok(())
}

fn bad_effect(message: impl Into<String>) -> GpuError {
    GpuError::InvalidDisplayList(message.into())
}

fn effect_range(value: i64, lo: i64, hi: i64, what: &str) -> Result<(), GpuError> {
    if !(lo..=hi).contains(&value) {
        return Err(bad_effect(format!("{what} {value} must be {lo}..={hi}")));
    }
    Ok(())
}

/// Particle bounds are rechecked on every path so directly-constructed lists
/// cannot make the CPU oracle or the MSL kernel loop beyond the bounded count.
fn validate_particles(count: i64, seed: i64, radius: i64) -> Result<(), GpuError> {
    effect_range(count, 0, PARTICLE_MAX_COUNT, "count")?;
    effect_range(seed, 0, PARTICLE_MAX_SEED, "seed")?;
    effect_range(radius, 0, PARTICLE_MAX_RADIUS, "radius")?;
    Ok(())
}

/// Centre of dot `index`: two independent integer hashes of `(seed, index)`
/// reduced with modulo into the rect. The MSL kernel evaluates the identical
/// `u32` expression, so no particle positions need to be uploaded.
fn particle_center(rect: [i64; 4], seed: i64, index: u32) -> (i64, i64) {
    let mix = |mut x: u32| -> u32 {
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    };
    let s = seed as u32;
    // The two constants decorrelate the axes; wrapping matches MSL `uint`.
    let hx = mix(s ^ index.wrapping_mul(0x9e37_79b9).wrapping_add(0x6c07_8965));
    let hy = mix(s ^ index.wrapping_mul(0x9e37_79b9).wrapping_add(0x85eb_ca6b));
    (
        rect[0] + i64::from(hx % rect[2] as u32),
        rect[1] + i64::from(hy % rect[3] as u32),
    )
}

fn effect_coords(values: [i64; 4], what: &str) -> Result<(), GpuError> {
    for value in values {
        effect_range(value, -COORD_LIMIT, COORD_LIMIT, what)?;
    }
    Ok(())
}

fn effect_normalized(
    values: [i64; 4],
    what: &str,
    width: u32,
    height: u32,
) -> Result<(), GpuError> {
    effect_coords(values, what)?;
    if values[0] < 0 || values[1] < 0 || values[2] < 0 || values[3] < 0 {
        return Err(bad_effect(format!("{what} must be normalized")));
    }
    if values[0].saturating_add(values[2]) > i64::from(width)
        || values[1].saturating_add(values[3]) > i64::from(height)
    {
        return Err(bad_effect(format!("{what} extends outside the viewport")));
    }
    Ok(())
}

fn effect_color(color: [u8; 4]) -> Result<(), GpuError> {
    if color[3] != 255 {
        return Err(bad_effect("effect color alpha must be 255"));
    }
    Ok(())
}

/// Validate an effect command at the parser trust boundary.
pub(crate) fn validate_effect(command: &Command, width: u32, height: u32) -> Result<(), GpuError> {
    let mut rect = [0i64; 4];
    let mut clip = [0i64; 4];
    let mut bounds = [0i64; 4];
    let mut shape_radius = 0i64;
    let mut radius = 0i64;
    let mut denom = 1i64;
    let mut alpha = 0i64;
    let mut color = [0u8, 0, 0, 255];
    let mut has_geometry = false;
    match *command {
        Command::EffectBackdrop => return Ok(()),
        Command::EffectGaussianBlur { radius } => {
            return effect_range(
                radius,
                GAUSSIAN_MIN_RADIUS,
                GAUSSIAN_MAX_RADIUS,
                "gaussian radius",
            );
        }
        Command::EffectBlur { radius: r } => radius = r,
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
        }
        Command::EffectShadow {
            bounds: b,
            radius: r,
            dx,
            dy,
            alpha: a,
            color: col,
        } => {
            bounds = b;
            radius = r;
            effect_range(dx, -EFFECT_MAX_OFFSET, EFFECT_MAX_OFFSET, "dx")?;
            effect_range(dy, -EFFECT_MAX_OFFSET, EFFECT_MAX_OFFSET, "dy")?;
            alpha = a;
            color = col;
        }
        Command::EffectGlow {
            bounds: b,
            radius: r,
            alpha: a,
            color: col,
        } => {
            bounds = b;
            radius = r;
            alpha = a;
            color = col;
        }
        Command::EffectGlass {
            rect: r,
            clip: c,
            bounds: b,
            shape_radius: sr,
            radius: rad,
            alpha: a,
            color: col,
        } => {
            rect = r;
            clip = c;
            bounds = b;
            shape_radius = sr;
            radius = rad;
            alpha = a;
            color = col;
            has_geometry = true;
        }
        Command::EffectLiquid {
            rect: r,
            clip: c,
            bounds: b,
            shape_radius: sr,
            denom: d,
            alpha: a,
            color: col,
        } => {
            rect = r;
            clip = c;
            bounds = b;
            shape_radius = sr;
            denom = d;
            alpha = a;
            color = col;
            has_geometry = true;
        }
        _ => return Ok(()),
    }
    effect_range(radius, 0, EFFECT_MAX_RADIUS, "radius")?;
    effect_range(alpha, 0, EFFECT_UNIT, "alpha")?;
    effect_range(denom, 1, COORD_LIMIT, "denom")?;
    effect_color(color)?;
    effect_coords(rect, "rect")?;
    if rect[2] < 0 || rect[3] < 0 {
        return Err(bad_effect("effect rect must have non-negative size"));
    }
    effect_normalized(bounds, "bounds", width, height)?;
    if has_geometry {
        effect_normalized(clip, "clip", width, height)?;
        effect_range(
            shape_radius,
            0,
            rect[2].min(rect[3]).max(0) / 2,
            "shape_radius",
        )?;
    }
    Ok(())
}

fn field_i64(parser: &mut Parser<'_>, key: &str) -> Result<i64, GpuError> {
    parser.expect(b',', "after effect field")?;
    parser.key(key)?;
    parser.integer()
}

fn field_rect(parser: &mut Parser<'_>, key: &str) -> Result<[i64; 4], GpuError> {
    parser.expect(b',', "after effect field")?;
    parser.key(key)?;
    parser.i64_array(key)
}

fn field_color(parser: &mut Parser<'_>, key: &str) -> Result<[u8; 4], GpuError> {
    parser.expect(b',', "after effect field")?;
    parser.key(key)?;
    parser.u8_array(key)
}

fn parse_effect_command(
    parser: &mut Parser<'_>,
    kind: &str,
    width: u32,
    height: u32,
) -> Result<Command, GpuError> {
    let command = match kind {
        "effect_backdrop" => {
            parser.expect(b'}', "after effect_backdrop")?;
            Command::EffectBackdrop
        }
        "effect_gaussian_blur" => {
            let radius = field_i64(parser, "radius")?;
            parser.expect(b'}', "after effect_gaussian_blur")?;
            Command::EffectGaussianBlur { radius }
        }
        "effect_blur" => {
            let radius = field_i64(parser, "radius")?;
            parser.expect(b'}', "after effect_blur")?;
            Command::EffectBlur { radius }
        }
        "effect_mask" => {
            let rect = field_rect(parser, "rect")?;
            let clip = field_rect(parser, "clip")?;
            let bounds = field_rect(parser, "bounds")?;
            let shape_radius = field_i64(parser, "shape_radius")?;
            let shape = field_i64(parser, "shape")?;
            parser.expect(b'}', "after effect_mask")?;
            effect_range(shape, 0, 1, "shape")?;
            Command::EffectMask {
                rect,
                clip,
                bounds,
                shape_radius,
                shape: shape == 1,
            }
        }
        "effect_shadow" => {
            let bounds = field_rect(parser, "bounds")?;
            let radius = field_i64(parser, "radius")?;
            let dx = field_i64(parser, "dx")?;
            let dy = field_i64(parser, "dy")?;
            let alpha = field_i64(parser, "alpha")?;
            let color = field_color(parser, "color")?;
            parser.expect(b'}', "after effect_shadow")?;
            Command::EffectShadow {
                bounds,
                radius,
                dx,
                dy,
                alpha,
                color,
            }
        }
        "effect_glow" => {
            let bounds = field_rect(parser, "bounds")?;
            let radius = field_i64(parser, "radius")?;
            let alpha = field_i64(parser, "alpha")?;
            let color = field_color(parser, "color")?;
            parser.expect(b'}', "after effect_glow")?;
            Command::EffectGlow {
                bounds,
                radius,
                alpha,
                color,
            }
        }
        "effect_glass" => {
            let rect = field_rect(parser, "rect")?;
            let clip = field_rect(parser, "clip")?;
            let bounds = field_rect(parser, "bounds")?;
            let shape_radius = field_i64(parser, "shape_radius")?;
            let radius = field_i64(parser, "radius")?;
            let alpha = field_i64(parser, "alpha")?;
            let color = field_color(parser, "color")?;
            parser.expect(b'}', "after effect_glass")?;
            Command::EffectGlass {
                rect,
                clip,
                bounds,
                shape_radius,
                radius,
                alpha,
                color,
            }
        }
        "effect_liquid" => {
            let rect = field_rect(parser, "rect")?;
            let clip = field_rect(parser, "clip")?;
            let bounds = field_rect(parser, "bounds")?;
            let shape_radius = field_i64(parser, "shape_radius")?;
            let denom = field_i64(parser, "denom")?;
            let alpha = field_i64(parser, "alpha")?;
            let color = field_color(parser, "color")?;
            parser.expect(b'}', "after effect_liquid")?;
            Command::EffectLiquid {
                rect,
                clip,
                bounds,
                shape_radius,
                denom,
                alpha,
                color,
            }
        }
        other => return Err(parser.error(format!("unsupported command kind {other:?}"))),
    };
    validate_effect(&command, width, height)?;
    Ok(command)
}

fn parse_regular_command(
    parser: &mut Parser<'_>,
    kind: &str,
    width: u32,
    height: u32,
) -> Result<Command, GpuError> {
    parser.expect(b',', "after kind")?;
    parser.key("rect")?;
    let rect = parser.i64_array("rect")?;
    parser.expect(b',', "after rect")?;
    parser.key("clip")?;
    let clip = parser.i64_array("clip")?;

    let command = match kind {
        "solid" => {
            parser.expect(b',', "after clip")?;
            parser.key("color")?;
            let color = parser.u8_array("color")?;
            parser.expect(b'}', "after solid command")?;
            Command::Solid { rect, clip, color }
        }
        "rounded" => {
            parser.expect(b',', "after clip")?;
            parser.key("radius")?;
            let radius = parser.integer()?;
            parser.expect(b',', "after radius")?;
            parser.key("color")?;
            let color = parser.u8_array("color")?;
            parser.expect(b'}', "after rounded command")?;
            Command::Rounded {
                rect,
                clip,
                radius,
                color,
            }
        }
        "gradient" | "gradient_horizontal" => {
            parser.expect(b',', "after clip")?;
            parser.key("radius")?;
            let radius = parser.integer()?;
            parser.expect(b',', "after radius")?;
            parser.key("top")?;
            let top = parser.u8_array("top")?;
            parser.expect(b',', "after top")?;
            parser.key("bottom")?;
            let bottom = parser.u8_array("bottom")?;
            parser.expect(b',', "after bottom")?;
            parser.key("alpha")?;
            let alpha = parser.byte("alpha")?;
            parser.expect(b'}', "after gradient command")?;
            if top[3] != alpha || bottom[3] != alpha {
                return Err(parser.error("gradient stop alpha must match the alpha field"));
            }
            if kind == "gradient_horizontal" {
                Command::GradientHorizontal {
                    rect,
                    clip,
                    radius,
                    top,
                    bottom,
                    alpha,
                }
            } else {
                Command::Gradient {
                    rect,
                    clip,
                    radius,
                    top,
                    bottom,
                    alpha,
                }
            }
        }
        "particles" => {
            parser.expect(b',', "after clip")?;
            parser.key("count")?;
            let count = parser.integer()?;
            parser.expect(b',', "after count")?;
            parser.key("seed")?;
            let seed = parser.integer()?;
            parser.expect(b',', "after seed")?;
            parser.key("radius")?;
            let radius = parser.integer()?;
            parser.expect(b',', "after radius")?;
            parser.key("color")?;
            let color = parser.u8_array("color")?;
            parser.expect(b'}', "after particles command")?;
            Command::Particles {
                rect,
                clip,
                count,
                seed,
                radius,
                color,
            }
        }
        other => return Err(parser.error(format!("unsupported command kind {other:?}"))),
    };
    let (_, _, radius) = command.geometry().expect("regular command");
    validate_geometry(rect, clip, radius, width, height)?;
    if let Command::Particles {
        count,
        seed,
        radius,
        ..
    } = &command
    {
        validate_particles(*count, *seed, *radius)?;
    }
    Ok(command)
}

fn parse_command(parser: &mut Parser<'_>, width: u32, height: u32) -> Result<Command, GpuError> {
    parser.expect(b'{', "in command")?;
    parser.key("kind")?;
    let kind = parser.string()?;
    if kind.starts_with("effect_") {
        parse_effect_command(parser, &kind, width, height)
    } else {
        parse_regular_command(parser, &kind, width, height)
    }
}

pub fn parse_display_list(text: &str) -> Result<DisplayList, GpuError> {
    let mut parser = Parser::new(text);
    parser.expect(b'{', "in display list")?;
    parser.key("version")?;
    if parser.integer()? != 1 {
        return Err(parser.error("unsupported display-list version"));
    }
    parser.expect(b',', "after version")?;
    parser.key("width")?;
    let width = parser.dimension("width")?;
    parser.expect(b',', "after width")?;
    parser.key("height")?;
    let height = parser.dimension("height")?;
    parser.expect(b',', "after height")?;
    parser.key("background")?;
    let background = parser.u8_array("background")?;
    if background[3] != 255 {
        return Err(parser.error("background alpha must be 255"));
    }
    parser.expect(b',', "after background")?;
    parser.key("commands")?;
    parser.expect(b'[', "in commands")?;
    let mut commands = Vec::new();
    if parser.peek()? != b']' {
        loop {
            commands.push(parse_command(&mut parser, width, height)?);
            if commands.len() > MAX_COMMANDS {
                return Err(parser.error("too many commands"));
            }
            match parser.peek()? {
                b',' => {
                    parser.expect(b',', "between commands")?;
                }
                b']' => break,
                other => {
                    return Err(parser.error(format!(
                        "expected ',' or ']' between commands, found '{}'",
                        other as char
                    )));
                }
            }
        }
    }
    parser.expect(b']', "after commands")?;
    parser.expect(b'}', "after display list")?;
    parser.finish()?;
    Ok(DisplayList {
        width,
        height,
        background,
        commands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ppm_bytes;

    fn header(width: u32, height: u32, background: [u8; 4], commands: &str) -> String {
        format!(
            "{{\"version\":1,\"width\":{width},\"height\":{height},\"background\":[{},{},{},{}],\"commands\":[{commands}]}}",
            background[0], background[1], background[2], background[3]
        )
    }

    const SOLID_CLIPPED: &str =
        "{\"kind\":\"solid\",\"rect\":[-2,-1,5,3],\"clip\":[0,0,4,2],\"color\":[200,100,50,128]}";

    #[test]
    fn empty_display_list_renders_opaque_background() {
        let json = header(2, 1, [7, 8, 9, 255], "");
        let list = parse_display_list(&json).unwrap();
        assert!(list.commands.is_empty());
        let pixels = render_cpu_commands(&list).unwrap();
        assert_eq!(pixels, vec![7, 8, 9, 255, 7, 8, 9, 255]);
        assert_eq!(
            ppm_bytes(list.width, list.height, &pixels).unwrap(),
            b"P6\n2 1\n255\n\x07\x08\x09\x07\x08\x09".to_vec()
        );
    }

    #[test]
    fn negative_bounds_are_clipped_and_alpha_blended() {
        let list = parse_display_list(&header(4, 2, [0, 0, 0, 255], SOLID_CLIPPED)).unwrap();
        assert_eq!(list.commands.len(), 1);
        let pixels = render_cpu_commands(&list).unwrap();
        let mut expected = Vec::new();
        for _ in 0..2 {
            for x in 0..4 {
                expected.extend_from_slice(if x < 3 {
                    &[100, 50, 25, 255]
                } else {
                    &[0, 0, 0, 255]
                });
            }
        }
        assert_eq!(pixels, expected);
    }

    #[test]
    fn gradient_stops_and_draw_order_are_preserved() {
        let first = "{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[100,100,100,255],\"alpha\":255}";
        let second = "{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,\"top\":[200,0,0,255],\"bottom\":[0,0,200,255],\"alpha\":255}";
        let json = header(2, 2, [0, 0, 0, 255], &format!("{first},{second}"));
        let list = parse_display_list(&json).unwrap();
        match list.commands[0] {
            Command::Gradient { top, bottom, .. } => {
                assert_eq!(top[0], 0);
                assert_eq!(bottom[0], 100);
            }
            _ => panic!("first command must be a gradient"),
        }
        match list.commands[1] {
            Command::Gradient { top, bottom, .. } => {
                assert_eq!(top[0], 200);
                assert_eq!(bottom[2], 200);
            }
            _ => panic!("second command must be a gradient"),
        }
        let pixels = render_cpu_commands(&list).unwrap();
        assert_eq!(
            pixels,
            // Second row uses t=1024; the shared integer lerp lands on 1, not 0.
            vec![200, 0, 0, 255, 200, 0, 0, 255, 1, 0, 200, 255, 1, 0, 200, 255,]
        );
        let reversed = header(2, 2, [0, 0, 0, 255], &format!("{second},{first}"));
        assert_ne!(
            render_cpu_commands(&parse_display_list(&reversed).unwrap()).unwrap(),
            pixels
        );
    }

    const HORIZONTAL_GRADIENT: &str = "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,1],\"clip\":[0,0,4,1],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255}";

    #[test]
    fn horizontal_gradient_interpolates_across_columns() {
        let list = parse_display_list(&header(4, 1, [0, 0, 0, 255], HORIZONTAL_GRADIENT)).unwrap();
        assert!(matches!(
            list.commands[0],
            Command::GradientHorizontal { .. }
        ));
        assert_eq!(
            render_cpu_commands(&list).unwrap(),
            [0, 0, 0, 255, 85, 85, 85, 255, 170, 170, 170, 255, 255, 255, 255, 255,].to_vec()
        );

        let negative = "{\"kind\":\"gradient_horizontal\",\"rect\":[-1,0,4,1],\"clip\":[0,0,3,1],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255}";
        let list = parse_display_list(&header(3, 1, [0, 0, 0, 255], negative)).unwrap();
        assert_eq!(
            render_cpu_commands(&list).unwrap(),
            [85, 85, 85, 255, 170, 170, 170, 255, 255, 255, 255, 255].to_vec()
        );
    }

    #[test]
    fn horizontal_gradient_clips_and_blends_fractional_alpha() {
        let command = "{\"kind\":\"gradient_horizontal\",\"rect\":[1,0,3,2],\"clip\":[2,0,2,2],\"radius\":0,\"top\":[200,100,0,128],\"bottom\":[0,100,200,128],\"alpha\":128}";
        let list = parse_display_list(&header(4, 2, [10, 20, 30, 255], command)).unwrap();
        let row = [
            10, 20, 30, 255, 10, 20, 30, 255, 55, 60, 65, 255, 5, 60, 115, 255,
        ];
        let mut expected = row.to_vec();
        expected.extend_from_slice(&row);
        assert_eq!(render_cpu_commands(&list).unwrap(), expected);
    }

    #[test]
    fn gradient_orientations_keep_distinct_integer_stops() {
        let stops = "\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255";
        let horizontal = format!(
            "{{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,{stops}}}"
        );
        let vertical = format!(
            "{{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,{stops}}}"
        );
        let horizontal_pixels = render_cpu_commands(
            &parse_display_list(&header(2, 2, [0, 0, 0, 255], &horizontal)).unwrap(),
        )
        .unwrap();
        let vertical_pixels = render_cpu_commands(
            &parse_display_list(&header(2, 2, [0, 0, 0, 255], &vertical)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            horizontal_pixels,
            [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255].to_vec()
        );
        assert_eq!(
            vertical_pixels,
            [0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255].to_vec()
        );
        assert_ne!(horizontal_pixels, vertical_pixels);
    }

    #[test]
    fn horizontal_gradient_packs_as_a_regular_command() {
        let stream =
            format!("{HORIZONTAL_GRADIENT},{{\"kind\":\"effect_backdrop\"}},{HORIZONTAL_GRADIENT}");
        let list = parse_display_list(&header(4, 1, [0, 0, 0, 255], &stream)).unwrap();
        assert_eq!(list.commands.len(), 3);
        let (words, passes) = list.build_passes().unwrap();
        assert_eq!(words.len(), 2 * COMMAND_WORDS);
        assert_eq!(words[0], GRADIENT_HORIZONTAL);
        assert_eq!(words[10], pack_rgba([0, 0, 0, 255]));
        assert_eq!(words[11], pack_rgba([255, 255, 255, 255]));
        assert_eq!(words[12], 255);
        let kernels: Vec<u32> = passes.iter().map(|p| p.kernel).collect();
        assert_eq!(
            kernels,
            vec![KERNEL_BATCH, KERNEL_BATCH, KERNEL_SNAPSHOT, KERNEL_BATCH,]
        );
        assert_eq!(passes[1].params[6], 1);
        assert_eq!(passes[3].params[5], COMMAND_WORDS as u32);
        assert_eq!(passes[3].params[6], 1);

        let invalid = [
            "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,2],\"clip\":[0,0,4,2],\"radius\":3,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255}",
            "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,2],\"clip\":[0,0,5,2],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255}",
            "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,2],\"clip\":[0,0,4,2],\"radius\":0,\"top\":[0,0,0,128],\"bottom\":[255,255,255,255],\"alpha\":128}",
            "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,0],\"clip\":[0,0,4,2],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255],\"alpha\":255}",
            "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,2],\"clip\":[0,0,4,2],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[255,255,255,255]}",
        ];
        for (index, command) in invalid.iter().enumerate() {
            let json = header(4, 2, [0, 0, 0, 255], command);
            let error =
                parse_display_list(&json).expect_err(&format!("invalid case {index} was accepted"));
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "invalid case {index} returned {error:?}"
            );
        }
    }

    #[test]
    fn horizontal_gradient_stop_channels_stay_in_range() {
        let json = header(4, 1, [0, 0, 0, 255], "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,1],\"clip\":[0,0,4,1],\"radius\":0,\"top\":[0,0,300,255],\"bottom\":[255,255,255,255],\"alpha\":255}");
        let error = parse_display_list(&json).expect_err("out-of-range stop channel was accepted");
        assert!(
            matches!(error, GpuError::InvalidDisplayList(_)),
            "{error:?}"
        );
        assert!(error.to_string().contains("top must be 0..=255"), "{error}");
    }

    #[test]
    fn inverted_horizontal_stops_are_valid_and_stay_byte_identical() {
        // Reversed stop order is a legal gradient direction, not a validation
        // failure; the descending integer lerp lands 1 above the final stop.
        let inverted = "{\"kind\":\"gradient_horizontal\",\"rect\":[0,0,4,1],\"clip\":[0,0,4,1],\"radius\":0,\"top\":[255,255,255,255],\"bottom\":[0,0,0,255],\"alpha\":255}";
        let parsed = parse_display_list(&header(4, 1, [0, 0, 0, 255], inverted)).unwrap();
        let constructed = DisplayList {
            width: 4,
            height: 1,
            background: [0, 0, 0, 255],
            commands: vec![Command::GradientHorizontal {
                rect: [0, 0, 4, 1],
                clip: [0, 0, 4, 1],
                radius: 0,
                top: [255, 255, 255, 255],
                bottom: [0, 0, 0, 255],
                alpha: 255,
            }],
        };
        assert_eq!(constructed, parsed);
        assert_eq!(constructed.pack().unwrap(), parsed.pack().unwrap());
        assert_eq!(
            render_cpu_commands(&parsed).unwrap(),
            [255, 255, 255, 255, 171, 171, 171, 255, 86, 86, 86, 255, 1, 1, 1, 255].to_vec()
        );
        assert_eq!(
            render_cpu_commands(&constructed).unwrap(),
            render_cpu_commands(&parsed).unwrap()
        );
    }

    const PARTICLE_STREAM: &str = "{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":6,\"seed\":7,\"radius\":1,\"color\":[200,100,50,255]}";

    #[test]
    fn particles_parse_and_pack_as_a_regular_command() {
        let json = header(8, 4, [0, 0, 0, 255], PARTICLE_STREAM);
        let first = parse_display_list(&json).unwrap();
        let second = parse_display_list(&json).unwrap();
        assert_eq!(first, second);
        match first.commands[0] {
            Command::Particles {
                rect,
                clip,
                count,
                seed,
                radius,
                color,
            } => {
                assert_eq!(rect, [0, 0, 8, 4]);
                assert_eq!(clip, [0, 0, 8, 4]);
                assert_eq!((count, seed, radius), (6, 7, 1));
                assert_eq!(color, [200, 100, 50, 255]);
            }
            _ => panic!("particles command must parse"),
        }
        let words = first.pack().unwrap();
        assert_eq!(words.len(), COMMAND_WORDS);
        assert_eq!(words[0], PARTICLES);
        assert_eq!(words[9], COORD_BIAS as u32, "no rounded-rect corners");
        assert_eq!(words[10], pack_rgba([200, 100, 50, 255]));
        assert_eq!(words[11], 6);
        assert_eq!(words[12], 7);
        assert_eq!(words[13], 1);
        assert_eq!(
            render_cpu_commands(&first).unwrap(),
            render_cpu_commands(&second).unwrap()
        );
    }

    #[test]
    fn particle_pixels_are_seed_sensitive_and_count_zero_is_inert() {
        let command = |count: i64, seed: i64, alpha: u8| {
            format!(
                "{{\"kind\":\"particles\",\"rect\":[0,0,16,8],\"clip\":[0,0,16,8],\"count\":{count},\"seed\":{seed},\"radius\":2,\"color\":[240,120,60,{alpha}]}}"
            )
        };
        let pixels = |c: &str| {
            render_cpu_commands(&parse_display_list(&header(16, 8, [10, 20, 30, 255], c)).unwrap())
                .unwrap()
        };
        let a = pixels(&command(16, 1, 255));
        let b = pixels(&command(16, 2, 255));
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b, "the seed must select the dot positions");
        let mut background = Vec::new();
        for _ in 0..16 * 8 {
            background.extend_from_slice(&[10, 20, 30, 255]);
        }
        assert_eq!(pixels(&command(0, 1, 255)), background);
        assert_eq!(pixels(&command(16, 1, 0)), background, "alpha 0 is inert");
    }

    #[test]
    fn particles_clip_to_rect_and_blend_with_fractional_alpha() {
        let command = "{\"kind\":\"particles\",\"rect\":[1,0,6,4],\"clip\":[2,0,2,4],\"count\":1,\"seed\":9,\"radius\":32,\"color\":[200,100,50,128]}";
        let list = parse_display_list(&header(8, 4, [10, 20, 30, 255], command)).unwrap();
        let mut expected = Vec::new();
        for _ in 0..4 {
            for x in 0..8 {
                let mut px = [10u8, 20, 30, 255];
                if (2..4).contains(&x) {
                    blend(&mut px, [200, 100, 50, 128]);
                }
                expected.extend_from_slice(&px);
            }
        }
        assert_eq!(render_cpu_commands(&list).unwrap(), expected);
    }

    #[test]
    fn particle_bounds_and_malformed_commands_are_rejected() {
        let invalid: &[(&str, &str)] = &[
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":513,\"seed\":1,\"radius\":2,\"color\":[1,2,3,255]}", "count 513 must be 0..=512"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":-1,\"seed\":1,\"radius\":2,\"color\":[1,2,3,255]}", "count -1 must be 0..=512"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":2,\"seed\":-1,\"radius\":2,\"color\":[1,2,3,255]}", "seed -1 must be 0..=4294967295"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":2,\"seed\":4294967296,\"radius\":2,\"color\":[1,2,3,255]}", "seed 4294967296 must be 0..=4294967295"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":2,\"seed\":1,\"radius\":33,\"color\":[1,2,3,255]}", "radius 33 must be 0..=32"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":2,\"seed\":1,\"radius\":-1,\"color\":[1,2,3,255]}", "radius -1 must be 0..=32"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,0,4],\"clip\":[0,0,8,4],\"count\":2,\"seed\":1,\"radius\":2,\"color\":[1,2,3,255]}", "positive width and height"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,9,4],\"count\":2,\"seed\":1,\"radius\":2,\"color\":[1,2,3,255]}", "clip extends outside the viewport"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":\"many\",\"seed\":1,\"radius\":2,\"color\":[1,2,3,255]}", "expected an integer"),
            ("{\"kind\":\"particles\",\"rect\":[0,0,8,4],\"clip\":[0,0,8,4],\"count\":2}", "expected ',' after count"),
        ];
        for (index, (command, expected)) in invalid.iter().enumerate() {
            let error = parse_display_list(&header(8, 4, [0, 0, 0, 255], command))
                .expect_err(&format!("invalid case {index} was accepted"));
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "invalid case {index} returned {error:?}"
            );
            assert!(
                error.to_string().contains(expected),
                "invalid case {index}: {error} must mention {expected:?}"
            );
        }
        // Directly-constructed lists recheck the same bounds.
        let direct = DisplayList {
            width: 8,
            height: 4,
            background: [0, 0, 0, 255],
            commands: vec![Command::Particles {
                rect: [0, 0, 8, 4],
                clip: [0, 0, 8, 4],
                count: 513,
                seed: 1,
                radius: 2,
                color: [1, 2, 3, 255],
            }],
        };
        assert!(render_cpu_commands(&direct)
            .unwrap_err()
            .to_string()
            .contains("count 513 must be 0..=512"));
    }

    #[test]
    fn particles_share_the_regular_batch_with_other_commands() {
        let stream =
            format!("{PARTICLE_STREAM},{{\"kind\":\"effect_backdrop\"}},{PARTICLE_STREAM}");
        let list = parse_display_list(&header(8, 4, [0, 0, 0, 255], &stream)).unwrap();
        let (words, passes) = list.build_passes().unwrap();
        assert_eq!(words.len(), 2 * COMMAND_WORDS);
        let kernels: Vec<u32> = passes.iter().map(|p| p.kernel).collect();
        assert_eq!(
            kernels,
            vec![KERNEL_BATCH, KERNEL_BATCH, KERNEL_SNAPSHOT, KERNEL_BATCH]
        );
        assert_eq!(passes[1].params[6], 1);
        assert_eq!(passes[3].params[5], COMMAND_WORDS as u32);
        assert_eq!(passes[3].params[6], 1);
        let again = parse_display_list(&header(8, 4, [0, 0, 0, 255], &stream)).unwrap();
        assert_eq!(
            render_cpu_commands(&list).unwrap(),
            render_cpu_commands(&again).unwrap()
        );
    }

    #[test]
    fn parsing_and_packing_are_deterministic() {
        let json = header(4, 2, [0, 0, 0, 255], SOLID_CLIPPED);
        let first = parse_display_list(&json).unwrap();
        let second = parse_display_list(&json).unwrap();
        assert_eq!(first, second);
        let words = first.pack().unwrap();
        assert_eq!(words, second.pack().unwrap());
        assert_eq!(words.len(), COMMAND_WORDS);
        assert_eq!(words[0], SOLID);
        assert_eq!(words[1], (-2 + COORD_BIAS) as u32);
        assert_eq!(words[10], pack_rgba([200, 100, 50, 128]));
    }

    #[test]
    fn invalid_display_lists_are_rejected() {
        let cases = [
            "{",
            "{\"version\":2,\"width\":1,\"height\":1,\"background\":[0,0,0,255],\"commands\":[]}",
            &header(
                1,
                1,
                [0, 0, 0, 255],
                "{\"kind\":\"triangle\",\"rect\":[0,0,1,1],\"clip\":[0,0,1,1],\"color\":[0,0,0,255]}",
            ),
            &header(
                1,
                1,
                [0, 0, 0, 255],
                "{\"kind\":\"solid\",\"rect\":[0,0,1,1],\"clip\":[0,0,1,1],\"color\":[256,0,0,255]}",
            ),
            "{\"version\":1,\"width\":0,\"height\":1,\"background\":[0,0,0,255],\"commands\":[]}",
            &format!("{}trailing", header(1, 1, [0, 0, 0, 255], "")),
            "{\"version\":1,\"width\":1,\"commands\":[]}",
        ];
        for (index, json) in cases.iter().enumerate() {
            let error = parse_display_list(json).expect_err(&format!("case {index} was accepted"));
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "case {index} returned {error:?}"
            );
            assert!(error.to_string().contains("InvalidDisplayList"));
        }
    }

    #[test]
    fn coordinate_bias_stays_within_msl_signed_int_range() {
        assert!(COORD_BIAS + COORD_LIMIT < i64::from(i32::MAX));
        assert_eq!(encode_coord(-COORD_LIMIT, "limit").unwrap(), 573_741_824);
        assert_eq!(encode_coord(COORD_LIMIT, "limit").unwrap(), 1_573_741_824);
    }

    const EFFECT_STREAM: &str = "\
{\"kind\":\"effect_backdrop\"},\
{\"kind\":\"effect_mask\",\"rect\":[2,2,6,4],\"clip\":[0,0,16,8],\"bounds\":[0,0,12,8],\"shape_radius\":2,\"shape\":1},\
{\"kind\":\"effect_shadow\",\"bounds\":[0,0,12,8],\"radius\":3,\"dx\":-5,\"dy\":4,\"alpha\":768,\"color\":[1,2,3,255]},\
{\"kind\":\"effect_glow\",\"bounds\":[0,0,12,8],\"radius\":2,\"alpha\":512,\"color\":[4,5,6,255]},\
{\"kind\":\"effect_glass\",\"rect\":[2,2,6,4],\"clip\":[0,0,16,8],\"bounds\":[2,2,6,4],\"shape_radius\":2,\"radius\":1,\"alpha\":384,\"color\":[7,8,9,255]},\
{\"kind\":\"effect_liquid\",\"rect\":[2,2,6,4],\"clip\":[0,0,16,8],\"bounds\":[2,2,6,4],\"shape_radius\":2,\"denom\":8,\"alpha\":640,\"color\":[10,11,12,255]},\
{\"kind\":\"effect_blur\",\"radius\":2}";

    #[test]
    fn every_effect_command_type_parses_with_integer_parameters() {
        let list = parse_display_list(&header(16, 8, [0, 0, 0, 255], EFFECT_STREAM)).unwrap();
        assert_eq!(list.commands.len(), 7);
        assert_eq!(list.commands[0], Command::EffectBackdrop);
        match list.commands[1] {
            Command::EffectMask {
                rect,
                bounds,
                shape_radius,
                shape,
                ..
            } => {
                assert_eq!(rect, [2, 2, 6, 4]);
                assert_eq!(bounds, [0, 0, 12, 8]);
                assert_eq!(shape_radius, 2);
                assert!(shape);
            }
            _ => panic!("effect_mask must parse"),
        }
        match list.commands[2] {
            Command::EffectShadow {
                radius,
                dx,
                dy,
                alpha,
                ..
            } => assert_eq!((radius, dx, dy, alpha), (3, -5, 4, 768)),
            _ => panic!("effect_shadow must parse"),
        }
        match list.commands[3] {
            Command::EffectGlow { radius, alpha, .. } => assert_eq!((radius, alpha), (2, 512)),
            _ => panic!("effect_glow must parse"),
        }
        match list.commands[4] {
            Command::EffectGlass {
                radius,
                shape_radius,
                alpha,
                ..
            } => assert_eq!((radius, shape_radius, alpha), (1, 2, 384)),
            _ => panic!("effect_glass must parse"),
        }
        match list.commands[5] {
            Command::EffectLiquid { denom, alpha, .. } => assert_eq!((denom, alpha), (8, 640)),
            _ => panic!("effect_liquid must parse"),
        }
        match list.commands[6] {
            Command::EffectBlur { radius } => assert_eq!(radius, 2),
            _ => panic!("effect_blur must parse"),
        }
        assert!(list.pack().unwrap().is_empty());
    }

    #[test]
    fn effect_passes_keep_word_offsets_and_kernel_order() {
        let solid =
            "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,8,4],\"color\":[9,8,7,255]}";
        let after_backdrop = "{\"kind\":\"effect_mask\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[0,0,8,4],\"shape_radius\":0,\"shape\":0},\
{\"kind\":\"effect_shadow\",\"bounds\":[0,0,8,4],\"radius\":1,\"dx\":1,\"dy\":1,\"alpha\":512,\"color\":[0,0,0,255]},\
{\"kind\":\"effect_blur\",\"radius\":1}";
        let json = header(
            8,
            4,
            [0, 0, 0, 255],
            &format!("{solid},{{\"kind\":\"effect_backdrop\"}},{solid},{after_backdrop},{solid}"),
        );
        let list = parse_display_list(&json).unwrap();
        let (words, passes) = list.build_passes().unwrap();
        assert_eq!(words.len(), 3 * COMMAND_WORDS);
        let kernels: Vec<u32> = passes.iter().map(|p| p.kernel).collect();
        assert_eq!(
            kernels,
            vec![
                KERNEL_BATCH,
                KERNEL_BATCH,
                KERNEL_SNAPSHOT,
                KERNEL_BATCH,
                KERNEL_MASK,
                KERNEL_EFFECT,
                KERNEL_BLUR_H,
                KERNEL_BLUR_V,
                KERNEL_BATCH,
            ]
        );
        assert_eq!(passes[0].params[2], 1, "the first pass clears the frame");
        assert_eq!(passes[1].params[5], 0);
        assert_eq!(passes[3].params[5], COMMAND_WORDS as u32);
        assert_eq!(passes[8].params[5], 2 * COMMAND_WORDS as u32);
        assert_eq!(passes[6].params[2], 1, "blur radius");
    }

    #[test]
    fn empty_effect_stream_is_valid_and_cpu_oracle_skips_effects() {
        let json = header(2, 1, [7, 8, 9, 255], "");
        let list = parse_display_list(&json).unwrap();
        let (words, passes) = list.build_passes().unwrap();
        assert!(words.is_empty());
        assert_eq!(passes.len(), 1);
        assert_eq!(render_cpu_commands(&list).unwrap().len(), 8);

        let only_effects = header(
            2,
            1,
            [7, 8, 9, 255],
            "{\"kind\":\"effect_blur\",\"radius\":0}",
        );
        let list = parse_display_list(&only_effects).unwrap();
        assert_eq!(list.build_passes().unwrap().1.len(), 3);
        assert_eq!(
            render_cpu_commands(&list).unwrap(),
            vec![7, 8, 9, 255, 7, 8, 9, 255]
        );
    }

    #[test]
    fn malformed_effect_streams_and_bounds_are_rejected() {
        let cases = [
            "{\"kind\":\"effect_warp\"}",
            "{\"kind\":\"effect_blur\",\"radius\":33}",
            "{\"kind\":\"effect_blur\",\"radius\":-1}",
            "{\"kind\":\"effect_shadow\",\"bounds\":[0,0,8,4],\"radius\":0,\"dx\":65,\"dy\":0,\"alpha\":0,\"color\":[0,0,0,255]}",
            "{\"kind\":\"effect_glow\",\"bounds\":[0,0,9,4],\"radius\":1,\"alpha\":0,\"color\":[0,0,0,255]}",
            "{\"kind\":\"effect_glass\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[2,2,4,2],\"shape_radius\":0,\"radius\":1,\"alpha\":0,\"color\":[1,2,3,128]}",
            "{\"kind\":\"effect_liquid\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[2,2,4,2],\"shape_radius\":0,\"denom\":0,\"alpha\":0,\"color\":[1,2,3,255]}",
            "{\"kind\":\"effect_mask\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[0,0,8,4],\"shape_radius\":0,\"shape\":2}",
            "{\"kind\":\"effect_mask\",\"rect\":[2,2,4,2],\"clip\":[0,0,8,4],\"bounds\":[0,0,8,4],\"shape_radius\":9,\"shape\":1}",
            "{\"kind\":\"effect_blur\"",
        ];
        for (index, commands) in cases.iter().enumerate() {
            let json = header(8, 4, [0, 0, 0, 255], commands);
            let error = parse_display_list(&json)
                .expect_err(&format!("malformed case {index} was accepted"));
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "case {index} returned {error:?}"
            );
        }
        let outside = "{\"kind\":\"effect_shadow\",\"bounds\":[-1,0,4,4],\"radius\":0,\"dx\":0,\"dy\":0,\"alpha\":0,\"color\":[0,0,0,255]}";
        let error = parse_display_list(&header(8, 4, [0, 0, 0, 255], outside)).unwrap_err();
        assert!(error.to_string().contains("normalized"), "{error}");
    }

    #[test]
    fn strict_parser_bounds_report_clear_errors_without_panicking() {
        let range_case = format!(
            "{{\"kind\":\"solid\",\"rect\":[0,0,{},1],\"clip\":[0,0,8,4],\"color\":[1,2,3,255]}}",
            COORD_LIMIT + 1
        );
        let cases: &[(&str, &str)] = &[
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[-1,0,2,2],\"color\":[1,2,3,255]}",
                "clip must be normalized",
            ),
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,9,2],\"color\":[1,2,3,255]}",
                "clip extends outside the viewport",
            ),
            (
                "{\"kind\":\"effect_shadow\",\"bounds\":[-1,0,4,4],\"radius\":1,\"dx\":0,\"dy\":0,\"alpha\":0,\"color\":[0,0,0,255]}",
                "bounds must be normalized",
            ),
            (
                "{\"kind\":\"effect_glow\",\"bounds\":[0,0,9,4],\"radius\":1,\"alpha\":0,\"color\":[0,0,0,255]}",
                "bounds extends outside the viewport",
            ),
            (&range_case, "exceeds the integer range"),
            (
                "{\"kind\":\"triangle\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"color\":[1,2,3,255]}",
                "unsupported command kind",
            ),
            ("{\"kind\":\"effect_warp\"}", "unsupported command kind"),
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"color\":[256,0,0,255]}",
                "color must be 0..=255",
            ),
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"color\":[-1,0,0,255]}",
                "color must be 0..=255",
            ),
            (
                "{\"kind\":\"effect_glow\",\"bounds\":[0,0,8,4],\"radius\":1,\"alpha\":0,\"color\":[1,2,3,256]}",
                "color must be 0..=255",
            ),
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"color\":[1,2,3",
                "expected ',' in color",
            ),
            (
                "{\"kind\":\"solid\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"color\":[1,2,3,255]",
                "expected '}' after solid command",
            ),
            (
                "{\"kind\":\"rounded\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":1,\"color\":[]}",
                "expected an integer",
            ),
        ];
        for (index, (command, expected)) in cases.iter().enumerate() {
            let json = header(8, 4, [0, 0, 0, 255], command);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parse_display_list(&json)
            }));
            let error = match result {
                Ok(Err(error)) => error,
                Ok(Ok(_)) => panic!("case {index} was accepted"),
                Err(_) => panic!("case {index} panicked"),
            };
            assert!(
                matches!(error, GpuError::InvalidDisplayList(_)),
                "case {index}: {error}"
            );
            assert!(
                error.to_string().contains(expected),
                "case {index}: {error} must mention {expected:?}"
            );
        }
        let background =
            "{\"version\":1,\"width\":1,\"height\":1,\"background\":[0,0,0,256],\"commands\":[]}";
        assert!(parse_display_list(background)
            .unwrap_err()
            .to_string()
            .contains("background must be 0..=255"));
        let truncated = "{\"version\":1,\"width\":8,\"height\":4,\"background\":[0,0,0,255],\"commands\":[{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,\"top\":[0,0,0,255]";
        assert!(parse_display_list(truncated)
            .unwrap_err()
            .to_string()
            .contains("unexpected end of JSON"));
    }

    #[test]
    fn valid_streams_keep_byte_identical_packed_and_pixel_output() {
        let parsed = parse_display_list(&header(4, 2, [0, 0, 0, 255], SOLID_CLIPPED)).unwrap();
        let mut expected_words = vec![SOLID];
        for value in [-2i64, -1, 5, 3, 0, 0, 4, 2, 0] {
            expected_words.push((value + COORD_BIAS) as u32);
        }
        expected_words.push(pack_rgba([200, 100, 50, 128]));
        expected_words.extend_from_slice(&[0u32; COMMAND_WORDS - 11]);
        assert_eq!(parsed.pack().unwrap(), expected_words);
        let constructed = DisplayList {
            width: 4,
            height: 2,
            background: [0, 0, 0, 255],
            commands: vec![Command::Solid {
                rect: [-2, -1, 5, 3],
                clip: [0, 0, 4, 2],
                color: [200, 100, 50, 128],
            }],
        };
        assert_eq!(constructed, parsed);
        assert_eq!(constructed.pack().unwrap(), expected_words);
        let mut expected_pixels = Vec::new();
        for _ in 0..2 {
            for x in 0..4 {
                expected_pixels.extend_from_slice(if x < 3 {
                    &[100, 50, 25, 255]
                } else {
                    &[0, 0, 0, 255]
                });
            }
        }
        assert_eq!(render_cpu_commands(&parsed).unwrap(), expected_pixels);
        assert_eq!(render_cpu_commands(&constructed).unwrap(), expected_pixels);

        let first = "{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,\"top\":[0,0,0,255],\"bottom\":[100,100,100,255],\"alpha\":255}";
        let second = "{\"kind\":\"gradient\",\"rect\":[0,0,2,2],\"clip\":[0,0,2,2],\"radius\":0,\"top\":[200,0,0,255],\"bottom\":[0,0,200,255],\"alpha\":255}";
        let gradients =
            parse_display_list(&header(2, 2, [0, 0, 0, 255], &format!("{first},{second}")))
                .unwrap();
        assert_eq!(
            render_cpu_commands(&gradients).unwrap(),
            vec![200, 0, 0, 255, 200, 0, 0, 255, 1, 0, 200, 255, 1, 0, 200, 255]
        );

        let effects = parse_display_list(&header(16, 8, [0, 0, 0, 255], EFFECT_STREAM)).unwrap();
        assert!(effects.pack().unwrap().is_empty());
        let kernels: Vec<u32> = effects
            .build_passes()
            .unwrap()
            .1
            .iter()
            .map(|pass| pass.kernel)
            .collect();
        assert_eq!(
            kernels,
            vec![
                KERNEL_BATCH,
                KERNEL_SNAPSHOT,
                KERNEL_MASK,
                KERNEL_EFFECT,
                KERNEL_EFFECT,
                KERNEL_EFFECT,
                KERNEL_EFFECT,
                KERNEL_BLUR_H,
                KERNEL_BLUR_V,
            ]
        );
    }
}
