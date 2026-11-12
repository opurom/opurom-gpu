//! Standard-library parser, packer and CPU unit oracle for R6 display lists.
//!
//! This module never substitutes for Metal. `raster` returns MetalUnavailable
//! off macOS; the CPU interpreter below is only for parser/kernel unit tests.

use crate::{GpuError, MAX_DIMENSION};
use std::str;

pub const MAX_COMMANDS: usize = 1_000_000;
pub const COMMAND_WORDS: usize = 16;
/// Signed coordinates are encoded as `value + COORD_BIAS` for MSL's `int`.
pub const COORD_BIAS: i64 = 1 << 30;
pub const COORD_LIMIT: i64 = 500_000_000;

const UNIT: i64 = 1024;
const SOLID: u32 = 0;
const ROUNDED: u32 = 1;
const GRADIENT: u32 = 2;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisplayList {
    pub width: u32,
    pub height: u32,
    pub background: [u8; 4],
    pub commands: Vec<Command>,
}

impl Command {
    fn geometry(&self) -> ([i64; 4], [i64; 4], i64) {
        match *self {
            Command::Solid { rect, clip, .. } => (rect, clip, 0),
            Command::Rounded {
                rect, clip, radius, ..
            }
            | Command::Gradient {
                rect, clip, radius, ..
            } => (rect, clip, radius),
        }
    }

    fn sample(&self, x: i64, y: i64) -> Option<[u8; 4]> {
        let (rect, clip, radius) = self.geometry();
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
            } => {
                let last = rect[3] - 1;
                let t = if last <= 0 {
                    UNIT
                } else {
                    ((y - rect[1]) * UNIT) / last
                };
                Some([
                    mix_channel(top[0], bottom[0], t),
                    mix_channel(top[1], bottom[1], t),
                    mix_channel(top[2], bottom[2], t),
                    alpha,
                ])
            }
        }
    }
}

impl DisplayList {
    pub fn pack(&self) -> Result<Vec<u32>, GpuError> {
        let mut words = Vec::with_capacity(self.commands.len() * COMMAND_WORDS);
        for command in &self.commands {
            let (rect, clip, radius) = command.geometry();
            let mut packed = [0u32; COMMAND_WORDS];
            packed[0] = match command {
                Command::Solid { .. } => SOLID,
                Command::Rounded { .. } => ROUNDED,
                Command::Gradient { .. } => GRADIENT,
            };
            for (offset, value) in rect.iter().enumerate() {
                packed[1 + offset] = encode_coord(*value, "rect")?;
            }
            for (offset, value) in clip.iter().enumerate() {
                packed[5 + offset] = encode_coord(*value, "clip")?;
            }
            packed[9] = encode_coord(radius, "radius")?;
            match *command {
                Command::Solid { color, .. } | Command::Rounded { color, .. } => {
                    packed[10] = crate::display::pack_rgba(color);
                }
                Command::Gradient {
                    top, bottom, alpha, ..
                } => {
                    packed[10] = crate::display::pack_rgba(top);
                    packed[11] = crate::display::pack_rgba(bottom);
                    packed[12] = u32::from(alpha);
                }
            }
            words.extend_from_slice(&packed);
        }
        Ok(words)
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
            "{what} coordinate {value} exceeds the R6 integer range"
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

/// Pure-Rust unit oracle for parser/kernel tests. The real `raster` command
/// never calls this on a non-Metal host.
pub fn render_cpu_commands(list: &DisplayList) -> Result<Vec<u8>, GpuError> {
    let byte_count = crate::display_byte_count(list.width, list.height)?;
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
                if let Some(src) = command.sample(i64::from(x), i64::from(y)) {
                    blend(&mut pixels[offset..offset + 4], src);
                }
            }
        }
    }
    Ok(pixels)
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
                "coordinate {value} exceeds the R6 integer range"
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

fn parse_command(parser: &mut Parser<'_>, width: u32, height: u32) -> Result<Command, GpuError> {
    parser.expect(b'{', "in command")?;
    parser.key("kind")?;
    let kind = parser.string()?;
    parser.expect(b',', "after kind")?;
    parser.key("rect")?;
    let rect = parser.i64_array("rect")?;
    parser.expect(b',', "after rect")?;
    parser.key("clip")?;
    let clip = parser.i64_array("clip")?;

    let command = match kind.as_str() {
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
        "gradient" => {
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
            Command::Gradient {
                rect,
                clip,
                radius,
                top,
                bottom,
                alpha,
            }
        }
        other => {
            return Err(parser.error(format!("unsupported command kind {other:?}")));
        }
    };
    let (_, _, radius) = command.geometry();
    validate_geometry(rect, clip, radius, width, height)?;
    Ok(command)
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
}
