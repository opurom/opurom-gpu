# opurom-gpu

Independent Rust crate: a **real macOS Metal compute backend** that fills an
RGBA8 buffer with a solid colour plus a deterministic two-stop vertical
gradient, reads the buffer back, and compares its bytes against an independent
CPU reference. In R6 it also **executes the exact display list dumped by
`opurom-render --commands`** with a Metal compute kernel and writes a P6 PPM
that is byte-identical to the independent software renderer. R9 adds real MSL
compute kernels for the five R8 visual effects (blur, shadow, glow, glass,
liquid), with a second cross-repo byte-equality proof for effect IR. This folder
is its own git repository. All files and comments are English.

## Scope

Implemented now:

- `MTLCreateSystemDefaultDevice()` through cached `objc2-metal 0.3.2` bindings.
- Runtime MSL compilation with `newLibraryWithSource`; no Xcode Metal Toolchain
  is needed at build time.
- A compute pipeline state, command queue, shared-storage RGBA8 output buffer,
  one dispatched thread per pixel, `commit`, `waitUntilCompleted`, and CPU
  readback.
- Explicit handling for every Objective-C `NSError` and nil return in this
  path. A nil system device is reported as `MetalUnavailable`.
- `probe`: prints the real `MTLDevice.name()`, the measured `Instant` elapsed
  time from commit through completion, the readback byte count, and the FNV-1a
  64-bit digest.
- `self-test`: compares the real GPU readback with `src/reference.rs` for four
  sizes and colour sets. A byte mismatch returns a nonzero exit status. On
  success it prints one line containing `PASS`, the real device name, the case
  count, and the stable aggregate digest.
- `raster <commands.json> <out.ppm>`: parses the renderer's deterministic
  display-list JSON with a standard-library parser (no new crate), packs the
  commands into a `device const uint*` buffer, uploads the output RGBA buffer,
  dispatches one Metal thread per viewport pixel over `raster_batch`, reads the
  result back, and writes the renderer's exact P6 PPM format
  (`P6\n<w> <h>\n255\n` + RGB triplets).
- R9 effect commands: `snapshot_frame`, `build_mask`, `effect_local` and the
  two `blur_*` kernels run as separate committed compute passes over two
  ping-pong RGBA frame buffers, one backdrop snapshot and one silhouette-mask
  buffer. Each pass dispatches one thread per pixel; the final buffer is read
  back on the CPU.

The pixel contract is opaque RGBA8. An inset border uses the solid colour; the
interior uses the same fixed-point two-stop vertical gradient arithmetic as
`opurom-render`: `t` is `0..=1024`, and each channel adds `512` before
truncating signed integer division. The MSL kernel and the CPU oracle use the
same integer order and packing, so the readback must byte-match.

## R6 display-list raster proof

`opurom-render --commands <out.json>` dumps the exact frame display list:
viewport, opaque background, and every solid rect, rounded rect and two-stop
vertical gradient in draw order with integer bounds and colours (bitmap glyph
runs are decomposed CPU-side into solid rect commands). `raster` executes that
stream on Metal: signed coordinates are encoded as `value + 2^30` for MSL's
`int`, corners use the renderer's integer quarter-circle test, and alpha
blending uses the renderer's `/255` integer blend, all evaluated per pixel in
draw order.

The cross-repo test in `tests/display.rs` locates the sibling renderer binary
`../opurom-render/target/debug/opurom-render` (override with
`OPUROM_RENDER_BIN`), runs both binaries and asserts byte equality:

```sh
# rect/gradient fixture: cmp exit 0, byte-identical
../opurom-render/target/debug/opurom-render \
  ../opurom-render/tests/fixtures/r6_rect_gradient.opuir.json 96 64 cpu.ppm \
  --commands fixture.commands.json
./target/debug/opurom-gpu raster fixture.commands.json gpu.ppm
cmp cpu.ppm gpu.ppm   # exit 0
# golden counter frame (includes CPU-decomposed glyphs): cmp exit 0 as well
../opurom-render/target/debug/opurom-render \
  ../opurom-spec/counter.opuir.json 320 240 cpu.ppm --commands counter.commands.json
./target/debug/opurom-gpu raster counter.commands.json gpu.ppm
cmp cpu.ppm gpu.ppm   # exit 0, PPM digest 56dace091d467614
```

That comparison is a real GPU/software proof between two independent
repositories, not a self-comparison against the same code. If the sibling
renderer binary is absent, the test prints a clear `SKIP` message and does not
claim success.

**Honest scope:** text glyph compositing stays CPU-side in R6 — glyphs reach
the GPU only as pre-decomposed solid rectangles; there is no GPU glyph atlas,
font shaping or full-scene GPU text. There is no GPU presentation to a window,
no dirty-rect integration, no Vulkan/ANGLE/D3D backend, no device conformance
matrix and no fps claim. `dispatch_ms` remains host-side commit-to-completion
latency, not an fps measurement.

## R9 GPU effect kernels and byte-equality proof

`opurom-render --commands` now emits optional effect commands with integer
parameters for IR carrying the five R8 effects. Effect-free streams stay
byte-identical to R6. The seven R9 command kinds are:

| Kind | Integer parameters | Metal pass |
| --- | --- | --- |
| `effect_backdrop` | none | `snapshot_frame`: copy the current frame to a backdrop buffer |
| `effect_mask` | rect, clip, padded bounds, shape radius, shape flag | `build_mask`: rounded silhouette or frame/backdrop diff |
| `effect_shadow` | bounds, radius, dx, dy, fixed-point alpha, colour | `effect_local` with a two-pass softened mask |
| `effect_glow` | bounds, radius, alpha, colour | `effect_local` with a two-pass softened mask |
| `effect_glass` | rect, clip, bounds, shape radius, radius, alpha, colour | `effect_local` with a two-pass blurred backdrop |
| `effect_liquid` | rect, clip, bounds, shape radius, denom, alpha, colour | `effect_local` with the integer specular band |
| `effect_blur` | radius | `blur_horizontal` + `blur_vertical` over the whole frame |

All arithmetic is integer and mirrors the software renderer exactly: box-blur
half-up rounding, `(src*a + dst*(255-a) + 127)/255` effect blending, clamped
edge windows and the same pre-order draw order. Regular commands are packed into
batches; a separate committed command buffer per pass guarantees that each pass
sees the previous pass's writes.

The committed effect fixture is
`../opurom-render/tests/fixtures/r9_effect.opuir.json` (all five effects).
`tests/r9.rs` runs both binaries and the system `cmp`:

```sh
../opurom-render/target/debug/opurom-render \
  ../opurom-render/tests/fixtures/r9_effect.opuir.json 96 64 cpu.ppm \
  --commands fx.commands.json      # digest=eb9ce1cec41bac13, dirty=1
./target/debug/opurom-gpu raster fx.commands.json gpu.ppm
cmp cpu.ppm gpu.ppm                 # exit 0, byte-identical
```

Measured result: `cmp` exits 0, both PPMs are 18445 bytes, and the FNV-1a 64
digest of the PPM bytes is `b493cd0714a4fb64`. The same test also proves
equality at `64 48` (9229 bytes, PPM digest `70d42c01f7ee35b2`). This is a real
comparison against the independent software renderer, not a self-comparison.
If the sibling renderer binary or fixture is absent, the test prints a clear
`SKIP` message and claims no success.

**Honest scope:** compute kernels only. No GPU presentation to a window, no GPU
text (glyphs remain CPU-decomposed solid rects), no Vulkan/ANGLE/D3D backend,
no device conformance matrix and no fps claim. `dispatch_ms` remains host-side
commit-to-completion latency. Non-macOS hosts return the documented
`MetalUnavailable` error and the tests print `SKIP`.

## Commands

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --offline
./target/debug/opurom-gpu probe
./target/debug/opurom-gpu self-test
./target/debug/opurom-gpu raster counter.commands.json out.ppm
```

`probe` is multi-line and has these labelled fields:

```text
device=<MTLDevice.name()>
backend=metal-compute
size=<width>x<height>
bytes=<readback byte count>
dispatch_ms=<measured commit-to-completion milliseconds>
digest=<FNV-1a 64-bit hexadecimal digest>
```

`self-test` prints exactly one line shaped like:

```text
PASS device=<MTLDevice.name()> cases=4 digest=<FNV-1a 64-bit hexadecimal digest>
```

`raster` prints nothing on success and exits 0; a malformed display list,
missing file or unwritable output path is a readable `error:` diagnostic with
exit code 1 (usage errors exit 2). On non-macOS, `raster` fails with the
documented `MetalUnavailable` message before touching the output file.

`dispatch_ms` is host-side commit-to-completion latency measured with
`Instant`, not a GPU timestamp and not an fps measurement. Both `probe` and
`self-test` are headless; they do not create a window or presentation layer.

## Offline build and tests

```sh
cargo fmt --all
cargo fmt --all -- --check
cargo test --offline
cargo build --offline
```

Metal dependencies appear only under
`[target.'cfg(target_os = "macos")'.dependencies]` in `Cargo.toml`. Linux CI
therefore builds and tests the pure-Rust parser, packer, pass builder,
reference, validation, digest and error paths without Metal frameworks or Metal
crates; `cargo check --tests --target x86_64-unknown-linux-gnu` is clean with
`-D warnings`. On non-macOS, GPU commands return the documented
`MetalUnavailable` error and exit nonzero; tests print a clear `SKIP` message
(visible with `cargo test --offline -- --nocapture`) and assert that no fake
success occurs. On macOS, `cargo test --offline` runs 27 tests, including the
two cross-repo `cmp` proofs in `tests/display.rs` and `tests/r9.rs`; GPU
execution cannot be validated by Linux CI.

## Honest limits / R9+

This is headless display-list work over rects plus five framebuffer effect
kernels. Glyph shapes reach the GPU only as CPU-decomposed rectangles. It is
**not** a full GPU UI rasterizer. There is no Vulkan, ANGLE, D3D, Metal
presentation, texture atlas, glyph rendering, image decoding, or dirty-rect
integration. There is no measured 120 fps claim, no benchmark claim, and no
device conformance matrix: only the system default Metal device is queried.
Other backends, presentation, GPU text and multi-device certification remain
R10+ work.
No Metal API call in this backend is substituted with a software stub; the only
software implementation is the explicitly separate CPU parsing/oracle path used
for unit tests, while `raster` always performs the Metal dispatch on macOS.

## API notes

`objc2-metal 0.3.2` exposes `MTLCreateSystemDefaultDevice` behind its
`MTLDevice` feature and requires CoreGraphics to be linked; `src/metal.rs`
adds that framework link. All source compilation is performed at runtime by
`newLibraryWithSource`, so the build does not require a `.metallib` or an Xcode
Metal Toolchain. No API limitation required replacing a requested Metal call.
