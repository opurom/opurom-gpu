# opurom-gpu

Independent Rust crate: a **real macOS Metal compute backend** that fills an
RGBA8 buffer with a solid colour plus a deterministic two-stop vertical
gradient, reads the buffer back, and compares its bytes against an independent
CPU reference. It also **executes the exact display list dumped by
`opurom-render --commands`** with a Metal compute kernel and writes a P6 PPM
that is byte-identical to the independent software renderer. Real MSL
compute kernels cover the five visual effects (blur, shadow, glow, glass,
liquid), with a second cross-repo byte-equality proof for effect IR. The
presentation path keeps
the compute output buffer on the GPU and presents it into a live AppKit
`CAMetalLayer` drawable through a runtime-loaded C ABI. This folder is its own
git repository. All files and comments are English.

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
- No font bitmap lives in this crate: `opurom-render` rasterizes anti-aliased
  glyph coverage into 1x1 solid commands with fractional alpha, and Metal
  blends them with the same integer `/255` contract as the software raster.
- Effect commands: `snapshot_frame`, `build_mask`, `effect_local` and the
  two `blur_*` kernels run as separate committed compute passes over two
  ping-pong RGBA frame buffers, one backdrop snapshot and one silhouette-mask
  buffer. Each pass dispatches one thread per pixel; the final buffer is read
  back on the CPU.
- Presentation: `raster_buffer` retains the final `MTLBuffer` on the GPU.
  `src/presentation.rs` exports `opr_present_display_list` from a cdylib; it
  sets an RGBA8 `CAMetalLayer`, calls `nextDrawable()`, copies the compute
  buffer into the drawable texture with a real `MTLBlitCommandEncoder`, then
  calls `presentDrawable`, `commit` and `waitUntilCompleted`. The presented
  drawable's own IOSurface is exposed as layer contents so the platform's
  `cacheDisplay(in:to:)` can composite the exact GPU surface offscreen.

The pixel contract is opaque RGBA8. An inset border uses the solid colour; the
interior uses the same fixed-point two-stop vertical gradient arithmetic as
`opurom-render`: `t` is `0..=1024`, and each channel adds `512` before
truncating signed integer division. The MSL kernel and the CPU oracle use the
same integer order and packing, so the readback must byte-match.

## Display-list raster proof

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
cmp cpu.ppm gpu.ppm   # exit 0, PPM digest 2d4f92cb70b30ab4
```

That comparison is a real GPU/software proof between two independent
repositories, not a self-comparison against the same code. If the sibling
renderer binary is absent, the test prints a clear `SKIP` message and does not
claim success.

Text glyph compositing stays CPU-side: glyphs reach
the GPU only as pre-decomposed solid rectangles; there is no GPU glyph atlas,
font shaping or full-scene GPU text. The `raster` command is headless; the
drawable presentation path is described below. There is no dirty-rect
integration, no Vulkan/ANGLE/D3D backend, no device conformance matrix and no
fps claim. `dispatch_ms` remains host-side commit-to-completion latency, not an
fps measurement.

## GPU effect kernels and byte-equality proof

`opurom-render --commands` also emits optional effect commands with integer
parameters for IR carrying the five visual effects. Effect-free streams stay
byte-identical to the plain display-list path. The seven effect command kinds
are:

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
`tests/effects.rs` runs both binaries and the system `cmp`:

```sh
../opurom-render/target/debug/opurom-render \
  ../opurom-render/tests/fixtures/r9_effect.opuir.json 96 64 cpu.ppm \
  --commands fx.commands.json      # digest=ab00722ca3ae679a, dirty=1
./target/debug/opurom-gpu raster fx.commands.json gpu.ppm
cmp cpu.ppm gpu.ppm                 # exit 0, byte-identical
```

Measured result: `cmp` exits 0, both PPMs are 18445 bytes, and the FNV-1a 64
digest of the PPM bytes is `292043461093fcd5`. The same test also proves
equality at `64 48` (9229 bytes, PPM digest `51ba485cdab48339`). This is a real
comparison against the independent software renderer, not a self-comparison.
If the sibling renderer binary or fixture is absent, the test prints a clear
`SKIP` message and claims no success.

The `raster` CLI itself is compute-only. Drawable presentation is reached
through `opurom-platform`; there is still no GPU text
(glyphs remain CPU-decomposed solid rects), no Vulkan/ANGLE/D3D backend,
no device conformance matrix and no fps claim. `dispatch_ms` remains host-side
commit-to-completion latency. Non-macOS hosts return the documented
`MetalUnavailable` error and the tests print `SKIP`.

## Real drawable presentation

`cargo build --offline` also produces `target/debug/libopurom_gpu.dylib`. The
`opurom-platform` CLI opens a live `NSWindow` and loads this dylib at runtime
with `dlopen`/`dlsym`; its own binary has no `objc2-metal` dependency and no
`-framework Metal` link (the same pattern as the X11 runtime loader). The C
entry point `opr_present_display_list(json, layer, reason, capacity)` takes the
renderer's display-list JSON and the retained `CAMetalLayer` pointer. It:

1. runs the display-list compute passes and **retains the final GPU
   `MTLBuffer`** (`raster_buffer`), instead of reading it back to the CPU;
2. sets the layer's device, RGBA8 pixel format and drawable size, then calls
   `nextDrawable()`;
3. encodes `copyFromBuffer:...:toTexture:` on a real `MTLBlitCommandEncoder`,
   calls `presentDrawable`, commits the command buffer and waits for completion;
4. exposes the presented drawable's **own IOSurface** as layer contents, with no
   CPU copy, so `cacheDisplay(in:to:)` can composite the exact on-screen GPU
   surface (Core Animation's offscreen renderer cannot see window-server
   drawable content directly).

Status `0` means a real presentation. No Metal device, no drawable, a missing
IOSurface, or a non-macOS host returns a documented SKIP status with a concrete
reason; Metal/API errors return FAIL. No CPU `CGImage` substitution is used for
the presented frame. Unit tests cover the blit geometry, the ordered
`blit`/`presentDrawable`/`commit`/`wait` encoding, the skip statuses and the
null-argument FFI path.

The cross-repo verification command is
`opurom-platform --verify-gpu-presentation` (see that repository's README).
`OPR_SKIP_WINDOW=1` forces a documented SKIP. On the reference macOS arm64 /
Apple M6 host, the real capture is a 200x182 PNG (200x150 content + 32 px title
bar) with **0 mismatching pixels of 30 000**, max channel delta 25 and
`CHANNEL_TOLERANCE=32` / `MISMATCH_BUDGET_PERCENT=2.0`: **PASS**. This is one
window and one frame, macOS/Metal only. There is no Vulkan/ANGLE/D3D backend,
no GPU text shaping, no measured fps claim and no device conformance matrix.

## Animated multi-frame presentation

`opr_present_display_list_timed(json, layer, reason, capacity, timings)` is the
per-frame C entry point for the bounded animation loop. It has the same status
vocabulary as `opr_present_display_list` and additionally writes a `repr(C)`
`FrameTimings { dispatch_ns, present_ns }` measured with `std::time::Instant`:
`dispatch_ns` covers the Metal compute raster of that frame, `present_ns` covers
drawable acquisition, the blit, `presentDrawable`, commit, completion and the
layer contents refresh. Timings are zero unless the frame really presented.

On every loop frame the layer contents is **rebound to the IOSurface of the
drawable that was just presented**: `setContents(None)` in a committed
`CATransaction`, a flush, then `setContents(Some(surface))` in another committed
transaction and a flush. `CAMetalLayer` recycles a small IOSurface pool, so
reassigning the same object without clearing it is a no-op for Core Animation
and `cacheDisplay(in:to:)` would keep showing the first frame. The single-frame
entry point keeps its original binding unchanged.

The loop driver and capture live in `opurom-platform` under
`--verify-gpu-animation-loop`; both are described there. This crate performs no
window capture and makes no fps or smoothness claim.

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
The presentation dylib requires the live window supplied by `opurom-platform`.

## Offline build and tests

```sh
cargo fmt --all
cargo fmt --all -- --check
RUSTFLAGS="-D warnings" cargo test --offline
RUSTFLAGS="-D warnings" cargo build --offline
RUSTFLAGS="-D warnings" cargo check --offline --tests --target x86_64-unknown-linux-gnu
```

Metal, QuartzCore and IOSurface dependencies appear only under
`[target.'cfg(target_os = "macos")'.dependencies]` in `Cargo.toml`; the C ABI
and all presentation code are `cfg`-gated, so Linux CI builds and tests the
pure-Rust parser, packer, pass builder, reference, validation, digest, blit
geometry and error paths without Metal frameworks or Metal crates. The
`cdylib` is loaded at runtime by `opurom-platform`; there is no link-time
framework requirement in that binary. On non-macOS, GPU commands return the
documented `MetalUnavailable` error and exit nonzero; tests print a clear
`SKIP` message (visible with `cargo test --offline -- --nocapture`) and assert
that no fake success occurs. On macOS, `cargo test --offline` runs **33 tests**
(26 display-list and effect tests plus seven presentation tests), including the
two cross-repo `cmp`
proofs in `tests/display.rs` and `tests/effects.rs`; GPU execution cannot be
validated by Linux CI.

## Backend scope

This backend is display-list work over rects plus five framebuffer effect kernels, with
one real Metal drawable presentation path for one live AppKit window. Glyph
shapes reach the GPU as CPU-decomposed rectangles, and the system default Metal device is the
one queried. The dispatch path, the readback equality proof and the on-screen presentation
comparison are the verified properties of this backend.
No Metal API call in this backend is substituted with a software stub; the only
software implementation is the explicitly separate CPU parsing/oracle path used
for unit tests, while `raster` always performs the Metal dispatch on macOS and
the presentation dylib always blits the compute buffer into a real drawable.

## API notes

`objc2-metal 0.3.2` exposes `MTLCreateSystemDefaultDevice` behind its
`MTLDevice` feature and requires CoreGraphics to be linked; `src/metal.rs`
adds that framework link. `objc2-quartz-core 0.3.2` supplies `CAMetalLayer` and
`nextDrawable()`, and `objc2-io-surface 0.3.2` exposes the presented drawable's
own surface for `cacheDisplay`. All source compilation is performed at runtime
by `newLibraryWithSource`, so the build does not require a `.metallib` or an
Xcode Metal Toolchain. No API limitation required replacing a requested Metal
call; the platform adapter resolves the C ABI with `dlopen`/`dlsym`.
