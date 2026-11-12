# opurom-gpu

Independent Rust crate: a **real macOS Metal compute backend** that fills an
RGBA8 buffer with a solid colour plus a deterministic two-stop vertical
gradient, reads the buffer back, and compares its bytes against an independent
CPU reference. This folder is its own git repository. All files and comments
are English.

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

The pixel contract is opaque RGBA8. An inset border uses the solid colour; the
interior uses the same fixed-point two-stop vertical gradient arithmetic as
`opurom-render`: `t` is `0..=1024`, and each channel adds `512` before
truncating signed integer division. The MSL kernel and the CPU oracle use the
same integer order and packing, so the readback must byte-match.

## Commands

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --offline
./target/debug/opurom-gpu probe
./target/debug/opurom-gpu self-test
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

`dispatch_ms` is host-side commit-to-completion latency measured with
`Instant`, not a GPU timestamp and not an fps measurement. Both commands are
headless; they do not create a window or presentation layer.

## Offline build and tests

```sh
cargo fmt --all
cargo fmt --all -- --check
cargo test --offline
cargo build --offline
```

Metal dependencies appear only under
`[target.'cfg(target_os = "macos")'.dependencies]` in `Cargo.toml`. Linux CI
therefore builds and tests the pure-Rust reference, validation, digest and
error paths without Metal frameworks or Metal crates. On non-macOS, both
commands return the documented `MetalUnavailable` error and exit nonzero;
tests print a clear `SKIP` message (visible with
`cargo test --offline -- --nocapture`) and assert that no fake success occurs.
The cross-layer harness on macOS runs
`opurom-gpu/target/debug/opurom-gpu self-test`, checks exit 0, and checks that
the output line contains `PASS` and the real device name. GPU execution cannot
be validated by Linux CI.

## Honest limits / R6+

This is compute-only fill/gradient work. It is **not** a full GPU UI
rasterizer. There is no Vulkan, ANGLE, D3D, Metal presentation, texture atlas,
glyph rendering, image decoding, or dirty-rect integration. There is no
measured 120 fps claim, no benchmark claim, and no device conformance matrix:
only the system default Metal device is queried. A full GPU UI rasterizer,
other backends, presentation, and multi-device certification remain R6+ work.
No Metal API call in this backend is substituted with a software stub; the only
software implementation is the explicitly separate CPU conformance oracle.

## API notes

`objc2-metal 0.3.2` exposes `MTLCreateSystemDefaultDevice` behind its
`MTLDevice` feature and requires CoreGraphics to be linked; `src/metal.rs`
adds that framework link. All source compilation is performed at runtime by
`newLibraryWithSource`, so the build does not require a `.metallib` or an Xcode
Metal Toolchain. No API limitation required replacing a requested Metal call.
