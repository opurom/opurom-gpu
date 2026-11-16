# SPIR-V kernels

The eight compute kernels here are one-to-one GLSL ports of the MSL kernels in
`src/metal.rs` (`raster_batch`, `snapshot_frame`, `build_mask`, `effect_local`,
`blur_horizontal`, `blur_vertical`, `gaussian_horizontal`, `gaussian_vertical`).
The integer pixel contract is shared: every intermediate that MSL declares
`long` is `int64_t` here, unsigned mixing stays `uint`, and rounding, clamping
and blend formulas are byte-for-byte the same, so the Vulkan readback must
match the CPU oracle and the Metal readback byte for byte.

## Buffers and bindings

| Binding | Content | Format |
| --- | --- | --- |
| 0 | frame / horizontal-blur source | `uint` per pixel, RGBA8 packed little-endian (`r \| g<<8 \| b<<16 \| a<<24`) |
| 1 | secondary / blur target | same packing |
| 2 | backdrop snapshot | same packing |
| 3 | silhouette mask | `uint` per pixel, `0` or `255` |
| 4 | packed command words | the same `u32` stream the Metal path uploads |

Pass parameters (28 `u32` words, `PARAM_WORDS`) arrive as push constants. The
host swaps bindings 0 and 1 between two descriptor sets exactly like the Metal
ping-pong: `effect_local` and the horizontal blurs read 0 and write 1, the
vertical blurs read 1 and write 0, and the frame is always read back from the
current primary.

## Provenance and regeneration

GLSL 4.50 compute, compiled to SPIR-V 1.5 with `--target-env vulkan1.0` by the
Khronos glslangValidator **11.0.0** (Khronos Tool ID 8). The validator binary
is a one-time local tool, not a build or cargo dependency: it is the
`glslangValidator` shipped inside the Android SDK emulator component
(`emulator/lib64/vulkan/glslangValidator`), copied to `~/tools/`. Regenerate
the committed blobs after editing any `.comp` source:

```sh
./scripts/generate_spirv.sh            # uses ~/tools/glslangValidator
# or: OPUROM_GLSLANG=/path/to/glslangValidator ./scripts/generate_spirv.sh
```

The committed `shaders/spirv/*.spv` blobs are the build inputs (they are
embedded with `include_bytes!` at compile time, so no runtime file access and
no runtime shader compilation). `tests/spirv.rs` pins them: the blobs must
exist with the SPIR-V magic, and when a validator is available on the host the
test recompiles every source and compares bytes, so a stale artifact fails the
suite instead of drifting silently.
