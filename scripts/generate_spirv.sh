#!/bin/sh
# Regenerate shaders/spirv/*.spv from shaders/glsl/*.comp with the local
# Khronos glslangValidator (provenance in shaders/README.md). The committed
# SPIR-V blobs are the build inputs; this script runs at development time
# only. Override the validator with OPUROM_GLSLANG=/path/to/glslangValidator.
set -eu

validator="${OPUROM_GLSLANG:-$HOME/tools/glslangValidator}"
if [ ! -x "$validator" ]; then
    echo "glslangValidator not found or not executable at $validator" >&2
    echo "see shaders/README.md for provenance" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
mkdir -p shaders/spirv
for src in shaders/glsl/*.comp; do
    name=$(basename "$src" .comp)
    "$validator" -V --target-env vulkan1.0 -o "shaders/spirv/$name.spv" "$src"
done
