#!/bin/bash
# Compile GLSL compute shaders to SPIR-V
# Requires glslc from Vulkan SDK

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for shader in "$SCRIPT_DIR"/*.comp; do
    [ -e "$shader" ] || continue
    name=$(basename "$shader" .comp)
    echo "Compiling $name.comp -> $name.spv"
    glslc -O --target-env=vulkan1.2 -o "$SCRIPT_DIR/$name.spv" "$shader"
done

echo "Done."
