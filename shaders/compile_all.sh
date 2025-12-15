#!/bin/bash
# Compile all GLSL compute shaders to SPIR-V

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling PagedAttention shaders..."

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Please install Vulkan SDK."
    echo "  Ubuntu: sudo apt install vulkan-sdk"
    echo "  Or download from: https://vulkan.lunarg.com/sdk/home"
    exit 1
fi

# List of shaders to compile
SHADERS=(
    "attention_f32_fast.comp"
    "attention_f16.comp"
    "attention_paged.comp"
    "copy_kv_to_paged.comp"
    "radix_count.comp"
    "radix_scatter_u16.comp"
)

COMPILED=0
FAILED=0

for shader in "${SHADERS[@]}"; do
    if [ -f "$shader" ]; then
        output="${shader%.comp}.spv"
        echo -n "  Compiling $shader → $output ... "

        if glslc "$shader" -o "$output" 2>&1 | tee /tmp/glslc_error.log; then
            size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output")
            size_kb=$((size / 1024))
            echo "OK (${size_kb}KB)"
            ((COMPILED++))
        else
            echo "FAILED"
            cat /tmp/glslc_error.log
            ((FAILED++))
        fi
    else
        echo "  Warning: $shader not found, skipping"
    fi
done

echo ""
echo "Summary: $COMPILED compiled, $FAILED failed"

if [ $FAILED -gt 0 ]; then
    echo "Error: Some shaders failed to compile"
    exit 1
fi

echo "All shaders compiled successfully!"
exit 0
