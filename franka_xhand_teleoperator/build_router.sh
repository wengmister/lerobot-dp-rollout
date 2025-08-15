#!/bin/bash
# Build script for VR Message Router

echo "🔨 Building VR Message Router..."

# Clean previous builds
rm -f vr_message_router*.so
rm -rf build_router/

# Build the extension
python3 setup_router.py build_ext --inplace

# Check if build succeeded
if [ -f vr_message_router*.so ]; then
    echo "✅ VR Message Router built successfully!"
    echo "📁 Generated: $(ls vr_message_router*.so)"
else
    echo "❌ Build failed!"
    exit 1
fi