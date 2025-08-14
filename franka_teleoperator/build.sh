#!/bin/bash
# Build script for VR IK Bridge

set -e

echo "Building VR IK Bridge for LeRobot..."

# Check if required dependencies are available
echo "Checking dependencies..."

# Check for pybind11
python3 -c "import pybind11" 2>/dev/null || {
    echo "Error: pybind11 not found. Install with: pip install pybind11"
    exit 1
}

# Check for Eigen3
if ! pkg-config --exists eigen3 2>/dev/null; then
    if [ ! -d "/usr/include/eigen3" ] && [ ! -d "/usr/local/include/eigen3" ]; then
        echo "Error: Eigen3 not found. Install with:"
        echo "  Ubuntu/Debian: sudo apt install libeigen3-dev"
        echo "  macOS: brew install eigen"
        exit 1
    fi
fi

# Check if source files exist
if [ ! -f "src/weighted_ik.cpp" ] || [ ! -f "src/geofik.cpp" ]; then
    echo "Error: Source files not found. Please ensure weighted_ik.cpp and geofik.cpp are in src/"
    exit 1
fi

# Build the module
echo "Building pybind11 module..."
python3 setup.py build_ext --inplace

# Check if build was successful
if ls vr_ik_bridge*.so 1> /dev/null 2>&1 || ls vr_ik_bridge*.pyd 1> /dev/null 2>&1; then
    echo "✓ Build successful!"
    echo ""
    echo "Usage:"
    echo "  1. Make sure your VR app is connected via adb"
    echo "  2. Run: python test_vr_integration.py"
    echo ""
    echo "Integration with LeRobot:"
    echo "  Use VRTeleoperator in lerobot-record with --teleop.type=vr"
else
    echo "✗ Build failed!"
    exit 1
fi