#!/bin/bash
# Build script for lightweight WeightedIK bridge

echo "Building lightweight WeightedIK bridge..."

# Clean previous build
rm -f weighted_ik_bridge*.so

# Build with g++
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I include \
    -I /usr/include/eigen3 \
    $(python3 -m pybind11 --includes) \
    weighted_ik_bridge.cpp \
    src/weighted_ik.cpp \
    src/geofik.cpp \
    -o weighted_ik_bridge$(python3-config --extension-suffix)

if [ $? -eq 0 ]; then
    echo "WeightedIK bridge built successfully!"
    echo "Output: weighted_ik_bridge$(python3-config --extension-suffix)"
else
    echo "Build failed!"
    exit 1
fi