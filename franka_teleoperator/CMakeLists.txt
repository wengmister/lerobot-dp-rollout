cmake_minimum_required(VERSION 3.12)
project(vr_ik_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add source files
set(SOURCES
    src/weighted_ik.cpp
    src/geofik.cpp
    vr_ik_bridge.cpp
)

# Create pybind11 module
pybind11_add_module(vr_ik_bridge ${SOURCES})

# Set compiler flags
target_compile_features(vr_ik_bridge PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(vr_ik_bridge PRIVATE Eigen3::Eigen)

# Set module properties
set_target_properties(vr_ik_bridge PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Install the module
install(TARGETS vr_ik_bridge DESTINATION .)