from setuptools import setup, Extension
import subprocess
import sys
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Check if Eigen3 is available
def find_eigen():
    """Find Eigen3 installation"""
    try:
        # Try pkg-config first
        result = subprocess.run(['pkg-config', '--cflags', 'eigen3'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            include_dir = result.stdout.strip().replace('-I', '')
            return [include_dir]
    except:
        pass
    
    # Common Eigen installation paths
    common_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/opt/homebrew/include/eigen3',  # macOS Homebrew
        '/usr/include'  # Sometimes Eigen is directly in /usr/include
    ]
    
    for path in common_paths:
        if os.path.exists(os.path.join(path, 'Eigen')):
            return [path]
    
    print("Warning: Eigen3 not found in common locations")
    return []

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "vr_ik_bridge",
        [
            "vr_ik_bridge.cpp",
            "src/weighted_ik.cpp", 
            "src/geofik.cpp"
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Local include directory
            "include/",
            # Eigen3 include directories
            *find_eigen()
        ],
        language='c++'
    ),
]

setup(
    name="vr_ik_bridge",
    version="0.1.0",
    description="VR IK Bridge for Franka teleoperation with LeRobot",
    author="LeRobot Team",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy",
    ]
)