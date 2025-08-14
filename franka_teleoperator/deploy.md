# VR IK Bridge Deployment Guide

## Quick Deploy (Recommended)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y libeigen3-dev build-essential

# Clone and install
git clone <your-repo>
cd franka_teleoperator
uv pip install -e .
```

## Alternative: Pre-built Wheel

```bash
# Build wheel (on dev machine)
python -m build --wheel

# Deploy wheel (on target)
uv pip install dist/vr_ik_bridge-*.whl
```

## Verify Installation

```python
# Test import
import vr_ik_bridge
print("VR IK Bridge installed successfully!")
```

## System Requirements

- Ubuntu 20.04+ (or compatible)
- Python 3.7+
- libeigen3-dev
- build-essential (for compilation)

## Integration with LeRobot

After installation, you can use:

```bash
python -m lerobot.record \
  --robot.type=franka_fer \
  --teleop.type=vr \
  ...
```

From any directory!