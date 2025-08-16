# Dual VR Recording Script

Direct Python script for recording data with Franka FER + XHand using dual VR teleoperator.

## Quick Start

```bash
# Start new dataset
python dual_vr_record_direct.py --dataset-name my_task --num-episodes 50

# Resume existing dataset 
python dual_vr_record_direct.py --dataset-name my_task --resume --num-episodes 100
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset-name` | Name of the dataset | `orange_cube_pick_and_place` |
| `--num-episodes` | **Total episodes desired** (not additional) | `100` |
| `--episode-time` | Duration of each episode (seconds) | `30` |
| `--task` | Task description | `"Teleoperate dual arm-hand system..."` |
| `--fps` | Recording frame rate | `30` |
| `--resume` | Resume from existing dataset | `false` |

## Resume Functionality

**Important:** `--num-episodes` is the **total target**, not additional episodes.

### Example:
- You have 3 episodes recorded (0, 1, 2)
- You want 10 total episodes
- Use: `--num-episodes 10` (will record 7 more episodes: 3, 4, 5, 6, 7, 8, 9)

## Recording Process

1. **Manual confirmation** before each episode
2. **Robot homing** (arm and hand return to safe positions)
3. **VR connection setup** and reference frame reset
4. **Episode recording** with teleoperation control
5. **Episode saving** and teleoperator cleanup

## Controls During Recording

- **Press 's'** - Stop recording
- **Press 'r'** - Re-record current episode
- **Ctrl+C** - Emergency stop and cleanup

## Dataset Structure

```
my_dataset/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── episode_000001.parquet
├── images/
│   ├── observation.images.tpv/
│   └── observation.images.wrist/
├── meta/
│   ├── info.json
│   └── episodes.jsonl
└── videos/
```

## Prerequisites

- VR headset connected and hand tracking app running
- Franka robot connected to `192.168.18.1:5000`
- XHand connected via RS485 on `/dev/ttyUSB0`
- Cameras configured on `/dev/video18` (TPV) and `/dev/video4` (wrist)

## Troubleshooting

- **CRC errors** are automatically suppressed for smooth recording
- **VR timeout increased to 1000ms** to handle periodic Quest delays
- **Manual homing** between episodes ensures consistent starting positions
- **VR reference reset** before each episode for accurate control