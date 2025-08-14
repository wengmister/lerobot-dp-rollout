#!/bin/bash

# Start policy server (runs inference in background)
echo "Starting policy server..."
python -m lerobot.scripts.server.policy_server \
  --host=localhost \
  --port=8080 \
  --fps=30 &

POLICY_SERVER_PID=$!
echo "Policy server started with PID: $POLICY_SERVER_PID"

# Wait a moment for server to start
sleep 3

# Start robot client (handles robot control)
echo "Starting robot client..."
python -m lerobot.scripts.server.robot_client \
  --policy_type=diffusion \
  --pretrained_name_or_path=./outputs/diffusion_policy_franka_wave/checkpoints/050000/pretrained_model \
  --robot.type=franka_fer \
  --robot.server_ip=192.168.18.1 \
  --robot.server_port=5000 \
  --server_address=localhost:8080 \
  --policy_device=cuda \
  --fps=30 \
  --actions_per_chunk=8 \
  --debug_visualize_queue_size=True \
  --task="Policy deployment test"

# Cleanup
echo "Stopping policy server..."
kill $POLICY_SERVER_PID