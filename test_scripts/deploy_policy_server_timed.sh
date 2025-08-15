#!/bin/bash

# Usage: ./deploy_policy_server_timed.sh [duration_seconds]
DURATION=${1:-20}  # Default to 10 seconds if no argument provided

echo "Starting policy server for ${DURATION} seconds..."

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

# Start robot client with timeout
echo "Starting robot client for ${DURATION} seconds..."
timeout ${DURATION}s python -m lerobot.scripts.server.robot_client \
  --policy_type=act \
  --pretrained_name_or_path=./outputs/act_policy_franka_wave/checkpoints/090000/pretrained_model \
  --robot.type=franka_fer \
  --robot.server_ip=192.168.18.1 \
  --robot.server_port=5000 \
  --server_address=localhost:8080 \
  --policy_device=cuda \
  --fps=30 \
  --actions_per_chunk=8 \
  --debug_visualize_queue_size=True \
  --task="Policy deployment test"

echo "Robot client finished after ${DURATION} seconds"

# Cleanup
echo "Stopping policy server..."
kill $POLICY_SERVER_PID
wait $POLICY_SERVER_PID 2>/dev/null

echo "Deployment completed successfully!"