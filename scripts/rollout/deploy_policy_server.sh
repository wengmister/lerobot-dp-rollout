#!/bin/bash

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Received interrupt signal - cleaning up..."
    
    # Kill the robot client if it's still running
    if [ ! -z "$ROBOT_CLIENT_PID" ]; then
        echo "Stopping robot client (PID: $ROBOT_CLIENT_PID)..."
        kill $ROBOT_CLIENT_PID 2>/dev/null
        wait $ROBOT_CLIENT_PID 2>/dev/null
    fi
    
    # Kill the policy server
    if [ ! -z "$POLICY_SERVER_PID" ]; then
        echo "Stopping policy server (PID: $POLICY_SERVER_PID)..."
        kill $POLICY_SERVER_PID 2>/dev/null
        wait $POLICY_SERVER_PID 2>/dev/null
    fi
    
    echo "Cleanup completed"
    exit 0
}

# Set up signal handlers to cleanup on exit
trap cleanup SIGINT SIGTERM EXIT

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

# Start robot client (handles robot control with automatic reset)
echo "Starting robot client..."
echo "Press Ctrl+C to stop the deployment at any time"

# Start robot client in background so we can handle signals properly
python -m lerobot.scripts.server.robot_client \
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
  --task="Policy deployment test" &

ROBOT_CLIENT_PID=$!
echo "Robot client started with PID: $ROBOT_CLIENT_PID"

# Set up a timeout mechanism that doesn't interfere with signal handling
TIMEOUT_DURATION=30
echo "Robot client will run for maximum $TIMEOUT_DURATION seconds..."

# Wait for either the process to finish or timeout
ELAPSED=0
while kill -0 $ROBOT_CLIENT_PID 2>/dev/null && [ $ELAPSED -lt $TIMEOUT_DURATION ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# Check if process is still running (hit timeout)
if kill -0 $ROBOT_CLIENT_PID 2>/dev/null; then
    echo "Timeout reached - stopping robot client..."
    kill $ROBOT_CLIENT_PID 2>/dev/null
    wait $ROBOT_CLIENT_PID 2>/dev/null
fi

echo "Robot client finished"

# Cleanup
echo "Stopping policy server..."
kill $POLICY_SERVER_PID