#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def push_dataset(dataset_name: str, username: str = "wengmister"):
    """Push a dataset to Hugging Face Hub with auto-generated paths."""
    repo_id = f"{username}/{dataset_name}"
    root = Path.home() / "lerobot" / "datasets" / dataset_name.replace("-", "_")
    
    if not root.exists():
        print(f"Error: Dataset directory does not exist: {root}")
        return False
    
    print(f"Pushing dataset:")
    print(f"  repo_id: {repo_id}")
    print(f"  root: {root}")
    
    dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
    dataset.push_to_hub()
    
    print(f"✅ Successfully pushed {repo_id} to Hub!")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python push_datasets.py <dataset_name>")
        print("Example: python push_datasets.py orange-cube-pick-and-place")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    push_dataset(dataset_name)