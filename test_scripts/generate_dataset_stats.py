#!/usr/bin/env python3
"""
Generate aggregated stats.json from dataset episodes_stats.jsonl
"""

import json
import numpy as np
from pathlib import Path
import torch

def aggregate_stats(episodes_stats_path):
    """Aggregate statistics across all episodes"""
    
    all_stats = {}
    
    with open(episodes_stats_path) as f:
        for line in f:
            episode = json.loads(line)
            episode_stats = episode["stats"]
            
            for key, stats in episode_stats.items():
                if key not in all_stats:
                    all_stats[key] = {
                        "min": [],
                        "max": [],
                        "mean": [],
                        "std": [],
                        "count": []
                    }
                
                # Handle different data types
                if isinstance(stats["min"], list):
                    # For nested lists (like images), just accumulate them
                    all_stats[key]["min"].append(stats["min"])
                    all_stats[key]["max"].append(stats["max"])
                    all_stats[key]["mean"].append(stats["mean"])
                    all_stats[key]["std"].append(stats["std"])
                else:
                    # For flat arrays
                    all_stats[key]["min"].append(stats["min"])
                    all_stats[key]["max"].append(stats["max"])
                    all_stats[key]["mean"].append(stats["mean"])
                    all_stats[key]["std"].append(stats["std"])
                
                all_stats[key]["count"].append(stats["count"][0])
    
    # Compute aggregated statistics
    aggregated = {}
    
    for key, stats in all_stats.items():
        if key.startswith("observation.images"):
            # For images, compute element-wise min/max/mean
            # Use the mean of means for simplicity
            aggregated[key] = {
                "mean": np.mean(stats["mean"], axis=0).tolist(),
                "std": np.mean(stats["std"], axis=0).tolist()
            }
        else:
            # For regular arrays
            counts = np.array(stats["count"])
            total_count = np.sum(counts)
            weights = counts / total_count
            
            # Element-wise min and max
            aggregated[key] = {
                "min": np.min(stats["min"], axis=0).tolist(),
                "max": np.max(stats["max"], axis=0).tolist(),
                "mean": np.average(stats["mean"], axis=0, weights=weights).tolist(),
                "std": np.average(stats["std"], axis=0, weights=weights).tolist()
            }
    
    return aggregated

def main():
    # Path to dataset
    dataset_path = Path("/home/zkweng/lerobot/datasets/orange_cube_pick_and_place")
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    
    print(f"Loading statistics from {episodes_stats_path}")
    aggregated_stats = aggregate_stats(episodes_stats_path)
    
    # Save to multiple locations
    output_paths = [
        Path("outputs/act_policy_orange_cube_bypass/checkpoints/stats.json"),
        Path("outputs/act_vision_orange_cube/stats.json")
    ]
    
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregated_stats, f, indent=2)
        print(f"Saved aggregated stats to {output_path}")

if __name__ == "__main__":
    main()