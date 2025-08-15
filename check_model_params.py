#!/usr/bin/env python3

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def count_parameters(model):
    """Count the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def format_param_count(count):
    """Format parameter count in human readable format"""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)

def analyze_model(model_path, dataset_path=None):
    """Analyze a model and print parameter information"""
    print(f"\nAnalyzing model: {model_path}")
    print("=" * 50)
    
    try:
        # Load policy config
        config = PreTrainedConfig.from_pretrained(model_path)
        print(f"Policy type: {config.type}")
        
        # Load dataset if provided for metadata
        if dataset_path:
            dataset = LeRobotDataset(dataset_path, root='./datasets/franka_wave')
            policy = make_policy(config, ds_meta=dataset.meta)
        else:
            # Try to load without dataset metadata
            policy = make_policy(config)
        
        # Count parameters
        total_params, trainable_params = count_parameters(policy)
        
        print(f"Total parameters: {format_param_count(total_params)} ({total_params:,})")
        print(f"Trainable parameters: {format_param_count(trainable_params)} ({trainable_params:,})")
        
        # Print model architecture summary
        print(f"\nModel architecture summary:")
        print(f"- Config: {type(config).__name__}")
        
        # Policy-specific details
        if hasattr(config, 'n_action_steps'):
            print(f"- Action steps: {config.n_action_steps}")
        if hasattr(config, 'n_obs_steps'):
            print(f"- Observation steps: {config.n_obs_steps}")
        if hasattr(config, 'chunk_size'):
            print(f"- Chunk size: {config.chunk_size}")
        if hasattr(config, 'horizon'):
            print(f"- Horizon: {config.horizon}")
        
        # Model size estimation
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        print(f"- Estimated model size: {model_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

def main():
    print("LeRobot Model Parameter Analysis")
    print("=" * 50)
    
    # Check diffusion policy
    diffusion_path = "./outputs/diffusion_policy_franka_wave/checkpoints/050000/pretrained_model"
    dataset_path = "local_dataset_franka_wave"
    
    analyze_model(diffusion_path, dataset_path)
    
    # Check if ACT model exists
    act_path = "./outputs/act_policy_franka_wave/checkpoints/090000/pretrained_model"
    try:
        analyze_model(act_path, dataset_path)
    except:
        print(f"\nACT model not found at: {act_path}")
        print("Train an ACT model first to compare parameters.")
    
    # You can also create a quick comparison of different policy types
    print(f"\n" + "=" * 50)
    print("Policy Type Comparison (typical parameter counts):")
    print("- Diffusion: 10-100M+ parameters (slow inference)")
    print("- ACT: 1-20M parameters (fast inference)")
    print("- SmolVLA: 100M-1B+ parameters (variable)")
    print("- Simple MLP: <1M parameters (very fast)")

if __name__ == "__main__":
    main()