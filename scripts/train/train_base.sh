#!/bin/bash
# Train base model

"""Script to train VisionCapt base model."""

import os
import sys
from datetime import datetime

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

# Parameters
config_path = os.path.join(project_root, "configs/base_config.yaml")
output_dir = os.path.join(project_root, "checkpoints/base_model")
data_dir = os.path.join(project_root, "data/flickr8k_processed")
num_gpus = 1  # Set to number of available GPUs
log_level = "INFO"
seed = 42

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Create timestamp for unique run ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"base_training_{timestamp}"

# Command to run training
cmd = f"""
python -m visioncapt.train.train \
    --config {config_path} \
    --output_dir {output_dir} \
    --wandb \
    --wandb_name {run_name} \
    --wandb_project "visioncapt" \
    --fp16 \
    --seed {seed} \
    --log_level {log_level}
"""

# Execute command
print(f"Starting base model training with command: {cmd}")
exit_code = os.system(cmd)

if exit_code != 0:
    print(f"Training failed with exit code {exit_code}")
    sys.exit(exit_code)

print(f"Training complete. Model saved to {output_dir}")