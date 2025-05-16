#!/bin/bash
# Evaluate model captions


"""Script to evaluate VisionCapt model on Flickr8k dataset."""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VisionCapt model on Flickr8k dataset")
    parser.add_argument("--model_path", type=str, default="checkpoints/base_model/final_model", 
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", 
                        help="Path to model config")
    parser.add_argument("--data_dir", type=str, default="data/flickr8k_processed", 
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for evaluation results")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate on")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--num_beams", type=int, default=5, 
                        help="Number of beams for beam search")
    parser.add_argument("--visualize", action="store_true", 
                        help="Whether to generate visualization images")
    parser.add_argument("--num_vis_samples", type=int, default=10, 
                        help="Number of samples to visualize")
    parser.add_argument("--save_captions", action="store_true", 
                        help="Whether to save generated captions")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        help="Logging level")
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Convert paths to absolute paths if necessary
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(project_root, args.model_path)
    if not os.path.isabs(args.config):
        args.config = os.path.join(project_root, args.config)
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for unique evaluation ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = f"eval_{args.split}_{timestamp}"
    eval_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Build command
    cmd = f"""
    python -m visioncapt.eval.evaluator \
        --model_path {args.model_path} \
        --config {args.config} \
        --data_dir {args.data_dir} \
        --output_dir {eval_dir} \
        --split {args.split} \
        --device {args.device} \
        --batch_size {args.batch_size} \
        --num_beams {args.num_beams} \
        --log_level {args.log_level}
    """
    
    # Add optional arguments
    if args.max_samples is not None:
        cmd += f" --max_samples {args.max_samples}"
    
    if args.visualize:
        cmd += f" --visualize --num_vis_samples {args.num_vis_samples}"
    
    if args.save_captions:
        cmd += " --save_captions"
    
    # Execute command
    print(f"Running evaluation with command: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"Evaluation failed with exit code {exit_code}")
        sys.exit(exit_code)
    
    print(f"Evaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    main()