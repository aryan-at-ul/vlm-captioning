"""Script to test model inference on sample images."""

import os
import sys
import argparse
import torch
import logging
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from visioncapt.model.builder import build_model
from visioncapt.utils.utils import setup_logging, set_seed
from visioncapt.utils.image_utils import load_and_preprocess_image

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test VisionCapt inference")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (optional)")
    parser.add_argument("--image_dir", type=str, default="data/flickr8k_processed/images", help="Path to directory with test images")
    parser.add_argument("--output_dir", type=str, default="results/test_inference", help="Path to output directory")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(project_root, args.config)
    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    
    # Update model config for quantization if requested
    if args.load_4bit:
        logger.info("Using 4-bit quantization")
        if not hasattr(config.model.language_model, "load_in_4bit"):
            config.model.language_model.load_in_4bit = True
    
    if args.load_8bit:
        logger.info("Using 8-bit quantization")
        if not hasattr(config.model.language_model, "load_in_8bit"):
            config.model.language_model.load_in_8bit = True
    
    # Build model
    logger.info("Building model...")
    if args.model_path:
        model = build_model(config, model_path=args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        model = build_model(config)
        logger.info("Created new model from config")
    
    # Move model to device
    model.to(args.device)
    model.eval()
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root, file))
    
    # Randomly select images
    import random
    random.seed(args.seed)
    if len(image_files) > args.num_images:
        image_files = random.sample(image_files, args.num_images)
    
    logger.info(f"Testing inference on {len(image_files)} images...")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path}")
        
        # Load and preprocess image
        image = load_and_preprocess_image(
            image_path,
            target_size=224,
            normalize=True,
            to_tensor=True,
            device=args.device
        )
        
        # Generate caption
        with torch.no_grad():
            captions = model.generate_captions(
                images=image,
                max_length=77,
                num_beams=5,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0,
            )
        
        # Display results
        caption = captions[0]
        logger.info(f"Generated caption: {caption}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display image
        pil_image = Image.open(image_path).convert("RGB")
        ax.imshow(pil_image)
        
        # Set title with generated caption
        ax.set_title(f"Caption: {caption}", fontsize=12)
        
        # Remove axes
        ax.axis("off")
        
        # Save figure
        output_path = os.path.join(args.output_dir, f"result_{i}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved result to {output_path}")
    
    logger.info(f"Inference test complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()