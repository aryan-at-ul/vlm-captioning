# CLI interface for model inference

"""CLI for VisionCapt model."""

import os
import argparse
import torch
import logging
from PIL import Image
from typing import Optional, List, Dict
from omegaconf import OmegaConf

from visioncapt.model.builder import build_model
from visioncapt.utils.utils import setup_logging
from visioncapt.utils.image_utils import load_and_preprocess_image

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VisionCapt CLI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--image_path", type=str, help="Path to image")
    parser.add_argument("--output_path", type=str, help="Path to save output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--max_length", type=int, default=77, help="Maximum length of generated captions")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of generated captions")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of captions to generate")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing multiple images")
    parser.add_argument("--image_dir", type=str, help="Directory containing images to process")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for processing")
    return parser.parse_args()

def process_single_image(
    model: torch.nn.Module,
    image_path: str,
    args: argparse.Namespace
) -> List[str]:
    """
    Process a single image.
    
    Args:
        model: Model to use
        image_path: Path to image
        args: Command-line arguments
        
    Returns:
        List[str]: Generated captions
    """
    # Load and preprocess image
    image = load_and_preprocess_image(
        image_path,
        target_size=args.image_size,
        normalize=True,
        to_tensor=True,
        device=args.device
    )
    
    # Generate captions
    captions = model.generate_captions(
        images=image,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_return_sequences
    )
    
    # Process multiple return sequences
    if args.num_return_sequences > 1:
        all_captions = []
        for i in range(args.num_return_sequences):
            if i < len(captions):
                all_captions.append(captions[i])
        captions = all_captions
    
    return captions

def process_image_directory(
    model: torch.nn.Module,
    image_dir: str,
    args: argparse.Namespace
) -> Dict[str, List[str]]:
    """
    Process a directory of images.
    
    Args:
        model: Model to use
        image_dir: Directory containing images
        args: Command-line arguments
        
    Returns:
        Dict[str, List[str]]: Generated captions for each image
    """
    # Get all image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                image_files.append(os.path.join(root, file))
    
    # Process images
    results = {}
    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        
        # Load and preprocess images
        batch_images = []
        for image_path in batch_files:
            image = load_and_preprocess_image(
                image_path,
                target_size=args.image_size,
                normalize=True,
                to_tensor=True,
                device=args.device
            )
            batch_images.append(image)
        
        # Stack images if batch size > 1
        if len(batch_images) > 1:
            batch_tensor = torch.cat(batch_images, dim=0)
        else:
            batch_tensor = batch_images[0]
        
        # Generate captions
        captions = model.generate_captions(
            images=batch_tensor,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=1  # For batch processing, use 1 caption per image
        )
        
        # Store results
        for j, image_path in enumerate(batch_files):
            if j < len(captions):
                results[image_path] = [captions[j]]
    
    return results

def save_results(
    results: Dict[str, List[str]],
    output_path: str
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results to save
        output_path: Path to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save results
    with open(output_path, "w") as f:
        for image_path, captions in results.items():
            f.write(f"Image: {image_path}\n")
            for i, caption in enumerate(captions):
                f.write(f"  Caption {i+1}: {caption}\n")
            f.write("\n")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = build_model(config, model_path=args.model_path)
    model.to(args.device)
    model.eval()
    
    # Process images
    results = {}
    
    if args.image_dir:
        # Process directory of images
        logger.info(f"Processing images in {args.image_dir}")
        results = process_image_directory(model, args.image_dir, args)
    elif args.image_path:
        # Process single image
        logger.info(f"Processing image {args.image_path}")
        captions = process_single_image(model, args.image_path, args)
        results[args.image_path] = captions
    else:
        logger.error("No image path or directory provided. Use --image_path or --image_dir.")
        return
    
    # Print results
    for image_path, captions in results.items():
        print(f"Image: {image_path}")
        for i, caption in enumerate(captions):
            print(f"  Caption {i+1}: {caption}")
    
    # Save results if output path is provided
    if args.output_path:
        logger.info(f"Saving results to {args.output_path}")
        save_results(results, args.output_path)

if __name__ == "__main__":
    main()