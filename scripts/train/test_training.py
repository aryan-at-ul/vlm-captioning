"""Script to test training on a small subset of the data."""

import os
import sys
import argparse
import torch
import logging
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor, AutoTokenizer

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from visioncapt.model.builder import build_model
from visioncapt.utils.data_utils import create_flickr8k_dataloader
from visioncapt.utils.utils import setup_logging, set_seed
from visioncapt.train.trainer import VisionCaptTrainer

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test VisionCapt training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/flickr8k_processed", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/test_training", help="Path to output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use for testing")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps to train for")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
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
    
    # Override config settings for testing
    config.training.batch_size = args.batch_size
    config.training.max_steps = args.num_steps
    config.training.logging_steps = 1
    config.training.save_steps = args.num_steps // 2
    config.training.eval_steps = args.num_steps // 2
    
    # Update model config for quantization if requested
    if args.load_4bit:
        logger.info("Using 4-bit quantization")
        if not hasattr(config.model.language_model, "load_in_4bit"):
            config.model.language_model.load_in_4bit = True
            config.model.language_model.load_in_8bit = False
    
    if args.load_8bit:
        logger.info("Using 8-bit quantization")
        if not hasattr(config.model.language_model, "load_in_8bit"):
            config.model.language_model.load_in_8bit = True
            config.model.language_model.load_in_4bit = False
            
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    logger.info(f"Model built. Moving to {args.device}...")
    model = model.to(args.device)
    
    # Get tokenizer and image processor
    language_model_name = config.model.language_model.model_name
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    
    # Add special tokens if needed
    special_tokens = {
        "additional_special_tokens": ["<image>", "</image>", "<|startoftext|>", "<|endoftext|>"]
    }
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            special_tokens["pad_token"] = tokenizer.eos_token
        else:
            special_tokens["pad_token"] = "[PAD]"
    
    # Update tokenizer with special tokens
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens to tokenizer")
        
        # Resize token embeddings
        if hasattr(model.language_model, "model"):
            model.language_model.model.resize_token_embeddings(len(tokenizer))
    
    # Get image processor
    vision_model_name = config.model.vision_encoder.model_name
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    
    # Create dataloaders with a subset of data
    logger.info("Creating dataloaders...")
    # Modify the paths to use the data directory
    train_file = os.path.join(args.data_dir, os.path.basename(config.data.train_file))
    val_file = os.path.join(args.data_dir, os.path.basename(config.data.val_file))
    image_dir = os.path.join(args.data_dir, "images")
    
    # Load a subset of data for testing
    logger.info(f"Loading {args.num_samples} samples for testing...")
    import pandas as pd
    
    # Load train data
    train_df = pd.read_csv(train_file)
    train_df = train_df.head(args.num_samples)  # Take only a few samples
    
    # Save subset for testing
    test_train_file = os.path.join(args.output_dir, "test_train.csv")
    train_df.to_csv(test_train_file, index=False)
    
    # Load val data
    val_df = pd.read_csv(val_file)
    val_df = val_df.head(min(args.num_samples // 2, len(val_df)))  # Take only a few samples
    
    # Save subset for testing
    test_val_file = os.path.join(args.output_dir, "test_val.csv")
    val_df.to_csv(test_val_file, index=False)
    
    # Create dataloaders
    train_dataloader = create_flickr8k_dataloader(
        image_dir=image_dir,
        captions_file=test_train_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size,
        max_seq_length=config.data.max_seq_length,
        split="train",
        shuffle=True,
        is_distributed=False,
        num_workers=0,  # Use 0 workers for testing
    )
    
    val_dataloader = create_flickr8k_dataloader(
        image_dir=image_dir,
        captions_file=test_val_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=args.batch_size,
        max_seq_length=config.data.max_seq_length,
        split="val",
        shuffle=False,
        is_distributed=False,
        num_workers=0,  # Use 0 workers for testing
    )
    

    # stabilize for sanity-check
    config.training.learning_rate = 1e-6
    config.training.warmup_steps   = 0
    config.training.gradient_accumulation_steps = 2
    args.fp16 = False
    args.load_8bit = False
    args.load_4bit = False


    # Create trainer
    logger.info("Creating trainer...")
    trainer = VisionCaptTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        max_steps=args.num_steps,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        fp16=args.fp16,
        local_rank=-1,  # No distributed training for testing
        seed=args.seed,
        use_wandb=False,  # No wandb for testing
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        metrics = trainer.train()
        logger.info(f"Training complete. Metrics: {metrics}")
        logger.info(f"Test passed! The model can be trained.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error("Test failed! Please check the error message above.")
        raise

if __name__ == "__main__":
    main()