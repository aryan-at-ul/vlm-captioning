# Entry point for training

"""Training script for VisionCapt model."""

import os
import argparse
import torch
import logging
import yaml
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any, Tuple

from visioncapt.model.builder import build_model
from visioncapt.utils.data_utils import create_flickr8k_dataloader
from visioncapt.utils.utils import (
    setup_logging, 
    set_seed, 
    get_optimizer, 
    get_scheduler, 
    format_dict,
    count_parameters
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train VisionCapt model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_name", type=str, help="Weights & Biases run name")
    parser.add_argument("--wandb_project", type=str, default="visioncapt", help="Weights & Biases project name")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision training")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def train(config, args):
    """
    Train VisionCapt model.
    
    Args:
        config: Training configuration
        args: Command-line arguments
    """
    # Set up output directory
    output_dir = args.output_dir or config.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    seed = args.seed or config.training.seed
    set_seed(seed)
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    logger.info(f"Training config: {format_dict(OmegaConf.to_container(config))}")
    
    # Initialize Weights & Biases
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"train_{int(time.time())}",
            config=OmegaConf.to_container(config, resolve=True),
        )
    
    # Set up distributed training
    is_distributed = args.local_rank != -1
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        world_size = torch.distributed.get_world_size()
        local_rank = args.local_rank
        logger.info(f"Initialized distributed training with world size: {world_size}, local rank: {local_rank}")
    else:
        world_size = 1
        local_rank = 0
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    
    # Log model info
    if local_rank == 0:
        total_params = count_parameters(model)
        logger.info(f"Model has {total_params:,} trainable parameters")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Get global step
        if "global_step" in checkpoint:
            start_step = checkpoint["global_step"]
            logger.info(f"Resuming from step {start_step}")
    
    # Get tokenizer and image processor
    from transformers import CLIPImageProcessor, AutoTokenizer
    
    # Load image processor
    vision_model_name = config.model.vision_encoder.model_name
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    
    # Load tokenizer
    language_model_name = config.model.language_model.model_name
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    
    # Add special tokens if needed
    special_tokens = {
        "additional_special_tokens": ["<image>", "</image>"],
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
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_flickr8k_dataloader(
        image_dir=config.data.image_dir,
        captions_file=config.data.train_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        batch_size=config.training.batch_size // world_size,
        max_seq_length=config.data.max_seq_length,
        split="train",
        shuffle=True,
        is_distributed=is_distributed,
        num_workers=4,
    )
    
    val_dataloader = None
    if os.path.exists(config.data.val_file):
        val_dataloader = create_flickr8k_dataloader(
            image_dir=config.data.image_dir,
            captions_file=config.data.val_file,
            tokenizer=tokenizer,
            image_processor=image_processor,
            batch_size=config.training.batch_size // world_size,
            max_seq_length=config.data.max_seq_length,
            split="val",
            shuffle=False,
            is_distributed=False,
            num_workers=2,
        )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = config.training.max_steps
    num_epochs = total_steps // steps_per_epoch + (1 if total_steps % steps_per_epoch > 0 else 0)
    
    logger.info(f"Training for {num_epochs} epochs, {total_steps} steps")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    
    # Set up optimizer and scheduler
    if hasattr(config.training, "optimizer"):
        optimizer = get_optimizer(model, config.training.optimizer)
    else:
        # Default optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    
    if hasattr(config.training, "scheduler"):
        scheduler = get_scheduler(
            optimizer,
            config.training.scheduler,
            num_training_steps=total_steps,
        )
    else:
        # Default scheduler
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps,
        )
    
    # Load optimizer and scheduler states if resuming
    if args.resume_from_checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Loaded optimizer state")
    
    if args.resume_from_checkpoint and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Loaded scheduler state")
    
    # Set up mixed precision training
    if args.fp16 or args.bf16:
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        logger.info(f"Using mixed precision training with {amp_dtype}")
    
    # Move model to device and wrap with DDP
    model.cuda()
    
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    
    # Set up training
    global_step = start_step
    tr_loss = 0.0
    logging_loss = 0.0
    best_val_loss = float("inf")
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    
    logger.info("***** Starting training *****")
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        pbar = tqdm(
            total=min(steps_per_epoch, total_steps - global_step),
            disable=local_rank != 0,
            desc=f"Epoch {epoch}"
        )
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps already processed
            if global_step >= total_steps:
                break
            
            # Move batch to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            if args.fp16 or args.bf16:
                with autocast(dtype=amp_dtype):
                    outputs = model(
                        images=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Update weights if gradient accumulation is complete
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
            else:
                outputs = model(
                    images=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if gradient accumulation is complete
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
            
            # Update loss tracking
            tr_loss += loss.item() * gradient_accumulation_steps
            
            # Update progress bar
            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Log metrics
            if local_rank == 0 and global_step % config.training.logging_steps == 0:
                # Calculate average loss
                avg_loss = (tr_loss - logging_loss) / config.training.logging_steps
                logging_loss = tr_loss
                
                logger.info(f"Step {global_step}: loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.2e}")
                
                if args.wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + step / len(train_dataloader),
                    }, step=global_step)
            
            # Evaluate model
            if (
                local_rank == 0 
                and val_dataloader is not None 
                and global_step % config.training.eval_steps == 0 
                and global_step > 0
            ):
                # Run evaluation
                eval_results = evaluate(model, val_dataloader, args)
                
                # Log evaluation results
                logger.info(f"Evaluation at step {global_step}: loss = {eval_results['loss']:.4f}")
                
                if args.wandb:
                    wandb.log({
                        "eval/loss": eval_results["loss"],
                        "eval/step": global_step,
                    }, step=global_step)
                
                # Save best model
                if eval_results["loss"] < best_val_loss:
                    best_val_loss = eval_results["loss"]
                    
                    # Save best model
                    best_model_dir = os.path.join(output_dir, "best_model")
                    logger.info(f"New best model! Saving to {best_model_dir}")
                    
                    save_model(
                        model, 
                        best_model_dir, 
                        tokenizer=tokenizer, 
                        is_distributed=is_distributed
                    )
                
                # Set model back to training mode
                model.train()
            
            # Save checkpoint
            if (
                local_rank == 0 
                and global_step % config.training.save_steps == 0 
                and global_step > 0
            ):
                # Save checkpoint
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    checkpoint_dir,
                    tokenizer=tokenizer,
                    is_distributed=is_distributed
                )
            
            # Check if we've reached the maximum number of steps
            if global_step >= total_steps:
                break
        
        pbar.close()
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Check if we've reached the maximum number of steps
        if global_step >= total_steps:
            break
    
    # Save final model
    if local_rank == 0:
        final_model_dir = os.path.join(output_dir, "final_model")
        logger.info(f"Saving final model to {final_model_dir}")
        
        save_model(
            model, 
            final_model_dir, 
            tokenizer=tokenizer, 
            is_distributed=is_distributed
        )
    
    # Return training metrics
    return {"train_loss": tr_loss / global_step}

def evaluate(model, dataloader, args):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        args: Command-line arguments
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Running evaluation...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Set up evaluation
    eval_loss = 0.0
    num_eval_steps = 0
    
    # Set up mixed precision evaluation
    amp_enabled = args.fp16 or args.bf16
    if amp_enabled:
        from torch.cuda.amp import autocast
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    # Evaluate
    for batch in tqdm(dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        # Move batch to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            if amp_enabled:
                with autocast(dtype=amp_dtype):
                    outputs = model(
                        images=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
            else:
                outputs = model(
                    images=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
        
        # Get loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Update metrics
        eval_loss += loss.item()
        num_eval_steps += 1
    
    # Calculate average metrics
    metrics = {
        "loss": eval_loss / max(num_eval_steps, 1),
    }
    
    return metrics

def save_model(model, output_dir, tokenizer=None, is_distributed=False):
    """
    Save model to output directory.
    
    Args:
        model: Model to save
        output_dir: Output directory
        tokenizer: Tokenizer to save
        is_distributed: Whether model is distributed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model to save
    model_to_save = model.module if is_distributed else model
    
    # Save model
    model_to_save.save_pretrained(output_dir)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    global_step, 
    output_dir, 
    tokenizer=None,
    is_distributed=False
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        global_step: Current global step
        output_dir: Output directory
        tokenizer: Tokenizer to save
        is_distributed: Whether model is distributed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model to save
    model_to_save = model.module if is_distributed else model
    
    # Save model
    model_to_save.save_pretrained(output_dir)
    
    # Save training state
    training_state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
    }
    
    torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Train model
    train(config, args)

if __name__ == "__main__":
    main()