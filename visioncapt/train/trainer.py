# Trainer class

"""Trainer class for VisionCapt model."""

import os
import logging
import time
import math
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
import wandb
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from visioncapt.utils.utils import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)

class VisionCaptTrainer:
    """Trainer for VisionCapt model."""
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "checkpoints",
        device: Optional[torch.device] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        max_steps: int = -1,
        max_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 500,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        fp16: bool = False,
        bf16: bool = False,
        local_rank: int = -1,
        seed: int = 42,
        use_wandb: bool = False,
        wandb_project: str = "visioncapt",
        wandb_name: Optional[str] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            config: Training configuration
            output_dir: Output directory for checkpoints
            device: Device to use
            optimizer: Optimizer to use
            scheduler: Learning rate scheduler
            max_steps: Maximum number of training steps
            max_epochs: Maximum number of training epochs
            gradient_accumulation_steps: Number of gradient accumulation steps
            warmup_steps: Number of warmup steps
            logging_steps: Number of steps between logging
            save_steps: Number of steps between saving checkpoints
            eval_steps: Number of steps between evaluation
            learning_rate: Learning rate
            weight_decay: Weight decay
            fp16: Whether to use mixed precision training (FP16)
            bf16: Whether to use mixed precision training (BF16)
            local_rank: Local rank for distributed training
            seed: Random seed
            use_wandb: Whether to use Weights & Biases
            wandb_project: Weights & Biases project name
            wandb_name: Weights & Biases run name
            callbacks: List of callback functions
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.bf16 = bf16
        self.local_rank = local_rank
        self.seed = seed
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.callbacks = callbacks or []
        
        # Set up distributed training
        self.is_distributed = local_rank != -1
        self.world_size = 1
        
        if self.is_distributed:
            self.world_size = torch.distributed.get_world_size()
        
        # Initialize optimizer
        self.optimizer = optimizer or self._create_optimizer()
        
        # Calculate number of training steps
        if self.max_steps > 0:
            self.t_total = self.max_steps
            self.max_epochs = self.max_steps // len(self.train_dataloader) + 1
        else:
            self.t_total = len(self.train_dataloader) * self.max_epochs
            self.max_steps = self.t_total
        
        # Initialize scheduler
        self.scheduler = scheduler or self._create_scheduler()
        
        # Initialize mixed precision training
        self.scaler = None
        if self.fp16 or self.bf16:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        
        # Move model to device and wrap with DDP
        self.model.to(self.device)
        
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        
        # Create output directory
        if self.local_rank in [-1, 0]:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Weights & Biases
        if self.use_wandb and self.local_rank in [-1, 0]:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name or f"train_{int(time.time())}",
                config=self.config,
            )
    
    def _create_optimizer(self):
        """
        Create optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        if hasattr(self.config, "optimizer"):
            return get_optimizer(self.model, self.config.optimizer)
        else:
            # Default: AdamW with weight decay
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            return torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
            )
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler.
        
        Returns:
            Any: Learning rate scheduler
        """
        if hasattr(self.config, "scheduler"):
            return get_scheduler(
                self.optimizer,
                self.config.scheduler,
                num_training_steps=self.t_total,
            )
        else:
            # Default: Linear warmup and decay
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.t_total,
            )
    
    def save_checkpoint(self, global_step, output_dir=None):
        """
        Save checkpoint.
        
        Args:
            global_step: Current training step
            output_dir: Output directory
        """
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model to save
        model_to_save = self.model.module if self.is_distributed else self.model
        
        # Save model
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(output_dir)
        else:
            # Fallback to saving state dict
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "global_step": global_step,
            },
            os.path.join(output_dir, "optimizer.pt"),
        )
        
        logger.info(f"Saved checkpoint to {output_dir}")
    
    def train(self):
        """
        Train model.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        # Log training parameters
        if self.local_rank in [-1, 0]:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
            logger.info(f"  Num epochs = {self.max_epochs}")
            logger.info(f"  Batch size = {self.train_dataloader.batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {self.t_total}")
            logger.info(f"  Learning rate = {self.learning_rate}")
            logger.info(f"  Mixed precision training = {self.fp16 or self.bf16}")
        
        # Training loop variables
        global_step = 0
        epoch = 0
        tr_loss = 0.0
        self.model.zero_grad()
        best_eval_loss = float("inf")
        
        # Training loop
        train_iterator = range(self.max_epochs)
        for epoch in train_iterator:
            epoch_start_time = time.time()
            
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            epoch_iterator = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                disable=self.local_rank not in [-1, 0],
                total=len(self.train_dataloader),
            )
            
            # Reset metrics
            epoch_loss = 0.0
            steps_in_epoch = 0
            
            for step, batch in enumerate(epoch_iterator):
                # Set model to training mode
                self.model.train()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                if self.fp16 or self.bf16:
                    from torch.cuda.amp import autocast
                    amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
                    
                    with autocast(dtype=amp_dtype):
                        outputs = self.model(
                            images=batch["image"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Update parameters if gradient accumulation is complete
                    if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                        # Clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Update parameters
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.model.zero_grad()
                        
                        global_step += 1
                else:
                    outputs = self.model(
                        images=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update parameters if gradient accumulation is complete
                    if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Update parameters
                        self.optimizer.step()
                        self.scheduler.step()
                        self.model.zero_grad()
                        
                        global_step += 1
                
                # Track loss
                tr_loss += loss.item() * self.gradient_accumulation_steps
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                steps_in_epoch += 1
                
                # Update progress bar
                if self.local_rank in [-1, 0]:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate
                    epoch_iterator.set_postfix({
                        "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    })
                
                # Log metrics
                if self.local_rank in [-1, 0] and self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    # Log to Weights & Biases if enabled
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": tr_loss / global_step,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                            "train/step": global_step,
                        })
                
                # Evaluate model
                if self.val_dataloader is not None and self.eval_steps > 0 and global_step % self.eval_steps == 0:
                    eval_results = self.evaluate()
                    
                    if self.local_rank in [-1, 0]:
                        logger.info(f"Evaluation results at step {global_step}: {eval_results}")
                        
                        # Log to Weights & Biases if enabled
                        if self.use_wandb:
                            wandb.log({
                                f"eval/{k}": v for k, v in eval_results.items()
                            }, step=global_step)
                        
                        # Save best model
                        if eval_results.get("loss", float("inf")) < best_eval_loss:
                            best_eval_loss = eval_results["loss"]
                            
                            # Save best model
                            best_model_dir = os.path.join(self.output_dir, "best_model")
                            logger.info(f"New best model (loss: {best_eval_loss:.4f})! Saving to {best_model_dir}")
                            
                            self.save_checkpoint(global_step, output_dir=best_model_dir)
                
                # Save checkpoint
                if self.local_rank in [-1, 0] and self.save_steps > 0 and global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step)
                
                # Check if we reached the maximum number of steps
                if global_step >= self.max_steps:
                    epoch_iterator.close()
                    break
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / steps_in_epoch
            
            if self.local_rank in [-1, 0]:
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, average loss: {avg_epoch_loss:.4f}")
                
                # Log to Weights & Biases if enabled
                if self.use_wandb:
                    wandb.log({
                        "train/epoch": epoch,
                        "train/epoch_loss": avg_epoch_loss,
                        "train/epoch_time": epoch_time,
                    })
            
            # Call epoch-end callbacks
            for callback in self.callbacks:
                callback(self, epoch=epoch, global_step=global_step)
            
            # Check if we reached the maximum number of steps
            if global_step >= self.max_steps:
                break
        
        # Save final model
        if self.local_rank in [-1, 0]:
            final_model_dir = os.path.join(self.output_dir, "final_model")
            logger.info(f"Training complete. Saving final model to {final_model_dir}")
            
            self.save_checkpoint(global_step, output_dir=final_model_dir)
        
        # Return training metrics
        return {
            "global_step": global_step,
            "train_loss": tr_loss / global_step,
            "best_eval_loss": best_eval_loss,
        }
    
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.val_dataloader is None:
            return {"loss": float("inf")}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation metrics
        eval_loss = 0.0
        num_eval_steps = 0
        
        # Evaluation loop
        logger.info("Running evaluation...")
        for batch in tqdm(
            self.val_dataloader,
            desc="Evaluating",
            disable=self.local_rank not in [-1, 0],
        ):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                if self.fp16 or self.bf16:
                    from torch.cuda.amp import autocast
                    amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
                    
                    with autocast(dtype=amp_dtype):
                        outputs = self.model(
                            images=batch["image"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                else:
                    outputs = self.model(
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
        eval_loss = eval_loss / num_eval_steps
        
        # Return evaluation metrics
        return {
            "loss": eval_loss,
        }
    
    def predict(self, dataloader):
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: Dataloader
            
        Returns:
            List[Dict[str, Any]]: Predictions
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Prediction results
        predictions = []
        
        # Prediction loop
        logger.info("Running prediction...")
        for batch in tqdm(
            dataloader,
            desc="Predicting",
            disable=self.local_rank not in [-1, 0],
        ):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                if self.fp16 or self.bf16:
                    from torch.cuda.amp import autocast
                    amp_dtype = torch.bfloat16 if self.bf16 else torch.float16
                    
                    with autocast(dtype=amp_dtype):
                        # Generate captions
                        captions = self.model.generate_captions(
                            images=batch["image"],
                            max_length=self.config.get("max_length", 77),
                            num_beams=self.config.get("num_beams", 5),
                            temperature=self.config.get("temperature", 1.0),
                            top_p=self.config.get("top_p", 0.9),
                            top_k=self.config.get("top_k", 50),
                            repetition_penalty=self.config.get("repetition_penalty", 1.0),
                        )
                else:
                    # Generate captions
                    captions = self.model.generate_captions(
                        images=batch["image"],
                        max_length=self.config.get("max_length", 77),
                        num_beams=self.config.get("num_beams", 5),
                        temperature=self.config.get("temperature", 1.0),
                        top_p=self.config.get("top_p", 0.9),
                        top_k=self.config.get("top_k", 50),
                        repetition_penalty=self.config.get("repetition_penalty", 1.0),
                    )
            
            # Store predictions
            for i, caption in enumerate(captions):
                predictions.append({
                    "caption": caption,
                    "image_name": batch.get("image_name", [])[i] if "image_name" in batch else f"image_{i}",
                    "ground_truth": batch.get("caption", [])[i] if "caption" in batch else "",
                })
        
        return predictions