"""General utility functions for VisionCapt."""

import os
import random
import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Union, Optional, Dict, Any

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
    """
    # Get numeric logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging_config = {
        "level": numeric_level,
        "format": "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    
    # Add file handler if log file is specified
    if log_file:
        logging_config["filename"] = log_file
        logging_config["filemode"] = "a"
    
    # Apply configuration
    logging.basicConfig(**logging_config)
    
    # Suppress unwanted logs
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> OmegaConf:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf: Configuration
    """
    # Load config from YAML or JSON
    config = OmegaConf.load(config_path)
    
    return config

def save_config(config: OmegaConf, output_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration
        output_path: Path to output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save config
    OmegaConf.save(config, output_path)

def get_optimizer(model, config):
    """
    Get optimizer based on config.
    
    Args:
        model: Model
        config: Optimizer configuration
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    # Get optimizer type
    optimizer_type = config.get("type", "adamw").lower()
    
    # Get optimizer parameters
    learning_rate = config.get("learning_rate", 5e-5)
    weight_decay = config.get("weight_decay", 0.01)
    
    # Create parameter groups with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    if optimizer_type == "adamw":
        # Get AdamW parameters
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        eps = config.get("eps", 1e-8)
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps
        )
    elif optimizer_type == "adam":
        # Get Adam parameters
        beta1 = config.get("beta1", 0.9)
        beta2 = config.get("beta2", 0.999)
        eps = config.get("eps", 1e-8)
        
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps
        )
    elif optimizer_type == "sgd":
        # Get SGD parameters
        momentum = config.get("momentum", 0.9)
        
        return torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def get_scheduler(optimizer, config, num_training_steps: int):
    """
    Get learning rate scheduler based on config.
    
    Args:
        optimizer: Optimizer
        config: Scheduler configuration
        num_training_steps: Number of training steps
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    # Get scheduler type
    scheduler_type = config.get("type", "linear").lower()
    
    # Get scheduler parameters
    warmup_steps = config.get("warmup_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.0)
    
    # Calculate warmup steps if ratio is provided
    if warmup_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_ratio)
    
    # Create scheduler
    if scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        # Get cosine parameters
        min_lr_ratio = config.get("min_lr_ratio", 0.1)
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio
        )
    elif scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get device to use.
    
    Args:
        device: Device string
        
    Returns:
        torch.device: Device to use
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)

def format_dict(d: Dict[str, Any], prefix: str = "") -> str:
    """
    Format a dictionary as a string.
    
    Args:
        d: Dictionary to format
        prefix: Prefix for each line
        
    Returns:
        str: Formatted string
    """
    lines = []
    
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively format nested dictionaries
            lines.append(f"{prefix}{k}:")
            lines.append(format_dict(v, prefix + "  "))
        else:
            lines.append(f"{prefix}{k}: {v}")
    
    return "\n".join(lines)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_parameter_groups(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get the number of parameters in each group of a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dict[str, int]: Number of parameters in each group
    """
    groups = {}
    
    # Count parameters for each module
    for name, module in model.named_modules():
        if name == "":
            continue
        
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if num_params > 0:
            groups[name] = num_params
    
    return groups

def make_model_deterministic(model: torch.nn.Module):
    """
    Make a model deterministic by setting appropriate flags.
    
    Args:
        model: Model to make deterministic
    """
    # Turn off dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

def print_model_info(model: torch.nn.Module):
    """
    Print information about a model.
    
    Args:
        model: Model to print information about
    """
    # Count trainable parameters
    trainable_params = count_parameters(model)
    
    # Count all parameters
    all_params = sum(p.numel() for p in model.parameters())
    
    # Print model information
    print(f"Model summary:")
    print(f"  Total parameters: {all_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Fixed parameters: {all_params - trainable_params:,}")
    
    # Print parameter groups
    print("\nParameter groups:")
    groups = get_parameter_groups(model)
    for name, num_params in sorted(groups.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {num_params:,}")

def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Move a batch of data to a device.
    
    Args:
        batch: Batch of data
        device: Device to move data to
        
    Returns:
        Any: Batch of data on the device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch

def get_free_gpu_memory():
    """
    Get the amount of free GPU memory.
    
    Returns:
        Dict[int, int]: Free memory in bytes for each GPU
    """
    result = {}
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) - torch.cuda.memory_reserved(i)
            result[i] = free_memory
    
    return result

def get_model_size_on_disk(model_path: str) -> int:
    """
    Get the size of a model on disk.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        int: Size in bytes
    """
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    
    return total_size

def format_size(size_bytes: int) -> str:
    """
    Format a size in bytes as a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human-readable size
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"