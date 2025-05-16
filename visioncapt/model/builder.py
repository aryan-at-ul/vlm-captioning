"""Builder functions for models."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Union
from omegaconf import OmegaConf

from .visioncapt_arch import VisionCaptArch

logger = logging.getLogger(__name__)

def build_model(
    config: Union[Dict, str, "OmegaConf"],
    model_path: Optional[str] = None
) -> nn.Module:
    """
    Build full model based on config.
    
    Args:
        config: Model config, path to config file, or OmegaConf object
        model_path: Optional path to pretrained model weights
        
    Returns:
        nn.Module: Full model
    """
    # Load config from file if it's a string
    if isinstance(config, str):
        logger.info(f"Loading config from {config}")
        config = OmegaConf.load(config)
    
    # Convert dict to OmegaConf if needed
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    
    # Determine model type
    model_type = config.get("model_type", "visioncapt")
    
    if model_type.lower() == "visioncapt":
        # Create model
        if model_path is not None:
            logger.info(f"Loading VisionCaptArch from {model_path}")
            model = VisionCaptArch.from_pretrained(model_path)
        else:
            logger.info("Creating new VisionCaptArch")
            model = VisionCaptArch(config)
        
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")