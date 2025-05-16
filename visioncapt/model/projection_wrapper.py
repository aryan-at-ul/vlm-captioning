# visioncapt/model/projection_wrapper.py

import torch
import torch.nn as nn
import logging
from typing import Any, Tuple, List, Dict, Optional, Union
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)

class ProjectionWrapper(nn.Module):
    """
    A wrapper around the projection layer that properly handles LoRA.
    This wrapper ensures that only image_features are passed to the projection layer.
    """
    def __init__(self, projection_layer: nn.Module):
        super().__init__()
        self.projection = projection_layer
        self.is_lora_wrapped = True

    def forward(self, *args, **kwargs):
        """
        Forward pass with robust argument handling:
        - Use first positional arg if available
        - Extract tensor from kwargs if no positional args
        - Log detailed debug info if any issues occur
        
        Args:
            *args: Positional arguments (first one should be image features)
            **kwargs: Keyword arguments
            
        Returns:
            torch.Tensor: Projected features
        """
        try:
            # Debug info
            logger.debug(f"ProjectionWrapper received args types: {[type(a) for a in args]}")
            logger.debug(f"ProjectionWrapper received kwargs keys: {list(kwargs.keys())}")
            
            # Extract input tensor from args or kwargs
            if len(args) > 0:
                # Use first positional arg
                x = args[0]
                logger.debug(f"Using first positional arg with shape: {x.shape}")
            elif kwargs:
                # Try common names for the input tensor
                for key in ['x', 'input', 'inputs', 'features', 'image_features', 'hidden_states']:
                    if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                        x = kwargs[key]
                        logger.debug(f"Using {key} from kwargs with shape: {x.shape}")
                        break
                else:
                    # If no common names found, use first tensor found
                    for key, value in kwargs.items():
                        if isinstance(value, torch.Tensor):
                            x = value
                            logger.debug(f"Using {key} from kwargs with shape: {x.shape}")
                            break
                    else:
                        raise ValueError(f"Cannot find a tensor in kwargs: {list(kwargs.keys())}")
            else:
                raise ValueError("No arguments provided to ProjectionWrapper")
            
            # Call projection with only the tensor
            return self.projection(x)
            
        except Exception as e:
            logger.error(f"Error in ProjectionWrapper.forward: {e}")
            logger.error(f"Args: {args}")
            logger.error(f"Kwargs keys: {list(kwargs.keys())}")
            logger.error(f"Projection type: {type(self.projection)}")
            raise

    def __getattr__(self, name: str) -> Any:
        """Forward attribute lookups to wrapped projection."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.projection, name)


def apply_lora_to_projection(model, config) -> nn.Module:
    """
    Apply LoRA to projection layer with a safe wrapper for argument handling.
    
    Args:
        model: The VisionCaptArch instance
        config: The full configuration
        
    Returns:
        nn.Module: The LoRA-enhanced projection wrapped in ProjectionWrapper
    """
    logger.info("Safely applying LoRA to projection layer...")
    projection = model.projection
    target_modules = []
    
    # Detect target modules based on projection type
    if isinstance(projection, nn.Sequential):
        for idx, module in enumerate(projection):
            if isinstance(module, nn.Linear):
                target_modules.append(str(idx))
        logger.info(f"Detected Linear layers in Sequential at indices: {target_modules}")
    
    elif isinstance(projection, nn.Linear):
        target_modules = ['']
        logger.info("Detected single Linear layer for LoRA")
    
    else:
        # Try to find named Linear modules
        for name, module in projection.named_modules():
            if isinstance(module, nn.Linear) and not name.startswith('module.'):
                clean_name = name.lstrip('.')
                if clean_name:
                    target_modules.append(clean_name)
        logger.info(f"Found Linear layers named: {target_modules}")
    
    # Use configured target modules as fallback
    if not target_modules and config.lora.get("projection_target_modules"):
        target_modules = config.lora.get("projection_target_modules")
        logger.info(f"Using configured projection_target_modules: {target_modules}")
    
    if not target_modules:
        logger.warning("No target modules found for projection layer. Skipping LoRA.")
        return projection
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=config.lora.get("r", 16),
        lora_alpha=config.lora.get("alpha", 32),
        lora_dropout=config.lora.get("dropout", 0.05),
        target_modules=target_modules,
    )
    
    # Apply LoRA
    try:
        logger.info(f"Applying LoRA to projection with target_modules: {target_modules}")
        lora_proj = get_peft_model(projection, peft_config)
        wrapped_proj = ProjectionWrapper(lora_proj)
        return wrapped_proj
    
    except Exception as e:
        logger.error(f"Failed to apply LoRA to projection: {e}")
        logger.warning("Falling back to regular projection")
        return projection