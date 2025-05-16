# Final solution for visioncapt/model/projection/builder.py

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List, Any, Tuple, Union

logger = logging.getLogger(__name__)

class KeywordIgnoringSequential(nn.Sequential):
    """
    A version of nn.Sequential that accepts any format of arguments
    but only uses the tensor data, ignoring all keywords.
    """
    def forward(self, *args, **kwargs):
        """
        Forward pass that works with any argument pattern:
        - First positional arg if available
        - First keyword arg if no positional args
        - Special handling for PEFT's internal argument passing
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Get input from args or kwargs
        if len(args) > 0:
            # Use the first positional argument
            x = args[0]
        elif kwargs:
            # If no positional args, try to find a tensor in kwargs
            # Look for common names for the input tensor
            for key in ['x', 'input', 'inputs', 'features', 'image_features', 'hidden_states']:
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    x = kwargs[key]
                    break
            else:
                # If none of the common names are found, take the first tensor we find
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        x = value
                        break
                else:
                    raise ValueError(
                        f"Cannot find a tensor in kwargs: {list(kwargs.keys())}"
                    )
        else:
            raise ValueError("No arguments provided to KeywordIgnoringSequential")
        
        # Feed forward through all modules
        for module in self:
            x = module(x)
        return x


def build_projection_layer(
    config: Dict,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None
) -> nn.Module:
    """Build projection layer based on config."""
    # Get dimensions
    if input_dim is None:
        input_dim = config.get("input_dim", 768)
    
    if output_dim is None:
        output_dim = config.get("output_dim", 768)
    
    # Get projection type
    proj_type = config.get("type", "mlp").lower()
    
    # Build projection layer
    if proj_type == "linear":
        logger.info(f"Building linear projection: {input_dim} -> {output_dim}")
        return LinearWithFlexibleInputs(input_dim, output_dim)
    
    elif proj_type == "mlp":
        hidden_dim = config.get("hidden_dim", 1024)
        dropout = config.get("dropout", 0.1)
        use_gelu = config.get("use_gelu", True)
        
        logger.info(f"Building MLP projection: {input_dim} -> {hidden_dim} -> {output_dim}")
        
        if use_gelu:
            return KeywordIgnoringSequential(
                LinearWithFlexibleInputs(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                LinearWithFlexibleInputs(hidden_dim, output_dim)
            )
        else:
            return KeywordIgnoringSequential(
                LinearWithFlexibleInputs(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                LinearWithFlexibleInputs(hidden_dim, output_dim)
            )
    
    elif proj_type == "deep_mlp":
        hidden_dims = config.get("hidden_dims", [1024, 1024])
        dropout = config.get("dropout", 0.1)
        use_gelu = config.get("use_gelu", True)
        use_layer_norm = config.get("use_layer_norm", True)
        use_residual = config.get("use_residual", True)
        
        logger.info(f"Building Deep MLP projection: {input_dim} -> {hidden_dims} -> {output_dim}")
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_residual and current_dim == hidden_dim:
                # Residual block
                residual_block = []
                residual_block.append(LinearWithFlexibleInputs(current_dim, hidden_dim))
                if use_gelu:
                    residual_block.append(nn.GELU())
                else:
                    residual_block.append(nn.ReLU())
                residual_block.append(nn.Dropout(dropout))
                
                # Add layer normalization if requested
                if use_layer_norm:
                    residual_block.append(nn.LayerNorm(hidden_dim))
                
                # Create residual connection
                layers.append(ResidualBlockWithFlexibleInputs(KeywordIgnoringSequential(*residual_block)))
            else:
                # Regular layer
                layers.append(LinearWithFlexibleInputs(current_dim, hidden_dim))
                if use_gelu:
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
                # Add layer normalization if requested
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
            
            current_dim = hidden_dim
        
        # Final output layer
        layers.append(LinearWithFlexibleInputs(current_dim, output_dim))
        
        return KeywordIgnoringSequential(*layers)
    
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


class LinearWithFlexibleInputs(nn.Linear):
    """Linear layer that accepts any format of arguments but only uses the tensor data."""
    
    def forward(self, *args, **kwargs):
        """
        Forward with flexible input handling.
        
        Args:
            *args: First arg used if available
            **kwargs: First tensor used if no args
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Extract input tensor using same logic as KeywordIgnoringSequential
        if len(args) > 0:
            return super().forward(args[0])
        elif kwargs:
            for key in ['x', 'input', 'inputs', 'features', 'image_features', 'hidden_states']:
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    return super().forward(kwargs[key])
            
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    return super().forward(value)
            
            raise ValueError(f"Cannot find a tensor in kwargs: {list(kwargs.keys())}")
        else:
            raise ValueError("No arguments provided to LinearWithFlexibleInputs")


class ResidualBlockWithFlexibleInputs(nn.Module):
    """Residual block that accepts any format of arguments."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        """
        Forward with residual connection, flexible input handling.
        
        Args:
            *args: First arg used if available
            **kwargs: First tensor used if no args
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        # Extract input tensor using same logic as KeywordIgnoringSequential
        if len(args) > 0:
            x = args[0]
        elif kwargs:
            for key in ['x', 'input', 'inputs', 'features', 'image_features', 'hidden_states']:
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    x = kwargs[key]
                    break
            else:
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        x = value
                        break
                else:
                    raise ValueError(f"Cannot find a tensor in kwargs: {list(kwargs.keys())}")
        else:
            raise ValueError("No arguments provided to ResidualBlockWithFlexibleInputs")
        
        return x + self.module(x)