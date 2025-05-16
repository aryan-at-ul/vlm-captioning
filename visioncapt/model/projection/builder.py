# Projection head

"""Builder functions for projection layers."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

def build_projection_layer(
    config: Dict,
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None
) -> nn.Module:
    """
    Build projection layer based on config.
    
    Args:
        config: Projection layer config
        input_dim: Input dimension (overrides config if provided)
        output_dim: Output dimension (overrides config if provided)
        
    Returns:
        nn.Module: Projection layer module
    """
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
        return nn.Linear(input_dim, output_dim)
    
    elif proj_type == "mlp":
        hidden_dim = config.get("hidden_dim", 1024)
        dropout = config.get("dropout", 0.1)
        use_gelu = config.get("use_gelu", True)
        
        logger.info(f"Building MLP projection: {input_dim} -> {hidden_dim} -> {output_dim}")
        
        if use_gelu:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
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
                residual_block.append(nn.Linear(current_dim, hidden_dim))
                if use_gelu:
                    residual_block.append(nn.GELU())
                else:
                    residual_block.append(nn.ReLU())
                residual_block.append(nn.Dropout(dropout))
                
                # Add layer normalization if requested
                if use_layer_norm:
                    residual_block.append(nn.LayerNorm(hidden_dim))
                
                # Create residual connection
                layers.append(ResidualBlock(nn.Sequential(*residual_block)))
            else:
                # Regular layer
                layers.append(nn.Linear(current_dim, hidden_dim))
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
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")


class ResidualBlock(nn.Module):
    """
    Residual block for projection layers.
    
    Applies the given module and adds a residual connection.
    """
    
    def __init__(self, module: nn.Module):
        """
        Initialize residual block.
        
        Args:
            module: Module to apply
        """
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        return x + self.module(x)