# Base language model interface
"""Base language model class for VisionCapt."""

import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class BaseLM(nn.Module, ABC):
    """
    Abstract base class for language models used in VisionCapt.
    
    All language model implementations should inherit from this class.
    """
    
    def __init__(self):
        """Initialize base language model."""
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        projected_visual_features: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Labels for language modeling of shape (batch_size, seq_len)
            projected_visual_features: Projected visual features of shape (batch_size, hidden_size)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing loss, logits, and any other model outputs
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        projected_visual_features: Optional[torch.FloatTensor] = None,
        max_length: int = 77,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text based on input tokens and visual features.
        
        Args:
            input_ids: Optional input token IDs, if not provided will use BOS token
            attention_mask: Optional attention mask
            projected_visual_features: Projected visual features
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_p: Top-p sampling probability
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation kwargs
            
        Returns:
            torch.LongTensor: Generated token IDs
        """
        pass
    
    @abstractmethod
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        projected_visual_features: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for text generation.
        
        Args:
            input_ids: Input token IDs
            projected_visual_features: Projected visual features
            **kwargs: Additional generation kwargs
            
        Returns:
            Dict: Model inputs for generation
        """
        pass
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to the specified directory.
        
        Args:
            save_directory: Directory to save the model
        """
        raise NotImplementedError("Method save_pretrained() not implemented")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "BaseLM":
        """
        Load model from pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            BaseLM: Loaded model
        """
        raise NotImplementedError("Method from_pretrained() not implemented")