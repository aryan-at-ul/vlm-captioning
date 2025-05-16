# Text preprocessing functions

"""Text utility functions for VisionCapt."""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any

def clean_caption(caption: str) -> str:
    """
    Clean a caption string.
    
    Args:
        caption: Caption to clean
        
    Returns:
        str: Cleaned caption
    """
    # Remove multiple spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove leading/trailing spaces
    caption = caption.strip()
    
    # Remove special characters and digits
    # caption = re.sub(r'[^\w\s]', '', caption)
    # caption = re.sub(r'\d+', '', caption)
    
    return caption

def tokenize_caption(
    caption: str,
    tokenizer: Any,
    max_length: int = 77,
    add_special_tokens: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a caption string.
    
    Args:
        caption: Caption to tokenize
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        return_tensors: Return format
        
    Returns:
        Dict[str, torch.Tensor]: Tokenized caption
    """
    # Clean caption
    caption = clean_caption(caption)
    
    # Tokenize
    tokenized = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors=return_tensors
    )
    
    return tokenized

def decode_caption(
    token_ids: torch.Tensor,
    tokenizer: Any,
    skip_special_tokens: bool = True
) -> str:
    """
    Decode token IDs to caption string.
    
    Args:
        token_ids: Token IDs
        tokenizer: Tokenizer
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        str: Decoded caption
    """
    # Remove batch dimension if present
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)
    
    # Decode
    caption = tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens
    )
    
    # Clean caption
    caption = clean_caption(caption)
    
    return caption

def prepare_caption_for_training(
    caption: str,
    tokenizer: Any,
    max_length: int = 77,
    template: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare a caption for training.
    
    Args:
        caption: Caption to prepare
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        template: Optional template string with {} where caption should be inserted
        
    Returns:
        Dict[str, torch.Tensor]: Tokenized caption with labels
    """
    # Clean caption
    caption = clean_caption(caption)
    
    # Apply template if provided
    if template:
        caption = template.format(caption)
    
    # Tokenize
    tokenized = tokenize_caption(
        caption,
        tokenizer,
        max_length=max_length,
        add_special_tokens=True
    )
    
    # Create labels for causal LM (shifted input_ids)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    labels = input_ids.clone()
    
    # Set padding tokens to -100 (ignored in loss calculation)
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Handle image token if present
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        
        # Find image token position
        image_token_pos = (labels == image_token_id).nonzero(as_tuple=True)
        
        if image_token_pos[0].size(0) > 0 and image_token_pos[1].size(0) > 0:
            # Set tokens before and including image token to -100
            for i in range(input_ids.size(0)):
                idx = (image_token_pos[0] == i).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    pos = image_token_pos[1][idx[0]]
                    labels[i, :pos+1] = -100  # Ignore BOS and image token
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def add_image_tokens_to_caption(
    caption: str,
    tokenizer: Any,
    max_length: int = 77,
    image_token: str = "<image>"
) -> Dict[str, torch.Tensor]:
    """
    Add image tokens to caption for inference.
    
    Args:
        caption: Caption string
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        image_token: Image token string
        
    Returns:
        Dict[str, torch.Tensor]: Tokenized caption with image token
    """
    # Add image token to caption
    caption_with_token = f"{image_token} {caption}"
    
    # Tokenize
    tokenized = tokenize_caption(
        caption_with_token,
        tokenizer,
        max_length=max_length,
        add_special_tokens=True
    )
    
    return tokenized

def caption_to_prompt(
    caption: str,
    prompt_template: str = "Generate a caption for this image: {}"
) -> str:
    """
    Convert a caption to a prompt.
    
    Args:
        caption: Caption string
        prompt_template: Prompt template string with {} where caption should be inserted
        
    Returns:
        str: Prompt string
    """
    return prompt_template.format(caption)

def calculate_caption_metrics(
    generated_captions: List[str],
    reference_captions: List[str]
) -> Dict[str, float]:
    """
    Calculate metrics for generated captions.
    
    Args:
        generated_captions: List of generated captions
        reference_captions: List of reference captions
        
    Returns:
        Dict[str, float]: Metrics
    """
    metrics = {}
    
    # Try to import NLG evaluation libraries
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        # Tokenize captions
        tokenized_refs = [[ref.split()] for ref in reference_captions]
        tokenized_hyps = [hyp.split() for hyp in generated_captions]
        
        # Calculate BLEU-1
        weights = (1.0, 0.0, 0.0, 0.0)
        metrics["bleu1"] = corpus_bleu(
            tokenized_refs,
            tokenized_hyps,
            weights=weights,
            smoothing_function=SmoothingFunction().method1
        )
        
        # Calculate BLEU-4
        weights = (0.25, 0.25, 0.25, 0.25)
        metrics["bleu4"] = corpus_bleu(
            tokenized_refs,
            tokenized_hyps,
            weights=weights,
            smoothing_function=SmoothingFunction().method1
        )
    except ImportError:
        print("NLTK not installed. Will not compute BLEU scores.")
    
    # Try to import ROUGE
    try:
        from rouge import Rouge
        
        rouge = Rouge()
        rouge_scores = rouge.get_scores(
            generated_captions,
            reference_captions,
            avg=True
        )
        
        metrics["rouge_l"] = rouge_scores["rouge-l"]["f"]
    except ImportError:
        print("Rouge not installed. Will not compute ROUGE scores.")
    
    # Try to import METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        
        meteor_scores = []
        for ref, hyp in zip(reference_captions, generated_captions):
            score = meteor_score([ref.split()], hyp.split())
            meteor_scores.append(score)
        
        metrics["meteor"] = np.mean(meteor_scores)
    except ImportError:
        print("NLTK not installed. Will not compute METEOR score.")
    
    # Try to import CIDEr
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for CIDEr calculation
        refs = {}
        hyps = {}
        
        for i, (ref, hyp) in enumerate(zip(reference_captions, generated_captions)):
            refs[i] = [ref]
            hyps[i] = hyp
        
        cider = Cider()
        cider_score, _ = cider.compute_score(refs, hyps)
        
        metrics["cider"] = cider_score
    except ImportError:
        print("pycocoevalcap not installed. Will not compute CIDEr score.")
    
    return metrics

def extract_clip_tokens(
    caption: str,
    start_token: str = "<image>",
    end_token: str = "</image>"
) -> Tuple[str, str, str]:
    """
    Extract the text before, inside, and after CLIP tokens.
    
    Args:
        caption: Caption string
        start_token: Start token for CLIP
        end_token: End token for CLIP
        
    Returns:
        Tuple[str, str, str]: Text before CLIP tokens, text inside CLIP tokens, text after CLIP tokens
    """
    # Find the positions of the CLIP tokens
    start_pos = caption.find(start_token)
    end_pos = caption.find(end_token)
    
    # Extract the text before, inside, and after the CLIP tokens
    before_clip = caption[:start_pos].strip() if start_pos != -1 else caption
    
    if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
        inside_clip = caption[start_pos + len(start_token):end_pos].strip()
        after_clip = caption[end_pos + len(end_token):].strip()
    else:
        inside_clip = ""
        after_clip = caption[start_pos + len(start_token):].strip() if start_pos != -1 else ""
    
    return before_clip, inside_clip, after_clip