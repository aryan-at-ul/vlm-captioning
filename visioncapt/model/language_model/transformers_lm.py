"""Transformers-based language models for VisionCapt."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)

from .base_lm import BaseLM
from ...constants import SMALL_LANGUAGE_MODELS

logger = logging.getLogger(__name__)

class TransformersLM(BaseLM):
    """
    Language model based on HuggingFace Transformers models.
    
    Supports various causal language models with efficient training options.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        pretrained: bool = True,
        freeze_except_lm_head: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        device_map: str = "auto",
    ):
        """
        Initialize Transformers language model.
        
        Args:
            model_name: Name of the pretrained model or path to a local model
            pretrained: Whether to use pretrained weights
            freeze_except_lm_head: Whether to freeze all parameters except the LM head
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: Configuration for LoRA
            device_map: Device mapping for model parallel execution
        """
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Log initialization
        logger.info(f"Initializing {model_name} language model")
        if load_in_8bit:
            logger.info("Loading model in 8-bit precision")
        elif load_in_4bit:
            logger.info("Loading model in 4-bit precision")
        
        # Set quantization parameters
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else False,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
            )
        
        # Load model and tokenizer
        if pretrained:
            kwargs = {}
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
                kwargs["device_map"] = device_map
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **kwargs
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True if "gemma" not in model_name.lower() else False
            )
        else:
            self.model = AutoModelForCausalLM.from_config(
                AutoModelForCausalLM.config_class.from_pretrained(model_name)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "[PAD]"
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Add image tokens if not present
        special_tokens = {}
        tokens_to_add = ["<image>", "</image>"]
        existing_tokens = self.tokenizer.get_vocab()
        
        for token in tokens_to_add:
            if token not in existing_tokens:
                special_tokens.setdefault("additional_special_tokens", []).append(token)
        
        if special_tokens:
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added {num_added} special tokens to tokenizer")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "n_embd", None) or getattr(self.config, "d_model", None)
        
        # Add visual projection
        self.visual_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Prepare for LoRA if needed
        if use_lora:
            if load_in_8bit or load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Set default LoRA config if not provided
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                }
            
            # Create LoRA config
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 16),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                bias="none",
            )
            
            # Apply LoRA to model
            logger.info(f"Applying LoRA to {model_name} with rank {peft_config.r}")
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Freeze parameters if requested
        elif freeze_except_lm_head and not (load_in_8bit or load_in_4bit):
            logger.info("Freezing all parameters except LM head")
            for name, param in self.model.named_parameters():
                if "lm_head" not in name:
                    param.requires_grad = False
    
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
            Dict containing loss, logits, and other model outputs
        """
        batch_size = input_ids.shape[0] if input_ids is not None else projected_visual_features.shape[0]
        
        # Process visual features if provided
        if projected_visual_features is not None:
            # Project visual features
            visual_emb = self.visual_projection(projected_visual_features)
            
            # Get model embeddings
            if hasattr(self.model, "get_input_embeddings"):
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
            else:
                # Fall back to different model architectures
                if hasattr(self.model, "transformer"):
                    inputs_embeds = self.model.transformer.wte(input_ids)
                elif hasattr(self.model, "model"):
                    inputs_embeds = self.model.model.embed_tokens(input_ids)
                else:
                    raise ValueError(f"Unsupported model architecture: {type(self.model)}")
            
            # Replace the embeddings of <image> token with visual features
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            
            for i in range(batch_size):
                idx = (image_token_positions[0] == i).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    position = image_token_positions[1][idx[0]]
                    inputs_embeds[i, position] = visual_emb[i]
            
            # Forward pass through model
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # Standard forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
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
            input_ids: Optional input token IDs
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
        # Get batch size
        if projected_visual_features is not None:
            batch_size = projected_visual_features.shape[0]
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = 1
        
        # If input_ids not provided, use BOS token
        if input_ids is None:
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                bos_token_id = self.tokenizer.bos_token_id
            else:
                bos_token_id = self.tokenizer.encode("<|startoftext|>")[0]
                
            # Create input IDs with BOS token and image token
            input_ids = torch.full(
                (batch_size, 2),
                bos_token_id,
                dtype=torch.long,
                device=projected_visual_features.device if projected_visual_features is not None else "cuda"
            )
            input_ids[:, 1] = self.tokenizer.convert_tokens_to_ids("<image>")
        
        # Get model embeddings for input tokens
        if hasattr(self.model, "get_input_embeddings"):
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        else:
            # Fall back to different model architectures
            if hasattr(self.model, "transformer"):
                inputs_embeds = self.model.transformer.wte(input_ids)
            elif hasattr(self.model, "model"):
                inputs_embeds = self.model.model.embed_tokens(input_ids)
            else:
                raise ValueError(f"Unsupported model architecture: {type(self.model)}")
        
        # Replace the embeddings of <image> token with visual features if provided
        if projected_visual_features is not None:
            # Project visual features
            visual_emb = self.visual_projection(projected_visual_features)
            
            # Find image token positions
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            
            for i in range(batch_size):
                idx = (image_token_positions[0] == i).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    position = image_token_positions[1][idx[0]]
                    inputs_embeds[i, position] = visual_emb[i]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Prepare model inputs for generation
        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Update with additional kwargs
        gen_kwargs.update(kwargs)
        
        # Generate text
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**model_kwargs, **gen_kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            # Fall back to regular generation without inputs_embeds
            gen_kwargs["input_ids"] = input_ids
            with torch.no_grad():
                generated_ids = self.model.generate(**gen_kwargs)
        
        # Decode generated IDs to text
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
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
        model_inputs = {
            "input_ids": input_ids,
        }
        
        # Add attention mask if provided
        if "attention_mask" in kwargs:
            model_inputs["attention_mask"] = kwargs["attention_mask"]
        
        # Process visual features if provided
        if projected_visual_features is not None:
            # Get batch size
            batch_size = input_ids.shape[0]
            
            # Project visual features
            visual_emb = self.visual_projection(projected_visual_features)
            
            # Get model embeddings
            if hasattr(self.model, "get_input_embeddings"):
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
            else:
                # Fall back to different model architectures
                if hasattr(self.model, "transformer"):
                    inputs_embeds = self.model.transformer.wte(input_ids)
                elif hasattr(self.model, "model"):
                    inputs_embeds = self.model.model.embed_tokens(input_ids)
                else:
                    raise ValueError(f"Unsupported model architecture: {type(self.model)}")
            
            # Replace the embeddings of <image> token with visual features
            image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
            
            for i in range(batch_size):
                idx = (image_token_positions[0] == i).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    position = image_token_positions[1][idx[0]]
                    inputs_embeds[i, position] = visual_emb[i]
            
            # Replace input_ids with inputs_embeds
            model_inputs.pop("input_ids")
            model_inputs["inputs_embeds"] = inputs_embeds
        
        return model_inputs
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to the specified directory.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model
        if self.use_lora:
            self.model.save_pretrained(save_directory)
        else:
            self.model.save_pretrained(os.path.join(save_directory, "model"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        
        # Save visual projection
        torch.save(self.visual_projection.state_dict(), os.path.join(save_directory, "visual_projection.pt"))
        
        # Save config
        with open(os.path.join(save_directory, "model_config.json"), "w") as f:
            import json
            json.dump({
                "model_name": self.model_name,
                "hidden_size": self.hidden_size,
                "use_lora": self.use_lora,
            }, f)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "TransformersLM":
        """
        Load model from pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional arguments
            
        Returns:
            TransformersLM: Loaded model
        """
        # Load config
        model_config_path = os.path.join(pretrained_model_name_or_path, "model_config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                import json
                config = json.load(f)
                
                # Get model name
                model_name = config.get("model_name", pretrained_model_name_or_path)
                use_lora = config.get("use_lora", False)
                
                # Create model
                model = cls(
                    model_name=model_name,
                    pretrained=True,
                    use_lora=use_lora,
                    **kwargs
                )
                
                # Load visual projection
                visual_projection_path = os.path.join(pretrained_model_name_or_path, "visual_projection.pt")
                if os.path.exists(visual_projection_path):
                    model.visual_projection.load_state_dict(torch.load(visual_projection_path))
                
                return model
        
        # If no config file, try loading as a regular Transformers model
        return cls(model_name=pretrained_model_name_or_path, **kwargs)