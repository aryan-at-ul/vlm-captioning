"""VisionCapt model architecture definition"""

import os
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from peft import PeftModel, LoraConfig, get_peft_model

from .vision_encoder.builder import build_vision_encoder
from .language_model.builder import build_language_model
from .projection.builder import build_projection_layer
from .projection_wrapper import ProjectionWrapper, apply_lora_to_projection  # Add this import

logger = logging.getLogger(__name__)

class VisionCaptArch(nn.Module):
    """
    Vision-Language Model for image captioning
    
    Architecture:
    1. Vision Encoder: Extracts features from images
    2. Projection Layer: Projects vision features to language model embedding space
    3. Language Model: Generates text based on projected image features
    """
    
    def __init__(self, config):
        """
        Initialize VisionCaptArch.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Build vision encoder
        logger.info("Building vision encoder...")
        self.vision_encoder = build_vision_encoder(config.model.vision_encoder)
        
        # Get vision encoder output dimension
        vision_dim = getattr(self.vision_encoder, "hidden_size", 768)
        
        # Build language model
        logger.info("Building language model...")
        self.language_model = build_language_model(config.model.language_model)
        
        # Get language model embedding dimension
        language_dim = getattr(self.language_model, "hidden_size", 768)
        
        # Update projection config with dimensions if not set
        if not hasattr(config.model.projection, "input_dim"):
            config.model.projection.input_dim = vision_dim
        
        if not hasattr(config.model.projection, "output_dim"):
            config.model.projection.output_dim = language_dim
        
        # Build projection layer
        logger.info("Building projection layer...")
        self.projection = build_projection_layer(
            config.model.projection,
            input_dim=vision_dim,
            output_dim=language_dim
        )
        
        # Initialize weights
        self._init_weights()
        
        # Apply LoRA if enabled
        if hasattr(config, "lora") and config.lora.get("enabled", False):
            # Only apply LoRA to specified modules
            if "vision_encoder" in config.lora.get("apply_to", []):
                logger.info("Applying LoRA to vision encoder...")
                peft_config = LoraConfig(
                    task_type="FEATURE_EXTRACTION",
                    r=config.lora.get("r", 8),
                    lora_alpha=config.lora.get("alpha", 16),
                    lora_dropout=config.lora.get("dropout", 0.05),
                    target_modules=config.lora.get("vision_target_modules", ["q_proj", "v_proj"]),
                )
                self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
            
            # The language model might already have LoRA applied in its constructor
            
            # Apply LoRA to projection layer if specified
            if "projection" in config.lora.get("apply_to", []):
                logger.info("Applying LoRA to projection layer with custom wrapper...")
                # Use our custom wrapper to safely apply LoRA to projection
                self.projection = apply_lora_to_projection(self, config)
        
        logger.info("VisionCaptArch initialized!")
    
    def _init_weights(self):
        """Initialize weights of the projection layer."""
        # Attempt to get the base projection if it's wrapped
        projection = getattr(self.projection, "projection", self.projection)
        
        if isinstance(projection, nn.Sequential):
            for module in projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(projection, nn.Linear):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        elif hasattr(projection, "apply"):
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            projection.apply(_init_weights)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Labels for language modeling of shape (batch_size, seq_len)
            pixel_values: Preprocessed image tensor (alternative to images)
            return_dict: Whether to return a dictionary or just the loss
            **kwargs: Additional kwargs for the language model
            
        Returns:
            Dict containing loss, logits, and other model outputs
        """
        # Process images
        if images is None and pixel_values is not None:
            images = pixel_values
        
        # Extract image features
        image_features = self.vision_encoder(images)
        
        # Project image features to language model embedding space
        # CRITICAL: Only pass the image features - nothing else
        try:
            # Call with positional arg only - no keywords
            projected_features = self.projection(image_features)
        except Exception as e:
            logger.error(f"Error projecting features: {e}")
            logger.error(f"Image features shape: {image_features.shape}")
            logger.error(f"Projection type: {type(self.projection)}")
            raise
        
        # Forward pass through language model with remaining arguments
        lm_kwargs = {k: v for k, v in kwargs.items()}
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            projected_visual_features=projected_features,
            **lm_kwargs
        )
        
        if return_dict:
            # Add projected features to outputs
            if not isinstance(outputs, dict):
                outputs = {
                    "loss": outputs[0] if len(outputs) > 0 else None,
                    "logits": outputs[1] if len(outputs) > 1 else None,
                }
            
            outputs["projected_features"] = projected_features
            outputs["image_features"] = image_features
            
            return outputs
        else:
            return outputs
    
    def generate_captions(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        max_length: int = 77,
        min_length: int = 5,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate captions for images.
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
            input_ids: Optional input token IDs to start generation
            attention_mask: Optional attention mask
            pixel_values: Preprocessed image tensor (alternative to images)
            max_length: Maximum length of generated captions
            min_length: Minimum length of generated captions
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_p: Top-p sampling probability
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation kwargs
            
        Returns:
            list: List of generated captions
        """
        # Process images
        if images is None and pixel_values is not None:
            images = pixel_values
        
        # Extract image features
        with torch.no_grad():
            image_features = self.vision_encoder(images)
            
            # Project image features - ONLY pass image_features directly
            # No keyword arguments, just positional
            projected_features = self.projection(image_features)
            
            # Create generation kwargs dictionary
            generation_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            generation_kwargs.update(kwargs)
            
            # Generate captions
            captions = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                projected_visual_features=projected_features,
                **generation_kwargs
            )
        
        return captions


    def save_pretrained(self, output_dir: str, save_lora_only: bool = False) -> None:
        """
        Save the model (and optionally only its LoRA adapters) to `output_dir`,
        along with a JSON/YAML dump of the fully resolved config.

        Args:
            output_dir: Path to output directory
            save_lora_only: Whether to save only LoRA weights
        """
        import os, json, yaml, torch, logging
        from omegaconf import OmegaConf
        from peft import PeftModel

        logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

        # Determine modules to save
        modules = []
        if save_lora_only:
            # Only save LoRA adapters if present
            for name, mod in [
                ("vision_encoder", self.vision_encoder),
                ("language_model", self.language_model),
                ("projection", self.projection),
            ]:
                if isinstance(mod, PeftModel) or (hasattr(mod, 'use_lora') and getattr(mod, 'use_lora', False)):
                    modules.append((name + '_lora', mod))
            if not modules:
                logger.warning("save_lora_only=True but no LoRA modules found; falling back to full save.")
                save_lora_only = False

        if not save_lora_only:
            modules = [
                ("vision_encoder", self.vision_encoder),
                ("language_model", self.language_model),
                ("projection", self.projection),
            ]

        # Save each selected module
        for subdir_name, module in modules:
            subdir = os.path.join(output_dir, subdir_name)
            os.makedirs(subdir, exist_ok=True)
            if hasattr(module, 'save_pretrained'):
                module.save_pretrained(subdir)
            else:
                torch.save(module.state_dict(), os.path.join(subdir, 'model.pt'))

        # Serialize config: convert OmegaConf → pure Python
        cfg = self.config
        if OmegaConf.is_config(cfg):
            config_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        elif isinstance(cfg, dict):
            config_dict = dict(cfg)
        else:
            config_dict = cfg  # assume already serializable

        # Write JSON
        json_path = os.path.join(output_dir, 'config.json')
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Write YAML
        yaml_path = os.path.join(output_dir, 'config.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(f"Saved model and config to {output_dir}")

    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "VisionCaptArch":
        """
        Load model from model_path.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional kwargs for model initialization
            
        Returns:
            VisionCaptArch: Loaded model
        """
        import json
        import os
        from omegaconf import OmegaConf
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_path, "config.yaml")
        
        if os.path.exists(config_path):
            # Load config
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                config = OmegaConf.load(config_path)
            
            # Convert to OmegaConf if it's a dict
            if isinstance(config, dict):
                config = OmegaConf.create(config)
            
            # Update config with kwargs
            for key, value in kwargs.items():
                OmegaConf.update(config, key, value)
        else:
            raise ValueError(f"Config file not found at {config_path}")
        
        # Create model
        model = cls(config)
        
        # Check for LoRA weights
        vision_encoder_lora_path = os.path.join(model_path, "vision_encoder_lora")
        language_model_lora_path = os.path.join(model_path, "language_model_lora")
        projection_lora_path = os.path.join(model_path, "projection_lora")
        
        lora_only = (
            os.path.exists(vision_encoder_lora_path) or 
            os.path.exists(language_model_lora_path) or 
            os.path.exists(projection_lora_path)
        )
        
        if lora_only:
            # Load base model first
            logger.info("Loading base model with LoRA weights...")
            
            # Load LoRA weights for vision encoder if exists
            if os.path.exists(vision_encoder_lora_path):
                from peft import PeftModel
                logger.info(f"Loading LoRA weights for vision encoder from {vision_encoder_lora_path}")
                model.vision_encoder = PeftModel.from_pretrained(model.vision_encoder, vision_encoder_lora_path)
            
            # Load LoRA weights for language model if exists
            if os.path.exists(language_model_lora_path) and hasattr(model.language_model, "model"):
                from peft import PeftModel
                logger.info(f"Loading LoRA weights for language model from {language_model_lora_path}")
                base_model = model.language_model.model
                model.language_model.model = PeftModel.from_pretrained(base_model, language_model_lora_path)
            
            # Load LoRA weights for projection layer if exists
            if os.path.exists(projection_lora_path):
                from peft import PeftModel
                logger.info(f"Loading LoRA weights for projection layer from {projection_lora_path}")
                model.projection = PeftModel.from_pretrained(model.projection, projection_lora_path)
        else:
            # Load full model weights
            vision_encoder_path = os.path.join(model_path, "vision_encoder")
            language_model_path = os.path.join(model_path, "language_model")
            projection_path = os.path.join(model_path, "projection")
            
            # Load vision encoder weights if exists
            if os.path.exists(vision_encoder_path):
                if hasattr(model.vision_encoder, "from_pretrained"):
                    logger.info(f"Loading vision encoder from {vision_encoder_path}")
                    model.vision_encoder = model.vision_encoder.__class__.from_pretrained(vision_encoder_path)
                else:
                    model_pt_path = os.path.join(vision_encoder_path, "model.pt")
                    if os.path.exists(model_pt_path):
                        logger.info(f"Loading vision encoder weights from {model_pt_path}")
                        model.vision_encoder.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
            
            # Load language model weights if exists
            if os.path.exists(language_model_path):
                if hasattr(model.language_model, "from_pretrained"):
                    logger.info(f"Loading language model from {language_model_path}")
                    model.language_model = model.language_model.__class__.from_pretrained(language_model_path)
                else:
                    model_pt_path = os.path.join(language_model_path, "model.pt")
                    if os.path.exists(model_pt_path):
                        logger.info(f"Loading language model weights from {model_pt_path}")
                        model.language_model.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
            
            # Load projection weights if exists
            if os.path.exists(projection_path):
                if hasattr(model.projection, "from_pretrained"):
                    logger.info(f"Loading projection layer from {projection_path}")
                    model.projection = model.projection.__class__.from_pretrained(projection_path)
                else:
                    model_pt_path = os.path.join(projection_path, "model.pt")
                    if os.path.exists(model_pt_path):
                        logger.info(f"Loading projection weights from {model_pt_path}")
                        model.projection.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
        
        return model