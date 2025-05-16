# # CLIP encoder

# """CLIP vision encoder implementation"""

# import torch
# import torch.nn as nn
# from transformers import CLIPVisionModel, CLIPImageProcessor
# import logging

# logger = logging.getLogger(__name__)

# class CLIPVisionEncoder(nn.Module):
#     """
#     Vision encoder based on CLIP ViT model
    
#     Extracts visual features from images using CLIP's vision transformer
#     """
    
#     def __init__(
#         self,
#         model_name="openai/clip-vit-base-patch32",
#         pretrained=True,
#         image_size=224,
#         freeze=False,
#     ):
#         """
#         Initialize CLIP vision encoder
        
#         Args:
#             model_name: Name of the pretrained CLIP model
#             pretrained: Whether to use pretrained weights
#             image_size: Size of input images
#             freeze: Whether to freeze the encoder parameters
#         """
#         super().__init__()
#         self.model_name = model_name
#         self.image_size = image_size
        
#         # Load CLIP vision model
#         logger.info(f"Loading CLIP vision model: {model_name}")
#         self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
#         # Load image processor
#         self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
#         # Get output dimension
#         self.hidden_size = self.vision_model.config.hidden_size
        
#         # Freeze parameters if specified
#         if freeze:
#             logger.info("Freezing CLIP vision model parameters")
#             for param in self.vision_model.parameters():
#                 param.requires_grad = False
    
#     def forward(self, pixel_values=None, images=None):
#         """
#         Forward pass
        
#         Args:
#             pixel_values: Preprocessed image tensor of shape (batch_size, channels, height, width)
#             images: Raw image tensor of shape (batch_size, channels, height, width)
#                    Will be preprocessed if pixel_values is None
                   
#         Returns:
#             torch.Tensor: Image features of shape (batch_size, hidden_size)
#         """
#         # Preprocess images if pixel_values is None
#         if pixel_values is None and images is not None:
#             if images.shape[-2:] != (self.image_size, self.image_size):
#                 # Resize images if necessary
#                 images = torch.nn.functional.interpolate(
#                     images, 
#                     size=(self.image_size, self.image_size), 
#                     mode="bilinear", 
#                     align_corners=False
#                 )
            
#             # Normalize images
#             if images.min() >= 0 and images.max() <= 1:
#                 # Images are already normalized to [0, 1]
#                 pixel_values = self.image_processor.normalize(images)
#             else:
#                 # Assume images are in range [0, 255]
#                 pixel_values = images / 255.0
#                 pixel_values = self.image_processor.normalize(pixel_values)
        
#         # Forward pass through vision model
#         outputs = self.vision_model(pixel_values=pixel_values)
        
#         # Return pooled features
#         image_features = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        
#         return image_features
    
#     def save_pretrained(self, output_dir):
#         """
#         Save model to output_dir
        
#         Args:
#             output_dir: Path to output directory
#         """
#         import os
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save vision model
#         self.vision_model.save_pretrained(output_dir)
        
#         # Save image processor
#         self.image_processor.save_pretrained(output_dir)
    
#     @classmethod
#     def from_pretrained(cls, model_path, **kwargs):
#         """
#         Load model from model_path
        
#         Args:
#             model_path: Path to model directory
#             **kwargs: Additional arguments to pass to __init__
            
#         Returns:
#             CLIPVisionEncoder: Loaded model
#         """
#         # Check if model_path is a directory or a model name
#         import os
#         if os.path.isdir(model_path):
#             # Load from local directory
#             model = cls(model_name=model_path, **kwargs)
#             model.vision_model = CLIPVisionModel.from_pretrained(model_path)
#             model.image_processor = CLIPImageProcessor.from_pretrained(model_path)
#         else:
#             # Load from model name
#             model = cls(model_name=model_path, **kwargs)
        
#         return model

# visioncapt/model/vision_encoder/clip_encoder.py

"""CLIP vision encoder implementation"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
import logging

logger = logging.getLogger(__name__)

class CLIPVisionEncoder(nn.Module):
    """
    Vision encoder based on CLIP ViT model.

    Extracts visual features from images using CLIP's vision transformer.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        pretrained: bool = True,
        image_size: int = 224,
        freeze: bool = False,
        use_fp16: bool = False,
    ):
        """
        Initialize CLIP vision encoder.
        
        Args:
            model_name: Name or path of the pretrained CLIP model
            pretrained: Whether to load pretrained weights
            image_size: Size to resize input images to
            freeze: Whether to freeze encoder parameters
            use_fp16: Whether to cast the model to half precision
        """
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        
        # Load CLIP vision model
        logger.info(f"Loading CLIP vision model: {model_name} (pretrained={pretrained})")
        self.vision_model = CLIPVisionModel.from_pretrained(
            model_name, 
            revision=None if pretrained else "main",
            torch_dtype=torch.float16 if (use_fp16 and pretrained) else None,
        )
        
        # Optionally cast to fp16 after loading
        if use_fp16 and pretrained:
            logger.info("Casting CLIP vision model to float16")
            self.vision_model = self.vision_model.half()
        
        # Load image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Output dimension
        self.hidden_size = self.vision_model.config.hidden_size
        
        # Freeze parameters if requested
        if freeze:
            logger.info("Freezing CLIP vision model parameters")
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor = None, images: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values: Preprocessed image tensor 
                of shape (batch, channels, height, width)
            images: Raw image tensor (0–1 or 0–255) of shape 
                (batch, channels, height, width); will be preprocessed if pixel_values is None
        
        Returns:
            image_features: Tensor of shape (batch, hidden_size)
        """
        # If user passed raw images, preprocess them
        if pixel_values is None:
            if images is None:
                raise ValueError("Either pixel_values or images must be provided")
            # Resize if needed
            if images.shape[-2:] != (self.image_size, self.image_size):
                images = F.interpolate(
                    images, size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False
                )
            # Normalize to [0,1] if coming in 0–255
            if images.max() > 1.0:
                images = images / 255.0
            pixel_values = self.image_processor(images, return_tensors="pt").pixel_values.to(images.device)

        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output  # (batch, hidden_size)

    def save_pretrained(self, output_dir: str):
        """
        Save model and processor to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.vision_model.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "CLIPVisionEncoder":
        """
        Load model from local directory or HF repo name.
        
        kwargs are forwarded to __init__ (e.g. use_fp16, freeze).
        """
        # Instantiate (this will load weights)
        encoder = cls(model_name=model_path, **kwargs)
        # If model_path is a directory, reload from there
        if os.path.isdir(model_path):
            encoder.vision_model = CLIPVisionModel.from_pretrained(model_path)
            encoder.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        return encoder
