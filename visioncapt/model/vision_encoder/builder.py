"""Builder functions for vision encoders."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict

from ...constants import SMALL_VISION_ENCODERS
from .clip_encoder import CLIPVisionEncoder
from .resnet_encoder import ResNetEncoder

logger = logging.getLogger(__name__)

def build_vision_encoder(config: Dict) -> nn.Module:
    """
    Build vision encoder based on config.
    
    Args:
        config: Vision encoder config
        
    Returns:
        nn.Module: Vision encoder module
    """
    encoder_type = config.get("type", "clip").lower()
    
    if encoder_type == "clip":
        # Get model name
        model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        
        # Other parameters
        pretrained = config.get("pretrained", True)
        image_size = config.get("image_size", 224)
        freeze = config.get("freeze", False)
        use_fp16 = config.get("use_fp16", False)
        
        logger.info(f"Building CLIP vision encoder: {model_name}")
        logger.info(f"  Pretrained: {pretrained}")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Freeze: {freeze}")
        logger.info(f"  Use FP16: {use_fp16}")
        
        # Create encoder
        return CLIPVisionEncoder(
            model_name=model_name,
            pretrained=pretrained,
            image_size=image_size,
            freeze=freeze,
            use_fp16=use_fp16
        )
    
    elif encoder_type == "resnet":
        # Get model name
        model_name = config.get("model_name", "resnet50")
        
        # Other parameters
        pretrained = config.get("pretrained", True)
        freeze = config.get("freeze", False)
        pooling_type = config.get("pooling_type", "avg")
        
        logger.info(f"Building ResNet vision encoder: {model_name}")
        logger.info(f"  Pretrained: {pretrained}")
        logger.info(f"  Freeze: {freeze}")
        logger.info(f"  Pooling type: {pooling_type}")
        
        # Create encoder
        return ResNetEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze,
            pooling_type=pooling_type
        )
    
    elif encoder_type == "dino":
        try:
            from .dino_encoder import DinoVisionEncoder
            
            # Get model name
            model_name = config.get("model_name", "facebook/dinov2-small")
            
            # Other parameters
            pretrained = config.get("pretrained", True)
            image_size = config.get("image_size", 224)
            freeze = config.get("freeze", False)
            use_fp16 = config.get("use_fp16", False)
            
            logger.info(f"Building DINOv2 vision encoder: {model_name}")
            logger.info(f"  Pretrained: {pretrained}")
            logger.info(f"  Image size: {image_size}")
            logger.info(f"  Freeze: {freeze}")
            logger.info(f"  Use FP16: {use_fp16}")
            
            # Create encoder
            return DinoVisionEncoder(
                model_name=model_name,
                pretrained=pretrained,
                image_size=image_size,
                freeze=freeze,
                use_fp16=use_fp16
            )
        except ImportError:
            logger.error("DINOv2 encoder requires the timm package. Please install it with: pip install timm")
            raise
    
    elif encoder_type == "vilt":
        try:
            from .vilt_encoder import ViltVisionEncoder
            
            # Get model name
            model_name = config.get("model_name", "dandelin/vilt-b32-mlm")
            
            # Other parameters
            pretrained = config.get("pretrained", True)
            image_size = config.get("image_size", 224)
            freeze = config.get("freeze", False)
            use_fp16 = config.get("use_fp16", False)
            
            logger.info(f"Building ViLT vision encoder: {model_name}")
            logger.info(f"  Pretrained: {pretrained}")
            logger.info(f"  Image size: {image_size}")
            logger.info(f"  Freeze: {freeze}")
            logger.info(f"  Use FP16: {use_fp16}")
            
            # Create encoder
            return ViltVisionEncoder(
                model_name=model_name,
                pretrained=pretrained,
                image_size=image_size,
                freeze=freeze,
                use_fp16=use_fp16
            )
        except ImportError:
            logger.error("ViLT encoder requires the transformers package. Please install it with: pip install transformers")
            raise
    
    elif encoder_type == "vqgan":
        try:
            from .vqgan_encoder import VQGANVisionEncoder
            
            # Get model name
            model_name = config.get("model_name", "stabilityai/sd-vae-ft-mse")
            
            # Other parameters
            pretrained = config.get("pretrained", True)
            image_size = config.get("image_size", 256)
            freeze = config.get("freeze", False)
            use_fp16 = config.get("use_fp16", False)
            latent_size = config.get("latent_size", 16)
            
            logger.info(f"Building VQGAN vision encoder: {model_name}")
            logger.info(f"  Pretrained: {pretrained}")
            logger.info(f"  Image size: {image_size}")
            logger.info(f"  Latent size: {latent_size}")
            logger.info(f"  Freeze: {freeze}")
            logger.info(f"  Use FP16: {use_fp16}")
            
            # Create encoder
            return VQGANVisionEncoder(
                model_name=model_name,
                pretrained=pretrained,
                image_size=image_size,
                latent_size=latent_size,
                freeze=freeze,
                use_fp16=use_fp16
            )
        except ImportError:
            logger.error("VQGAN encoder requires the diffusers package. Please install it with: pip install diffusers")
            raise
    
    else:
        raise ValueError(f"Unknown vision encoder type: {encoder_type}")