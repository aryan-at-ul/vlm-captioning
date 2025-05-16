# ResNet encoder

"""ResNet vision encoder implementation"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
import logging

logger = logging.getLogger(__name__)

class ResNetEncoder(nn.Module):
    """
    Vision encoder based on ResNet model
    
    Extracts visual features from images using ResNet
    """
    
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        freeze=False,
    ):
        """
        Initialize ResNet vision encoder
        
        Args:
            model_name: Name of the ResNet model (resnet50, resnet101, resnet152)
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze the encoder parameters
        """
        super().__init__()
        self.model_name = model_name
        
        # Load ResNet model
        logger.info(f"Loading ResNet model: {model_name}")
        
        if model_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            self.hidden_size = 2048  # ResNet50 final feature dimension
        elif model_name == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet101(weights=weights)
            self.hidden_size = 2048  # ResNet101 final feature dimension
        elif model_name == "resnet152":
            weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet152(weights=weights)
            self.hidden_size = 2048  # ResNet152 final feature dimension
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove classification head
        self.model = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze parameters if specified
        if freeze:
            logger.info("Freezing ResNet model parameters")
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Normalization parameters for preprocessing
        self.register_buffer(
            "mean", 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def preprocess(self, images):
        """
        Preprocess images for ResNet
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Preprocessed images
        """
        # Resize to 224x224 if necessary
        if images.shape[-2:] != (224, 224):
            images = torch.nn.functional.interpolate(
                images, 
                size=(224, 224), 
                mode="bilinear", 
                align_corners=False
            )
        
        # Normalize
        if images.min() >= 0 and images.max() <= 1:
            # Images are already in range [0, 1]
            normalized = (images - self.mean) / self.std
        else:
            # Assume images are in range [0, 255]
            normalized = ((images / 255.0) - self.mean) / self.std
            
        return normalized
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Image features of shape (batch_size, hidden_size)
        """
        # Preprocess images
        processed_images = self.preprocess(images)
        
        # Forward pass through ResNet
        features = self.model(processed_images)
        
        # Flatten features
        batch_size = features.shape[0]
        return features.view(batch_size, self.hidden_size)
    
    def save_pretrained(self, output_dir):
        """
        Save model to output_dir
        
        Args:
            output_dir: Path to output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save model config
        import json
        config = {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load model from model_path
        
        Args:
            model_path: Path to model directory
            
        Returns:
            ResNetEncoder: Loaded model
        """
        import os
        import json
        
        # Load config
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(model_name=config["model_name"], pretrained=False)
        
        # Load model weights
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
        
        return model