# Image processing helpers

"""Image utility functions for VisionCapt."""

import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Union, List, Dict, Optional, Tuple, Any

def load_image(
    image_path_or_pil: Union[str, Image.Image], 
    target_size: Optional[Tuple[int, int]] = (224, 224),
    normalize: bool = True,
    mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
    std: List[float] = [0.26862954, 0.26130258, 0.27577711],
    to_tensor: bool = True,
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, Image.Image]:
    """
    Load an image from path or PIL Image and preprocess it.
    
    Args:
        image_path_or_pil: Path to image or PIL Image
        target_size: Target image size (height, width)
        normalize: Whether to normalize the image
        mean: Mean for normalization
        std: Standard deviation for normalization
        to_tensor: Whether to convert to tensor
        device: Device to put the tensor on
        
    Returns:
        torch.Tensor or PIL.Image: Processed image
    """
    # Load image
    if isinstance(image_path_or_pil, str):
        pil_image = Image.open(image_path_or_pil).convert("RGB")
    else:
        pil_image = image_path_or_pil
    
    # Resize if target size is provided
    if target_size is not None:
        pil_image = pil_image.resize(target_size, Image.BICUBIC)
    
    # Return PIL image if not converting to tensor
    if not to_tensor:
        return pil_image
    
    # Convert to tensor
    img_tensor = T.ToTensor()(pil_image)
    
    # Normalize if needed
    if normalize:
        img_tensor = T.Normalize(mean=mean, std=std)(img_tensor)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Move to device if specified
    if device is not None:
        img_tensor = img_tensor.to(device)
    
    return img_tensor

def save_image(
    tensor: torch.Tensor,
    output_path: str,
    denormalize: bool = True,
    mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
    std: List[float] = [0.26862954, 0.26130258, 0.27577711],
    quality: int = 95
) -> None:
    """
    Save a tensor as an image.
    
    Args:
        tensor: Image tensor
        output_path: Path to save the image
        denormalize: Whether to denormalize the image
        mean: Mean for denormalization
        std: Standard deviation for denormalization
        quality: JPEG quality
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed
    if denormalize:
        tensor = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL image
    pil_image = T.ToPILImage()(tensor)
    
    # Save image
    pil_image.save(output_path, quality=quality)

def create_image_grid(
    images: List[torch.Tensor],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Create a grid of images.
    
    Args:
        images: List of image tensors
        nrow: Number of images in each row
        padding: Padding between images
        normalize: Whether to normalize the grid
        value_range: Range of values to normalize to
        scale_each: Whether to scale each image individually
        pad_value: Value for padding
        
    Returns:
        torch.Tensor: Grid of images
    """
    from torchvision.utils import make_grid
    
    # Convert list of tensors to a single tensor
    if isinstance(images, list):
        # Remove batch dimension if present
        processed_images = []
        for img in images:
            if img.dim() == 4:
                img = img.squeeze(0)
            processed_images.append(img)
        
        # Stack images
        images = torch.stack(processed_images)
    
    # Create grid
    grid = make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value
    )
    
    return grid

def resize_image(
    image: Union[torch.Tensor, Image.Image],
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bicubic"
) -> Union[torch.Tensor, Image.Image]:
    """
    Resize an image.
    
    Args:
        image: Image to resize
        size: Target size (single value for shortest edge, or (height, width))
        interpolation: Interpolation method
        
    Returns:
        torch.Tensor or PIL.Image: Resized image
    """
    # Handle PIL image
    if isinstance(image, Image.Image):
        if interpolation == "bicubic":
            pil_interpolation = Image.BICUBIC
        elif interpolation == "bilinear":
            pil_interpolation = Image.BILINEAR
        elif interpolation == "nearest":
            pil_interpolation = Image.NEAREST
        else:
            pil_interpolation = Image.BICUBIC
        
        # Handle size
        if isinstance(size, int):
            # Resize smallest edge to size while maintaining aspect ratio
            w, h = image.size
            if h < w:
                new_h, new_w = size, int(size * w / h)
            else:
                new_h, new_w = int(size * h / w), size
            size = (new_w, new_h)
        
        # Resize image
        return image.resize(size, pil_interpolation)
    
    # Handle tensor
    else:
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Set interpolation mode
        if interpolation == "bicubic":
            mode = "bicubic"
        elif interpolation == "bilinear":
            mode = "bilinear"
        elif interpolation == "nearest":
            mode = "nearest"
        else:
            mode = "bicubic"
        
        # Set align_corners
        align_corners = False if mode in ["bilinear", "bicubic"] else None
        
        # Handle size
        if isinstance(size, int):
            # Get current size
            _, _, h, w = image.shape
            
            # Resize smallest edge to size while maintaining aspect ratio
            if h < w:
                new_h, new_w = size, int(size * w / h)
            else:
                new_h, new_w = int(size * h / w), size
            size = (new_h, new_w)
        
        # Resize image
        return F.interpolate(image, size=size, mode=mode, align_corners=align_corners)

def center_crop(
    image: Union[torch.Tensor, Image.Image],
    size: Union[int, Tuple[int, int]]
) -> Union[torch.Tensor, Image.Image]:
    """
    Center crop an image.
    
    Args:
        image: Image to crop
        size: Target size (single value for both dimensions, or (height, width))
        
    Returns:
        torch.Tensor or PIL.Image: Cropped image
    """
    # Handle PIL image
    if isinstance(image, Image.Image):
        if isinstance(size, int):
            size = (size, size)
        
        # Get current size
        w, h = image.size
        
        # Calculate crop coordinates
        left = (w - size[1]) // 2
        top = (h - size[0]) // 2
        right = left + size[1]
        bottom = top + size[0]
        
        # Crop image
        return image.crop((left, top, right, bottom))
    
    # Handle tensor
    else:
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Convert int to tuple
        if isinstance(size, int):
            size = (size, size)
        
        # Get current size
        _, _, h, w = image.shape
        
        # Calculate crop coordinates
        top = (h - size[0]) // 2
        left = (w - size[1]) // 2
        
        # Crop image
        return image[:, :, top:top+size[0], left:left+size[1]]

def apply_transforms(
    image: Union[str, Image.Image, torch.Tensor],
    transforms: List[Dict[str, Any]]
) -> Union[Image.Image, torch.Tensor]:
    """
    Apply a sequence of transformations to an image.
    
    Args:
        image: Image to transform
        transforms: List of transformation configs
        
    Returns:
        PIL.Image or torch.Tensor: Transformed image
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Apply each transformation
    for transform in transforms:
        transform_type = transform.get("type", "").lower()
        
        if transform_type == "resize":
            image = resize_image(
                image,
                size=transform.get("size", 224),
                interpolation=transform.get("interpolation", "bicubic")
            )
        
        elif transform_type == "center_crop":
            image = center_crop(
                image,
                size=transform.get("size", 224)
            )
        
        elif transform_type == "to_tensor":
            if not isinstance(image, torch.Tensor):
                image = T.ToTensor()(image)
        
        elif transform_type == "normalize":
            if isinstance(image, torch.Tensor):
                image = T.Normalize(
                    mean=transform.get("mean", [0.48145466, 0.4578275, 0.40821073]),
                    std=transform.get("std", [0.26862954, 0.26130258, 0.27577711])
                )(image)
        
        elif transform_type == "random_horizontal_flip":
            if isinstance(image, Image.Image) and torch.rand(1).item() < transform.get("p", 0.5):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(image, torch.Tensor) and torch.rand(1).item() < transform.get("p", 0.5):
                image = image.flip(-1)
        
        elif transform_type == "random_crop":
            size = transform.get("size", 224)
            if isinstance(image, Image.Image):
                i, j, h, w = T.RandomCrop.get_params(image, (size, size))
                image = image.crop((j, i, j + w, i + h))
            elif isinstance(image, torch.Tensor):
                i, j, h, w = T.RandomCrop.get_params(
                    image if image.dim() == 3 else image.permute(1, 2, 0),
                    (size, size)
                )
                image = image[:, i:i+h, j:j+w] if image.dim() == 3 else image[:, :, i:i+h, j:j+w]
    
    return image

def load_and_preprocess_image(
    image_path: str,
    target_size: int = 224,
    normalize: bool = True,
    to_tensor: bool = True,
    device: Optional[torch.device] = None,
    return_pil: bool = False
) -> Union[torch.Tensor, Image.Image]:
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to image
        target_size: Target image size
        normalize: Whether to normalize the image
        to_tensor: Whether to convert to tensor
        device: Device to put the tensor on
        return_pil: Whether to return PIL image instead of tensor
        
    Returns:
        torch.Tensor or PIL.Image: Processed image
    """
    # Define transforms
    transforms = []
    
    # Resize
    transforms.append({
        "type": "resize",
        "size": target_size,
        "interpolation": "bicubic"
    })
    
    # Center crop
    transforms.append({
        "type": "center_crop",
        "size": target_size
    })
    
    # Convert to tensor
    if to_tensor:
        transforms.append({
            "type": "to_tensor"
        })
    
    # Normalize
    if normalize and to_tensor:
        transforms.append({
            "type": "normalize",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711]
        })
    
    # Apply transforms
    image = apply_transforms(image_path, transforms)
    
    # Handle device
    if isinstance(image, torch.Tensor) and device is not None:
        image = image.to(device)
    
    # Add batch dimension if tensor
    if isinstance(image, torch.Tensor) and image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Convert back to PIL if requested
    if return_pil and isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = T.ToPILImage()(image)
    
    return image