"""Data utilities for training and evaluation"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import logging
from transformers import CLIPImageProcessor
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class Flickr8kDataset(Dataset):
    """
    Dataset for Flickr8k
    
    Loads images and captions from Flickr8k dataset
    """
    
    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        transform=None,
        tokenizer=None,
        max_seq_length: int = 77,
        split: str = "train"
    ):
        """
        Initialize Flickr8k dataset
        
        Args:
            image_dir: Directory containing images
            captions_file: Path to captions file
            transform: Image transformation function
            tokenizer: Tokenizer for encoding captions
            max_seq_length: Maximum sequence length for captions
            split: Dataset split (train, val, test)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        
        # Load captions
        self.captions_df = self._load_captions(captions_file)
        
        # Filter by split if specified
        if "split" in self.captions_df.columns:
            self.captions_df = self.captions_df[self.captions_df["split"] == split]
        
        logger.info(f"Loaded {len(self.captions_df)} caption entries for split: {split}")
    
    def _load_captions(self, captions_file: str) -> pd.DataFrame:
        """
        Load captions from file
        
        Args:
            captions_file: Path to captions file
            
        Returns:
            pd.DataFrame: DataFrame containing image filenames and captions
        """
        # Check file extension
        ext = os.path.splitext(captions_file)[1]
        
        if ext == ".txt":
            # Load from text file (typical Flickr8k format)
            # Format: image_name#id caption
            lines = []
            with open(captions_file, "r") as f:
                for line in f:
                    if line.strip():
                        components = line.strip().split(" ", 1)
                        image_name = components[0].split("#")[0]
                        caption = components[1].strip()
                        lines.append({"image_name": image_name, "caption": caption})
            
            return pd.DataFrame(lines)
        
        elif ext == ".csv":
            # Load from CSV file
            return pd.read_csv(captions_file)
        
        elif ext in [".json", ".jsonl"]:
            # Load from JSON file
            with open(captions_file, "r") as f:
                if ext == ".jsonl":
                    # JSONL format (one JSON object per line)
                    data = [json.loads(line) for line in f]
                else:
                    # Regular JSON format
                    data = json.load(f)
            
            return pd.DataFrame(data)
        
        else:
            raise ValueError(f"Unsupported captions file format: {ext}")
    
    def __len__(self) -> int:
        """
        Get dataset length
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.captions_df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            dict: Dictionary containing image, caption, and metadata
        """
        # Get caption and image name
        caption_item = self.captions_df.iloc[idx]
        image_name = caption_item["image_name"]
        caption = caption_item["caption"]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # Apply transform if specified
        if self.transform is not None:
            image = self.transform(image)
        
        # Create sample
        sample = {
            "image": image,
            "caption": caption,
            "image_name": image_name
        }
        
        # Tokenize caption if tokenizer is provided
        if self.tokenizer is not None:
            # Add special tokens for image and text
            text = f"<|startoftext|> <image> {caption} <|endoftext|>"
            
            # Tokenize
            encodings = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Add to sample
            sample["input_ids"] = encodings["input_ids"].squeeze(0)
            sample["attention_mask"] = encodings["attention_mask"].squeeze(0)
            
            # Create labels for causal LM (shifted input_ids)
            labels = encodings["input_ids"].clone()
            
            # Set tokens before image token to -100 (ignored in loss calculation)
            try:
                image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
                image_token_pos = (labels == image_token_id).nonzero(as_tuple=True)[1][0]
                labels[0, :image_token_pos+1] = -100  # Ignore BOS and image token
            except (IndexError, AttributeError):
                # If image token not found, just ignore padding tokens
                padding_token_id = self.tokenizer.pad_token_id
                if padding_token_id is not None:
                    labels[labels == padding_token_id] = -100
            
            sample["labels"] = labels.squeeze(0)
        
        return sample


def create_flickr8k_dataloader(
    image_dir: str,
    captions_file: str,
    tokenizer,
    image_processor: Optional[Any] = None,
    batch_size: int = 32,
    max_seq_length: int = 77,
    split: str = "train",
    shuffle: bool = True,
    is_distributed: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """
    Create Flickr8k dataloader
    
    Args:
        image_dir: Directory containing images
        captions_file: Path to captions file
        tokenizer: Tokenizer for encoding captions
        image_processor: Image processor
        batch_size: Batch size
        max_seq_length: Maximum sequence length for captions
        split: Dataset split (train, val, test)
        shuffle: Whether to shuffle the data
        is_distributed: Whether to use distributed training
        num_workers: Number of dataloader workers
        
    Returns:
        DataLoader: DataLoader for Flickr8k dataset
    """
    # Create image transform
    if image_processor is None:
        # Default transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    else:
        # Use provided image processor
        if isinstance(image_processor, CLIPImageProcessor):
            # CLIP processor
            def transform(image):
                return image_processor(
                    images=image, 
                    return_tensors="pt"
                ).pixel_values[0]
        else:
            # Custom processor
            transform = image_processor
    
    # Create dataset
    dataset = Flickr8kDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        transform=transform,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        split=split
    )
    
    # Set up sampler for distributed training
    sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Shuffle is handled by sampler
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader


def prepare_flickr8k(
    flickr_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Prepare Flickr8k dataset for training
    
    Args:
        flickr_path: Path to Flickr8k dataset
        output_dir: Output directory
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        seed: Random seed
    """
    import numpy as np
    import shutil
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Load captions
    captions_file = os.path.join(flickr_path, "captions.txt")
    
    if not os.path.exists(captions_file):
        captions_file = os.path.join(flickr_path, "Flickr8k.token.txt")
    
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found at {captions_file}")
    
    # Read and format captions
    logger.info(f"Reading captions from {captions_file}")
    captions = []
    with open(captions_file, "r") as f:
        for line in f:
            if line.strip():
                components = line.strip().split(" ", 1)
                image_name = components[0].split("#")[0]
                caption = components[1].strip()
                captions.append({"image_name": image_name, "caption": caption})
    
    # Convert to DataFrame
    captions_df = pd.DataFrame(captions)
    
    # Get unique image names
    image_names = captions_df["image_name"].unique()
    
    # Shuffle image names
    np.random.shuffle(image_names)
    
    # Split into train, val, test
    n_images = len(image_names)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    train_images = image_names[:n_train]
    val_images = image_names[n_train:n_train+n_val]
    test_images = image_names[n_train+n_val:]
    
    # Create split column
    def get_split(image_name):
        if image_name in train_images:
            return "train"
        elif image_name in val_images:
            return "val"
        else:
            return "test"
    
    captions_df["split"] = captions_df["image_name"].apply(get_split)
    
    # Save processed captions
    logger.info("Saving processed captions")
    captions_df.to_csv(os.path.join(output_dir, "captions.csv"), index=False)
    
    # Save split-specific captions
    for split in ["train", "val", "test"]:
        split_df = captions_df[captions_df["split"] == split]
        split_df.to_csv(os.path.join(output_dir, f"{split}_captions.csv"), index=False)
    
    # Copy images
    image_dir = os.path.join(flickr_path, "Images")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(flickr_path, "images")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found at {image_dir}")
    
    logger.info(f"Copying images from {image_dir} to {os.path.join(output_dir, 'images')}")
    for image_name in image_names:
        src_path = os.path.join(image_dir, image_name)
        dst_path = os.path.join(output_dir, "images", image_name)
        shutil.copy(src_path, dst_path)
    
    logger.info(f"Dataset prepared and saved to {output_dir}")