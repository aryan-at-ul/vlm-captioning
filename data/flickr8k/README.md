# Flickr8k Dataset

This directory should contain the Flickr8k dataset files after running the preparation script.

## Dataset Structure

After running the preparation script (`scripts/preprocess/prepare_flickr8k.py`), the directory structure should look like:

```
data/flickr8k_processed/
├── captions.csv          # All captions with split information
├── train_captions.csv    # Training split captions
├── val_captions.csv      # Validation split captions
├── test_captions.csv     # Test split captions
├── images/               # Directory containing all image files
└── README.md             # Dataset information
```

## Getting the Flickr8k Dataset

The Flickr8k dataset can be downloaded from:
- [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [University of Illinois](https://forms.illinois.edu/sec/1713398)

## Preparing the Dataset

To prepare the dataset, run:

```bash
python scripts/preprocess/prepare_flickr8k.py --flickr_path /path/to/flickr8k --output_dir data/flickr8k_processed
```

This will:
1. Extract captions from the text files
2. Split images into train/val/test sets
3. Copy images to the output directory
4. Create CSV files with captions and split information

## Dataset Statistics

- Total images: 8,092
- Total captions: 40,460 (5 captions per image)

Split by default:
- Training: 80% (~6,474 images)
- Validation: 10% (~809 images)
- Test: 10% (~809 images)