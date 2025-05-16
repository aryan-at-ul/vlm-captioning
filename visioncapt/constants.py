"""Constants for the VisionCapt model"""

# Special tokens
ENDOFTEXT = "<|endoftext|>"
IMAGE_TOKEN_BEGIN = "<image>"
IMAGE_TOKEN_END = "</image>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOI_TOKEN = "<image>"
DEFAULT_EOI_TOKEN = "</image>"

# Model constants
DEFAULT_IMAGE_SIZE = 224
DEFAULT_IMAGE_PATCH_SIZE = 32

# Training constants
DEFAULT_MAX_LENGTH = 77
DEFAULT_PAD_TOKEN_ID = 0

# Templates
DEFAULT_CAPTION_TEMPLATE = "Generate a caption for this image: {}"
SIMPLE_TEMPLATE = "{}"
INSTRUCT_TEMPLATE = "Below is an image. Please provide a detailed caption for this image.\n\n{}\n\n"

# Conversation templates
SYSTEM_PROMPT = "You are a helpful vision-language assistant. You are given an image and you need to generate a caption for it."
USER_PROMPT = "Generate a caption for this image: {}"
ASSISTANT_PROMPT = "{}"

# Evaluation constants
BLEU_MAX_ORDER = 4
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
METEOR_ALPHA = 0.9
CIDER_N = 4

# Small model options
SMALL_LANGUAGE_MODELS = [
    "google/gemma-2b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "stabilityai/stablelm-3b-4e1t",
    "facebook/opt-1.3b",
    "EleutherAI/pythia-1.4b",
    "bigscience/bloom-1b7",
]

SMALL_VISION_ENCODERS = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-small",
    "Alibaba-NLP/vilt-b32-mlm",
]

# LoRA configuration
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05