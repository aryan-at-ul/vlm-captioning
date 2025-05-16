# visioncapt/model/language_model/builder.py

import logging
from typing import Union, Dict
from torch import nn
from .transformers_lm import TransformersLM
from .base_lm         import BaseLM
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def build_language_model(
    config: Union[Dict, "OmegaConf"]
) -> nn.Module:
    """
    Factory to build the language model based on config.
    
    Args:
        config: dict or OmegaConf with:
            - type: "transformers" or "base"
            - any other kwargs expected by the target LM class
    """
    # Convert OmegaConf to plain dict
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)

    # Pop out the type so it doesn't get passed as a constructor kwarg
    model_type = config.pop("type", "transformers").lower()

    if model_type == "transformers":
        logger.info("Building TransformersLM")
        # config now contains only valid kwargs for TransformersLM.__init__
        return TransformersLM(**config)

    elif model_type == "base":
        logger.info("Building BaseLM")
        return BaseLM(**config)

    else:
        raise ValueError(f"Unknown language model type: {model_type}")
