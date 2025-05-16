"""Quick test to make sure the scheduler works."""

import os
import sys
import torch
import logging
from omegaconf import OmegaConf

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, project_root)

from visioncapt.utils.utils import get_scheduler, setup_logging

def main():
    """Test the scheduler function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create a dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Load the config
    config_path = os.path.join(project_root, "configs/lora_config.yaml")
    config = OmegaConf.load(config_path)
    
    # Create the scheduler
    num_training_steps = 10000
    scheduler = get_scheduler(optimizer, config.training.scheduler, num_training_steps)
    
    # Test the scheduler
    logger.info("Testing scheduler...")
    logger.info(f"Initial learning rate: {scheduler.get_last_lr()[0]}")
    
    # Test a few steps
    for step in range(5):
        optimizer.step()
        scheduler.step()
        logger.info(f"Step {step+1}, learning rate: {scheduler.get_last_lr()[0]}")
    
    # Test middle of training
    mid_step = num_training_steps // 2
    for _ in range(mid_step - 5):
        optimizer.step()
        scheduler.step()
    
    logger.info(f"Mid-training (step {mid_step}), learning rate: {scheduler.get_last_lr()[0]}")
    
    # Test end of training
    for _ in range(num_training_steps - mid_step - 1):
        optimizer.step()
        scheduler.step()
    
    logger.info(f"End of training (step {num_training_steps}), learning rate: {scheduler.get_last_lr()[0]}")
    
    # Verify that the final learning rate respects min_lr_ratio
    initial_lr = 0.001
    min_lr_ratio = config.training.scheduler.min_lr_ratio
    expected_min_lr = initial_lr * min_lr_ratio
    
    logger.info(f"Min LR ratio: {min_lr_ratio}")
    logger.info(f"Expected minimum learning rate: {expected_min_lr}")
    logger.info(f"Actual final learning rate: {scheduler.get_last_lr()[0]}")
    
    if abs(scheduler.get_last_lr()[0] - expected_min_lr) < 1e-6:
        logger.info("✅ Scheduler test PASSED!")
    else:
        logger.error("❌ Scheduler test FAILED! Final learning rate does not match expected minimum.")

if __name__ == "__main__":
    main()