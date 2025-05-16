"""Register worker with controller."""

import os
import argparse
import requests
import logging
import json
import time

logger = logging.getLogger(__name__)

def register_worker(
    controller_address: str,
    worker_address: str,
    model_name: str,
    device: str = "cuda",
    max_batch_size: int = 4,
    load_8bit: bool = False,
    load_4bit: bool = False,
    log_level: str = "INFO"
):
    """
    Register worker with controller.
    
    Args:
        controller_address: Controller address
        worker_address: Worker address
        model_name: Model name
        device: Device
        max_batch_size: Maximum batch size
        load_8bit: Whether to load model in 8-bit precision
        load_4bit: Whether to load model in 4-bit precision
        log_level: Logging level
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Generate worker name
    import uuid
    worker_id = str(uuid.uuid4())[:8]
    worker_name = f"{model_name}-{worker_id}"
    
    # Create worker info
    worker_info = {
        "model_name": model_name,
        "worker_name": worker_name,
        "worker_address": worker_address,
        "device": device,
        "max_batch_size": max_batch_size,
        "load_8bit": load_8bit,
        "load_4bit": load_4bit,
    }
    
    # Register with controller
    try:
        logger.info(f"Registering worker with controller at {controller_address}")
        
        # Send registration
        url = f"{controller_address}/register_worker"
        response = requests.post(url, json=worker_info)
        
        if response.status_code == 200:
            logger.info("Successfully registered with controller")
            
            # Start heartbeat loop
            while True:
                try:
                    # Send heartbeat
                    url = f"{controller_address}/worker_heartbeat"
                    response = requests.post(
                        url,
                        json={
                            "worker_name": worker_name,
                            "queue_length": 0,
                        },
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Heartbeat failed: {response.text}")
                
                except Exception as e:
                    logger.warning(f"Heartbeat error: {e}")
                
                # Sleep for 10 seconds
                time.sleep(10)
        else:
            logger.error(f"Failed to register with controller: {response.text}")
    
    except Exception as e:
        logger.error(f"Error registering with controller: {e}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Register worker with controller")
    parser.add_argument("--controller_address", type=str, default="http://localhost:10000", help="Controller address")
    parser.add_argument("--worker_address", type=str, required=True, help="Worker address")
    parser.add_argument("--model_name", type=str, default="visioncapt", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Register worker
    register_worker(
        controller_address=args.controller_address,
        worker_address=args.worker_address,
        model_name=args.model_name,
        device=args.device,
        max_batch_size=args.max_batch_size,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()