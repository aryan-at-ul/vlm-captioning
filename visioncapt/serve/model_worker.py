# Model worker service
"""Model worker for distributed inference."""

import os
import json
import torch
import uuid
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from omegaconf import OmegaConf

from visioncapt.model.builder import build_model
from visioncapt.utils.utils import setup_logging
from visioncapt.utils.image_utils import load_and_preprocess_image

logger = logging.getLogger(__name__)

class ModelWorker:
    """
    Model worker for distributed inference.
    
    Handles loading models and processing inference requests.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        controller_address: str = "http://localhost:10000",
        worker_address: str = "localhost:50051",
        model_name: str = "visioncapt",
        no_register: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size: int = 4,
        load_8bit: bool = False,
        load_4bit: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize model worker.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to model config
            controller_address: Address of controller server
            worker_address: Address for this worker
            model_name: Name of the model
            no_register: Whether to skip registration with controller
            device: Device to use
            max_batch_size: Maximum batch size for inference
            load_8bit: Whether to load model in 8-bit precision
            load_4bit: Whether to load model in 4-bit precision
            log_level: Logging level
        """
        self.model_path = model_path
        self.config_path = config_path
        self.controller_address = controller_address
        self.worker_address = worker_address
        self.model_name = model_name
        self.no_register = no_register
        self.device = device
        self.max_batch_size = max_batch_size
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        
        # Set up logging
        setup_logging(log_level=log_level)
        
        # Set up workers
        self.worker_id = str(uuid.uuid4())[:8]
        self.worker_threads = []
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.load_model()
        
        # Register with controller
        if not self.no_register:
            self.register_with_controller()
        
        # Set up heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        # Set up inference threads
        for i in range(1):  # Single thread for inference
            thread = threading.Thread(target=self.inference_worker)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
    
    def load_model(self):
        """Load model from checkpoint."""
        try:
            # Load config
            config = OmegaConf.load(self.config_path)
            
            # Update config for inference
            if hasattr(config, "model") and hasattr(config.model, "language_model"):
                if self.load_8bit:
                    config.model.language_model.load_in_8bit = True
                if self.load_4bit:
                    config.model.language_model.load_in_4bit = True
                
                # Set device map for model parallelism if using quantized models
                if self.load_8bit or self.load_4bit:
                    config.model.language_model.device_map = "auto"
            
            # Build model
            self.model = build_model(config, model_path=self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Get model info
            from transformers import AutoTokenizer
            if hasattr(config.model.language_model, "model_name"):
                tokenizer_name = config.model.language_model.model_name
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    logger.info(f"Tokenizer loaded from {tokenizer_name}")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {e}")
                    self.tokenizer = None
            else:
                self.tokenizer = None
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def register_with_controller(self):
        """Register this worker with the controller."""
        try:
            import requests
            
            logger.info(f"Registering with controller at {self.controller_address}")
            
            # Model info
            model_info = {
                "model_name": self.model_name,
                "worker_name": f"{self.model_name}-{self.worker_id}",
                "worker_address": self.worker_address,
                "device": self.device,
                "max_batch_size": self.max_batch_size,
                "load_8bit": self.load_8bit,
                "load_4bit": self.load_4bit,
            }
            
            # Send registration
            url = f"{self.controller_address}/register_worker"
            response = requests.post(url, json=model_info)
            
            if response.status_code == 200:
                logger.info("Successfully registered with controller")
            else:
                logger.error(f"Failed to register with controller: {response.text}")
        
        except Exception as e:
            logger.error(f"Error registering with controller: {e}")
    
    def heartbeat_worker(self):
        """Periodically send heartbeat to controller."""
        import requests
        
        if self.no_register:
            return
        
        while not self.stop_event.is_set():
            try:
                url = f"{self.controller_address}/worker_heartbeat"
                response = requests.post(
                    url,
                    json={
                        "worker_name": f"{self.model_name}-{self.worker_id}",
                        "queue_length": self.request_queue.qsize(),
                    },
                )
                
                if response.status_code != 200:
                    logger.warning(f"Heartbeat failed: {response.text}")
            
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            
            # Sleep for 10 seconds
            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def inference_worker(self):
        """Process inference requests from queue."""
        while not self.stop_event.is_set():
            try:
                # Get request from queue with 1 second timeout
                request_data = self.request_queue.get(timeout=1)
                
                # Process request
                try:
                    result = self.process_request(request_data)
                    request_data["result"] = result
                    request_data["status"] = "success"
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    request_data["result"] = str(e)
                    request_data["status"] = "error"
                
                # Complete request
                request_data["done_event"].set()
                self.request_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
    
    def process_request(self, request_data: Dict) -> Dict:
        """
        Process an inference request.
        
        Args:
            request_data: Request data
            
        Returns:
            Dict: Result
        """
        # Get request parameters
        params = request_data["params"]
        
        # Handle different request types
        request_type = params.get("type", "generate_caption")
        
        if request_type == "generate_caption":
            return self.generate_caption(params)
        elif request_type == "encode_image":
            return self.encode_image(params)
        elif request_type == "answer_question":
            return self.answer_question(params)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def generate_caption(self, params: Dict) -> Dict:
        """
        Generate caption for an image.
        
        Args:
            params: Request parameters
            
        Returns:
            Dict: Generated caption
        """
        # Get image data
        image_data = params.get("image")
        
        # Convert image data to tensor
        if isinstance(image_data, str) and os.path.exists(image_data):
            # Load image from path
            image = load_and_preprocess_image(
                image_data,
                target_size=params.get("image_size", 224),
                normalize=True,
                to_tensor=True,
                device=self.device
            )
        elif isinstance(image_data, list) and len(image_data) > 0:
            # Handle base64 encoded image
            import base64
            from io import BytesIO
            
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data[0])
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
                # Preprocess image
                image = load_and_preprocess_image(
                    pil_image,
                    target_size=params.get("image_size", 224),
                    normalize=True,
                    to_tensor=True,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                raise
        else:
            raise ValueError("Invalid image data")
        
        # Generation parameters
        gen_params = {
            "max_length": params.get("max_length", 77),
            "min_length": params.get("min_length", 5),
            "num_beams": params.get("num_beams", 5),
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 50),
            "repetition_penalty": params.get("repetition_penalty", 1.0),
            "num_return_sequences": params.get("num_return_sequences", 1),
        }
        
        # Generate caption
        with torch.no_grad():
            captions = self.model.generate_captions(
                images=image,
                **gen_params
            )
        
        # Return result
        return {
            "captions": captions,
            "params": gen_params
        }
    
    def encode_image(self, params: Dict) -> Dict:
        """
        Encode an image to features.
        
        Args:
            params: Request parameters
            
        Returns:
            Dict: Encoded features
        """
        # Get image data
        image_data = params.get("image")
        
        # Convert image data to tensor
        if isinstance(image_data, str) and os.path.exists(image_data):
            # Load image from path
            image = load_and_preprocess_image(
                image_data,
                target_size=params.get("image_size", 224),
                normalize=True,
                to_tensor=True,
                device=self.device
            )
        elif isinstance(image_data, list) and len(image_data) > 0:
            # Handle base64 encoded image
            import base64
            from io import BytesIO
            
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data[0])
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
                # Preprocess image
                image = load_and_preprocess_image(
                    pil_image,
                    target_size=params.get("image_size", 224),
                    normalize=True,
                    to_tensor=True,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                raise
        else:
            raise ValueError("Invalid image data")
        
        # Extract features
        with torch.no_grad():
            # Forward pass through vision encoder
            image_features = self.model.vision_encoder(image)
            
            # Project features if requested
            return_projections = params.get("return_projections", False)
            if return_projections:
                projected_features = self.model.projection(image_features)
                features_dict = {
                    "image_features": image_features.cpu().numpy().tolist(),
                    "projected_features": projected_features.cpu().numpy().tolist()
                }
            else:
                features_dict = {
                    "image_features": image_features.cpu().numpy().tolist()
                }
        
        # Return result
        return features_dict
    
    def answer_question(self, params: Dict) -> Dict:
        """
        Answer a question about an image.
        
        Args:
            params: Request parameters
            
        Returns:
            Dict: Answer
        """
        # Get image data
        image_data = params.get("image")
        
        # Get question
        question = params.get("question", "What is in this image?")
        
        # Convert image data to tensor
        if isinstance(image_data, str) and os.path.exists(image_data):
            # Load image from path
            image = load_and_preprocess_image(
                image_data,
                target_size=params.get("image_size", 224),
                normalize=True,
                to_tensor=True,
                device=self.device
            )
        elif isinstance(image_data, list) and len(image_data) > 0:
            # Handle base64 encoded image
            import base64
            from io import BytesIO
            
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data[0])
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
                # Preprocess image
                image = load_and_preprocess_image(
                    pil_image,
                    target_size=params.get("image_size", 224),
                    normalize=True,
                    to_tensor=True,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                raise
        else:
            raise ValueError("Invalid image data")
        
        # Generate answer
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "tokenizer"):
            # Create input prompt with question
            tokenizer = self.model.language_model.tokenizer
            
            # Process the question
            if not question.strip().endswith("?"):
                question = question.strip() + "?"
            
            # Create prompt with image token
            prompt = f"<image> {question}"
            
            # Tokenize prompt
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=params.get("max_length", 77) // 2,
                truncation=True
            ).input_ids.to(self.device)
            
            # Generate answer
            with torch.no_grad():
                answer = self.model.generate_captions(
                    images=image,
                    input_ids=input_ids,
                    max_length=params.get("max_length", 77),
                    min_length=params.get("min_length", 5),
                    num_beams=params.get("num_beams", 5),
                    temperature=params.get("temperature", 1.0),
                    top_p=params.get("top_p", 0.9),
                    top_k=params.get("top_k", 50),
                    repetition_penalty=params.get("repetition_penalty", 1.0),
                )[0]
            
            # Extract answer (remove the question part)
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()
        else:
            # Fallback to caption generation
            with torch.no_grad():
                answer = self.model.generate_captions(
                    images=image,
                    max_length=params.get("max_length", 77),
                    min_length=params.get("min_length", 5),
                    num_beams=params.get("num_beams", 5),
                    temperature=params.get("temperature", 1.0),
                    top_p=params.get("top_p", 0.9),
                    top_k=params.get("top_k", 50),
                    repetition_penalty=params.get("repetition_penalty", 1.0),
                )[0]
        
        # Return result
        return {
            "question": question,
            "answer": answer
        }
    
    def add_request(self, request_data: Dict) -> None:
        """
        Add a request to the queue.
        
        Args:
            request_data: Request data
        """
        # Create done event
        request_data["done_event"] = threading.Event()
        
        # Add to queue
        self.request_queue.put(request_data)
        
        logger.info(f"Added request to queue (size: {self.request_queue.qsize()})")
    
    def wait_for_request(self, request_id: str, timeout: Optional[float] = None) -> Dict:
        """
        Wait for a request to be processed.
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            Dict: Request data
        """
        # Find request in queue
        for item in list(self.request_queue.queue):
            if item.get("request_id") == request_id:
                # Wait for request to be processed
                done = item["done_event"].wait(timeout=timeout)
                
                if done:
                    return item
                else:
                    raise TimeoutError(f"Request {request_id} timed out")
        
        raise ValueError(f"Request {request_id} not found")
    
    def stop(self):
        """Stop all workers."""
        logger.info("Stopping worker...")
        self.stop_event.set()
        
        # Wait for all threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=1)
        
        self.heartbeat_thread.join(timeout=1)
        
        logger.info("Worker stopped")


def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="VisionCapt model worker")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--controller_address", type=str, default="http://localhost:10000", help="Controller address")
    parser.add_argument("--worker_address", type=str, default="localhost:50051", help="Worker address")
    parser.add_argument("--model_name", type=str, default="visioncapt", help="Model name")
    parser.add_argument("--no_register", action="store_true", help="Skip registration with controller")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Maximum batch size for inference")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--enable_grpc", action="store_true", help="Enable gRPC server")
    parser.add_argument("--enable_http", action="store_true", help="Enable HTTP server")
    args = parser.parse_args()
    
    # Start worker
    worker = ModelWorker(
        model_path=args.model_path,
        config_path=args.config,
        controller_address=args.controller_address,
        worker_address=args.worker_address,
        model_name=args.model_name,
        no_register=args.no_register,
        device=args.device,
        max_batch_size=args.max_batch_size,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        log_level=args.log_level,
    )
    
    try:
        # Start HTTP server
        if args.enable_http:
            import uvicorn
            from fastapi import FastAPI, BackgroundTasks, HTTPException
            from pydantic import BaseModel, Field
            from typing import List, Optional
            
            app = FastAPI()
            
            class CaptionRequest(BaseModel):
                image: Union[str, List[str]]
                max_length: int = 77
                min_length: int = 5
                num_beams: int = 5
                temperature: float = 1.0
                top_p: float = 0.9
                top_k: int = 50
                repetition_penalty: float = 1.0
                num_return_sequences: int = 1
                image_size: int = 224
            
            class QuestionRequest(BaseModel):
                image: Union[str, List[str]]
                question: str
                max_length: int = 77
                min_length: int = 5
                num_beams: int = 5
                temperature: float = 1.0
                top_p: float = 0.9
                top_k: int = 50
                repetition_penalty: float = 1.0
                image_size: int = 224
            
            @app.post("/generate_caption")
            async def generate_caption(request: CaptionRequest, background_tasks: BackgroundTasks):
                try:
                    # Convert request to params
                    params = {
                        "type": "generate_caption",
                        "image": request.image,
                        "max_length": request.max_length,
                        "min_length": request.min_length,
                        "num_beams": request.num_beams,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "repetition_penalty": request.repetition_penalty,
                        "num_return_sequences": request.num_return_sequences,
                        "image_size": request.image_size,
                    }
                    
                    # Add request to queue
                    request_id = str(uuid.uuid4())
                    request_data = {
                        "request_id": request_id,
                        "params": params
                    }
                    
                    worker.add_request(request_data)
                    
                    # Wait for request to be processed
                    result = worker.wait_for_request(request_id, timeout=30)
                    
                    if result.get("status") == "success":
                        return result.get("result", {})
                    else:
                        raise HTTPException(status_code=500, detail=result.get("result", "Unknown error"))
                
                except Exception as e:
                    logger.error(f"HTTP error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.post("/answer_question")
            async def answer_question(request: QuestionRequest, background_tasks: BackgroundTasks):
                try:
                    # Convert request to params
                    params = {
                        "type": "answer_question",
                        "image": request.image,
                        "question": request.question,
                        "max_length": request.max_length,
                        "min_length": request.min_length,
                        "num_beams": request.num_beams,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "top_k": request.top_k,
                        "repetition_penalty": request.repetition_penalty,
                        "image_size": request.image_size,
                    }
                    
                    # Add request to queue
                    request_id = str(uuid.uuid4())
                    request_data = {
                        "request_id": request_id,
                        "params": params
                    }
                    
                    worker.add_request(request_data)
                    
                    # Wait for request to be processed
                    result = worker.wait_for_request(request_id, timeout=30)
                    
                    if result.get("status") == "success":
                        return result.get("result", {})
                    else:
                        raise HTTPException(status_code=500, detail=result.get("result", "Unknown error"))
                
                except Exception as e:
                    logger.error(f"HTTP error: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Start server
            host, port = args.worker_address.split(":")
            uvicorn.run(app, host=host, port=int(port))
        else:
            # Wait for keyboard interrupt
            logger.info("Worker started. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping worker...")
    
    finally:
        worker.stop()


if __name__ == "__main__":
    main()