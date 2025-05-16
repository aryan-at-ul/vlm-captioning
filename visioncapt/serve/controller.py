"""Controller for distributed inference."""

import os
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
import uvicorn

logger = logging.getLogger(__name__)

class WorkerInfo(BaseModel):
    """Information about a worker."""
    model_name: str
    worker_name: str
    worker_address: str
    device: str
    max_batch_size: int = 4
    load_8bit: bool = False
    load_4bit: bool = False
    last_heartbeat: float = 0.0
    queue_length: int = 0

class Controller:
    """
    Controller for distributed inference.
    
    Manages a pool of model workers and dispatches requests to them.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 10000,
        worker_timeout: float = 60.0,
        log_level: str = "INFO",
    ):
        """
        Initialize controller.
        
        Args:
            host: Host to listen on
            port: Port to listen on
            worker_timeout: Timeout for worker heartbeats in seconds
            log_level: Logging level
        """
        self.host = host
        self.port = port
        self.worker_timeout = worker_timeout
        
        # Set up logging
        self._setup_logging(log_level)
        
        # Set up FastAPI app
        self.app = FastAPI()
        
        # Set up worker registry
        self.worker_registry: Dict[str, WorkerInfo] = {}
        
        # Set up cleaner thread
        self.stop_event = threading.Event()
        self.cleaner_thread = threading.Thread(target=self._worker_cleanup_thread)
        self.cleaner_thread.daemon = True
        
        # Set up routes
        self._setup_routes()
    
    def _setup_logging(self, log_level: str):
        """Set up logging."""
        # Get numeric logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        
        # Configure logging
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    def _setup_routes(self):
        """Set up FastAPI routes."""
        # Register worker
        @self.app.post("/register_worker")
        async def register_worker(worker_info: WorkerInfo):
            return self.register_worker(worker_info)
        
        # Worker heartbeat
        @self.app.post("/worker_heartbeat")
        async def worker_heartbeat(request: Request):
            data = await request.json()
            return self.update_worker_heartbeat(data)
        
        # List models
        @self.app.get("/list_models")
        async def list_models():
            return self.list_models()
        
        # List workers
        @self.app.get("/list_workers")
        async def list_workers():
            return self.list_workers()
        
        # Generate caption
        @self.app.post("/generate_caption")
        async def generate_caption(request: Request):
            data = await request.json()
            return await self.generate_caption(data)
        
        # Answer question
        @self.app.post("/answer_question")
        async def answer_question(request: Request):
            data = await request.json()
            return await self.answer_question(data)
        
        # Health check
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}
    
    def register_worker(self, worker_info: WorkerInfo) -> Dict:
        """
        Register a new worker.
        
        Args:
            worker_info: Worker information
            
        Returns:
            Dict: Registration result
        """
        # Set initial heartbeat
        worker_info.last_heartbeat = time.time()
        
        # Add to registry
        self.worker_registry[worker_info.worker_name] = worker_info
        
        logger.info(f"Registered worker: {worker_info.worker_name} ({worker_info.model_name} on {worker_info.device})")
        
        return {"status": "ok"}
    
    def update_worker_heartbeat(self, data: Dict) -> Dict:
        """
        Update worker heartbeat.
        
        Args:
            data: Heartbeat data
            
        Returns:
            Dict: Heartbeat result
        """
        worker_name = data.get("worker_name")
        queue_length = data.get("queue_length", 0)
        
        if worker_name in self.worker_registry:
            self.worker_registry[worker_name].last_heartbeat = time.time()
            self.worker_registry[worker_name].queue_length = queue_length
            
            return {"status": "ok"}
        else:
            raise HTTPException(status_code=404, detail=f"Worker {worker_name} not found")
    
    def list_models(self) -> Dict:
        """
        List available models.
        
        Returns:
            Dict: Model list
        """
        models = {}
        
        for worker_name, worker_info in self.worker_registry.items():
            if time.time() - worker_info.last_heartbeat <= self.worker_timeout:
                if worker_info.model_name not in models:
                    models[worker_info.model_name] = {
                        "workers": []
                    }
                
                models[worker_info.model_name]["workers"].append({
                    "worker_name": worker_name,
                    "worker_address": worker_info.worker_address,
                    "device": worker_info.device,
                    "queue_length": worker_info.queue_length,
                })
        
        return {"models": models}
    
    def list_workers(self) -> Dict:
        """
        List active workers.
        
        Returns:
            Dict: Worker list
        """
        workers = {}
        
        for worker_name, worker_info in self.worker_registry.items():
            if time.time() - worker_info.last_heartbeat <= self.worker_timeout:
                workers[worker_name] = {
                    "model_name": worker_info.model_name,
                    "worker_address": worker_info.worker_address,
                    "device": worker_info.device,
                    "max_batch_size": worker_info.max_batch_size,
                    "load_8bit": worker_info.load_8bit,
                    "load_4bit": worker_info.load_4bit,
                    "last_heartbeat": worker_info.last_heartbeat,
                    "queue_length": worker_info.queue_length,
                }
        
        return {"workers": workers}
    
    def _find_suitable_worker(self, model_name: str) -> Optional[WorkerInfo]:
        """
        Find a suitable worker for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Optional[WorkerInfo]: Worker information if found, None otherwise
        """
        suitable_workers = []
        
        # Find workers for the model
        for worker_name, worker_info in self.worker_registry.items():
            if (worker_info.model_name == model_name and 
                time.time() - worker_info.last_heartbeat <= self.worker_timeout):
                suitable_workers.append(worker_info)
        
        if not suitable_workers:
            return None
        
        # Sort by queue length (shortest first)
        suitable_workers.sort(key=lambda w: w.queue_length)
        
        return suitable_workers[0]
    
    async def generate_caption(self, data: Dict) -> Dict:
        """
        Generate caption for an image.
        
        Args:
            data: Request data
            
        Returns:
            Dict: Generated caption
        """
        model_name = data.get("model_name", "visioncapt")
        
        # Find suitable worker
        worker = self._find_suitable_worker(model_name)
        
        if worker is None:
            raise HTTPException(status_code=404, detail=f"No active workers found for model {model_name}")
        
        # Forward request to worker
        try:
            response = requests.post(
                f"http://{worker.worker_address}/generate_caption",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        
        except Exception as e:
            logger.error(f"Error forwarding request to worker: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def answer_question(self, data: Dict) -> Dict:
        """
        Answer a question about an image.
        
        Args:
            data: Request data
            
        Returns:
            Dict: Answer
        """
        model_name = data.get("model_name", "visioncapt")
        
        # Find suitable worker
        worker = self._find_suitable_worker(model_name)
        
        if worker is None:
            raise HTTPException(status_code=404, detail=f"No active workers found for model {model_name}")
        
        # Forward request to worker
        try:
            response = requests.post(
                f"http://{worker.worker_address}/answer_question",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        
        except Exception as e:
            logger.error(f"Error forwarding request to worker: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _worker_cleanup_thread(self):
        """Clean up stale workers."""
        while not self.stop_event.is_set():
            try:
                # Find stale workers
                stale_workers = []
                
                for worker_name, worker_info in self.worker_registry.items():
                    if time.time() - worker_info.last_heartbeat > self.worker_timeout:
                        stale_workers.append(worker_name)
                
                # Remove stale workers
                for worker_name in stale_workers:
                    logger.info(f"Removing stale worker: {worker_name}")
                    del self.worker_registry[worker_name]
            
            except Exception as e:
                logger.error(f"Error in worker cleanup thread: {e}")
            
            # Sleep for a while
            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def start(self):
        """Start the controller."""
        # Start cleaner thread
        self.cleaner_thread.start()
        
        # Start FastAPI server
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def stop(self):
        """Stop the controller."""
        self.stop_event.set()
        self.cleaner_thread.join(timeout=5)


def main():
    """Main function."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="VisionCapt controller")
    parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    parser.add_argument("--port", type=int, default=10000, help="Port to listen on")
    parser.add_argument("--worker_timeout", type=float, default=60.0, help="Timeout for worker heartbeats in seconds")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Start controller
    controller = Controller(
        host=args.host,
        port=args.port,
        worker_timeout=args.worker_timeout,
        log_level=args.log_level,
    )
    
    try:
        controller.start()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping controller...")
    
    finally:
        controller.stop()


if __name__ == "__main__":
    main()