"""
Model Manager for Local LLM
Manages lifecycle of local language models (Llama 2, Mistral, etc).
Supports Ollama for easy deployment and model switching.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelStatus(Enum):
    """Model status indicators."""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"


@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: ModelProvider
    model_id: str  # e.g., "mistral:7b" for Ollama
    description: str = ""
    context_length: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ModelMetrics:
    """Metrics about model performance and status."""
    status: ModelStatus
    last_check: str
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    tokens_processed: int = 0
    inference_count: int = 0


class ModelConnectionError(Exception):
    """Raised when model connection fails."""
    pass


class ModelInferenceError(Exception):
    """Raised when model inference fails."""
    pass


class LocalLMManager:
    """
    Manages local language model lifecycle and inference.
    
    Supports:
    - Ollama for easy local LLM deployment
    - Hugging Face Transformers for direct local models
    - Health checks and status monitoring
    - Multi-model support with switching
    """

    # Recommended models for proposal generation
    RECOMMENDED_MODELS = {
        "llama3.1:8b": ModelConfig(
            name="Llama 3.1 8B",
            provider=ModelProvider.OLLAMA,
            model_id="llama3.1:8b",
            description="Best quality - excellent for proposal generation and reasoning.",
            context_length=8192,
            max_tokens=2048
        ),
        "mistral:7b-instruct": ModelConfig(
            name="Mistral 7B Instruct",
            provider=ModelProvider.OLLAMA,
            model_id="mistral:7b-instruct",
            description="Instruction-tuned. Fastest option with good quality.",
            context_length=8192,
            max_tokens=2048
        ),
        "deepseek-r1:8b": ModelConfig(
            name="DeepSeek R1 8B",
            provider=ModelProvider.OLLAMA,
            model_id="deepseek-r1:8b",
            description="Best reasoning. Ideal for requirement extraction and analysis.",
            context_length=4096,
            max_tokens=2048
        ),
        "mistral:7b": ModelConfig(
            name="Mistral 7B (Base)",
            provider=ModelProvider.OLLAMA,
            model_id="mistral:7b",
            description="Fast baseline model. Good fallback option.",
            context_length=8192,
            max_tokens=2048
        ),
    }

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b"
    ):
        """
        Initialize model manager.
        
        Args:
            ollama_base_url: Base URL for Ollama API
            default_model: Default model to load (llama3.1:8b recommended)
        """
        self.ollama_base_url = ollama_base_url
        self.default_model_id = default_model
        self.current_model: Optional[ModelConfig] = None
        self.metrics = ModelMetrics(
            status=ModelStatus.UNINITIALIZED,
            last_check=datetime.now().isoformat()
        )
        self._inference_cache: Dict[str, str] = {}
        
        logger.info(f"LocalLMManager initialized with Ollama at {ollama_base_url}")

    # ========================
    # Model Health & Status
    # ========================

    def check_ollama_availability(self) -> bool:
        """
        Check if Ollama service is available and running.
        
        Returns:
            bool: True if Ollama is reachable
        """
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=3
            )
            available = response.status_code == 200
            
            if available:
                logger.info("Ollama service is available")
                self.metrics.status = ModelStatus.READY
            else:
                logger.warning("Ollama service returned non-200 status")
                self.metrics.status = ModelStatus.NOT_AVAILABLE
            
            self.metrics.last_check = datetime.now().isoformat()
            return available
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama availability check failed: {str(e)}")
            self.metrics.status = ModelStatus.NOT_AVAILABLE
            self.metrics.error_message = str(e)
            self.metrics.last_check = datetime.now().isoformat()
            return False

    def list_available_models(self) -> List[str]:
        """
        Get list of models available in Ollama.
        
        Returns:
            List[str]: List of model names
            
        Raises:
            ModelConnectionError: If Ollama is not available
        """
        if not self.check_ollama_availability():
            raise ModelConnectionError(
                "Ollama service not available. "
                "Please install and start Ollama: https://ollama.in"
            )
        
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            logger.info(f"Available models in Ollama: {models}")
            return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise ModelConnectionError(f"Failed to list models: {str(e)}")

    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_id: Model identifier
            
        Returns:
            bool: True if model is available
        """
        try:
            available_models = self.list_available_models()
            return model_id in available_models
        except ModelConnectionError:
            return False

    # ========================
    # Model Loading & Management
    # ========================

    def load_model(self, model_id: Optional[str] = None) -> ModelConfig:
        """
        Load a model for inference.
        
        Args:
            model_id: Model identifier (e.g., 'mistral:7b'). 
                     If None, uses default model.
        
        Returns:
            ModelConfig: Configuration of loaded model
            
        Raises:
            ModelConnectionError: If model cannot be loaded
        """
        model_id = model_id or self.default_model_id
        
        # Check if model is recommended (for easy reference)
        if model_id in self.RECOMMENDED_MODELS:
            self.current_model = self.RECOMMENDED_MODELS[model_id]
        else:
            self.current_model = ModelConfig(
                name=model_id,
                provider=ModelProvider.OLLAMA,
                model_id=model_id
            )
        
        logger.info(f"Setting current model to: {model_id}")
        
        # Verify model availability
        if not self.is_model_available(model_id):
            raise ModelConnectionError(
                f"Model '{model_id}' is not available in Ollama. "
                f"Available models: {self.list_available_models()}"
            )
        
        self.metrics.status = ModelStatus.READY
        self.metrics.last_check = datetime.now().isoformat()
        
        return self.current_model

    def unload_model(self) -> None:
        """Unload current model."""
        self.current_model = None
        self.metrics.status = ModelStatus.UNINITIALIZED
        logger.info("Model unloaded")

    def get_current_model(self) -> Optional[ModelConfig]:
        """Get currently loaded model configuration."""
        return self.current_model

    # ========================
    # Model Inference
    # ========================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        timeout: int = 120
    ) -> str:
        """
        Generate text using the current model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (uses model default if None)
            temperature: Sampling temperature (uses model default if None)
            top_p: Nucleus sampling parameter (uses model default if None)
            system_prompt: System/instruction prompt for role-based behavior
            stream: Whether to stream response (returns after full response currently)
            timeout: Request timeout in seconds (default: 120)
        
        Returns:
            str: Generated text
            
        Raises:
            ModelInferenceError: If generation fails
            ModelConnectionError: If model is not loaded
        """
        if not self.current_model:
            raise ModelConnectionError("No model loaded. Call load_model() first.")
        
        # Use model defaults if not specified
        max_tokens = max_tokens or self.current_model.max_tokens
        temperature = temperature or self.current_model.temperature
        top_p = top_p or self.current_model.top_p
        
        # Build full prompt with system message
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.current_model.model_id,
                    "prompt": full_prompt,
                    "stream": False,  # For MVP, use non-streaming
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": max_tokens,
                        **self.current_model.parameters
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            # Update metrics
            self.metrics.inference_count += 1
            self.metrics.tokens_processed += result.get('eval_count', 0)
            
            logger.info(f"Generation successful. Tokens: {result.get('eval_count', 0)}")
            return generated_text.strip()
        
        except requests.exceptions.Timeout:
            raise ModelInferenceError("Generation timed out. Model may be slow or overloaded.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Generation failed: {str(e)}")
            raise ModelInferenceError(f"Failed to generate text: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}")
            raise ModelInferenceError(f"Unexpected error: {str(e)}")

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate text using chat format (for multi-turn conversations).
        
        Args:
            messages: List of message dicts with 'role' ('user'/'assistant') and 'content'
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            str: Generated response
            
        Raises:
            ModelInferenceError: If generation fails
        """
        if not self.current_model:
            raise ModelConnectionError("No model loaded. Call load_model() first.")
        
        # Convert messages to prompt format
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user').upper()
            content = msg.get('content', '')
            prompt_parts.append(f"{role}: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nASSISTANT:"
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

    # ========================
    # Utility Methods
    # ========================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model performance and status metrics.
        
        Returns:
            Dict: Metrics including status, inference count, tokens processed
        """
        return {
            'status': self.metrics.status.value,
            'last_check': self.metrics.last_check,
            'response_time_ms': self.metrics.response_time_ms,
            'error_message': self.metrics.error_message,
            'tokens_processed': self.metrics.tokens_processed,
            'inference_count': self.metrics.inference_count,
            'current_model': self.current_model.model_id if self.current_model else None
        }

    def get_setup_instructions(self) -> str:
        """
        Get instructions for setting up Ollama and recommended models.
        
        Returns:
            str: Installation and setup instructions
        """
        return """
## Setting up Local LLM with Ollama

### Step 1: Install Ollama
- Download from https://ollama.ai
- Follow installation instructions for your OS

### Step 2: Start Ollama Service
Linux/Mac:
```bash
ollama serve
```

Windows:
- Ollama runs as a background service automatically after installation

### Step 3: Pull Recommended Model
```bash
# Choose one:
ollama pull mistral:7b          # Recommended (fastest)
ollama pull llama2:7b            # Meta's Llama 2
ollama pull neural-chat:7b       # Optimized for chat
```

### Step 4: Verify Installation
```bash
curl http://localhost:11434/api/tags
```

Recommended models for proposals:
- **llama3.1:8b** (9GB, best quality - PRIMARY)
- **mistral:7b-instruct** (8GB, fastest)
- **deepseek-r1:8b** (9GB, best reasoning - extraction)
"""

    def estimate_resource_requirements(self, model_id: str) -> Dict[str, str]:
        """
        Estimate resource requirements for a model.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Dict: Estimated RAM, VRAM, disk space
        """
        # Approximate sizes for common models
        size_map = {
            "7b": {"ram": "16GB", "vram": "8GB (optional)", "disk": "15GB"},
            "13b": {"ram": "24GB", "vram": "12GB", "disk": "28GB"},
        }
        
        # Try to match model size
        for size_key, requirements in size_map.items():
            if size_key in model_id:
                return requirements
        
        return {"ram": "16GB+", "vram": "8GB+", "disk": "15GB+"}


# Global model manager instance (lazy-loaded)
_model_manager: Optional[LocalLMManager] = None


def get_model_manager() -> LocalLMManager:
    """
    Get or create the global model manager instance.
    
    Returns:
        LocalLMManager: Global model manager
    """
    global _model_manager
    if _model_manager is None:
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        default_model = os.getenv('DEFAULT_LLM_MODEL', 'mistral:7b')
        _model_manager = LocalLMManager(
            ollama_base_url=ollama_url,
            default_model=default_model
        )
    return _model_manager
