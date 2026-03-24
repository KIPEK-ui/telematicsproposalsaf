"""AI Models module - Ollama model registry"""

# Model registry and paths
MODELS = {
    'llama3.1:8b': {
        'name': 'Llama 3.1 8B',
        'provider': 'ollama',
        'description': 'Best quality - excellent for proposal generation',
        'vram': '8GB',
        'context_length': 8192,
        'type': 'generation',
    },
    'mistral:7b-instruct': {
        'name': 'Mistral 7B Instruct',
        'provider': 'ollama',
        'description': 'Instruction-tuned - fastest, good quality',
        'vram': '8GB',
        'context_length': 8192,
        'type': 'generation',
    },
    'deepseek-r1:8b': {
        'name': 'DeepSeek R1 8B',
        'provider': 'ollama',
        'description': 'Best reasoning - ideal for requirement extraction',
        'vram': '8GB',
        'context_length': 4096,
        'type': 'reasoning',
    },
    'mistral:7b': {
        'name': 'Mistral 7B (Base)',
        'provider': 'ollama',
        'description': 'Fallback option - fast baseline',
        'vram': '8GB',
        'context_length': 8192,
        'type': 'generation',
    },
}

# Default models for different tasks
DEFAULT_MODEL = 'llama3.1:8b'
EXTRACTION_MODEL = 'deepseek-r1:8b'  # Better reasoning for requirements
GENERATION_MODEL = 'llama3.1:8b'     # Best quality proposals
