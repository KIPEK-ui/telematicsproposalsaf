# Telematics IQ - AI-Assisted Proposal Generator for Telematic Solutions

A production-ready MVP proposal generator that accepts tenders in multiple formats (PDF, text, form), automatically extracts technical requirements using local LLMs, generates professional proposals with RAG-based context, and exports them as branded Word documents.

## Features

✅ **3-Format Tender Input**
- PDF upload with automatic text extraction
- Plain text paste
- Structured form entry

✅ **Intelligent Requirement Extraction**
- LLM-powered extraction of fleet requirements, technical specs, timeline, budget, compliance
- Fallback pattern-based extraction if LLM unavailable
- Structured JSON output with validation

✅ **AI-Powered Proposal Generation**
- 7-section professional proposals
- RAG (Retrieval-Augmented Generation) context from past proposals
- LLM-driven content generation using local Ollama
- Optimized models: Llama 3.1 8B (primary), Mistral 7B Instruct (fast), DeepSeek R1 8B (reasoning)

✅ **Iterative Refinement**
- Per-section proposal refinement
- Chat-based interaction with the AI
- Keep other sections intact during edits

✅ **Professional Export**
- Word (.docx) format with Safaricom branding
- Consistent formatting, headers, footers
- Organization info box
- Ready for client delivery

✅ **Fully Local & Private**
- No cloud dependencies beyond Ollama
- Session-scoped SQLite database
- All processing on your machine
- No data sent to external services

## Quick Start

### 1. Prerequisites

- Python 3.9+ (tested on 3.13.12)
- [Ollama](https://ollama.ai) installed and running
- 16GB+ RAM (recommended for 7B models)
- ~10GB disk space (for models)

### 2. Installation

```bash
# Clone/navigate to project directory
cd telematicsproposalsaf

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup

```bash
# Copy environment template
cp .env.example .env

# (Optional) Edit .env with custom settings
# nano .env OR code .env

# Create data directories (auto-created, shown in app on first run)
mkdir -p data/training_proposals
```

### 4. Prepare LLM

```bash
# In a separate terminal, start Ollama
ollama serve

# In another terminal, pull a model (choose one or all):
ollama pull llama3.1:8b         # Recommended: best quality
ollama pull mistral:7b-instruct # Alternative: fastest
ollama pull deepseek-r1:8b      # Optional: best reasoning for extraction

# Verify it's available:
ollama list
```

### 5. Run the App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## ✨ NEW: Document Manager (RAG Training)

The **Document Manager** allows you to upload and manage training documents to improve proposal quality through RAG (Retrieval-Augmented Generation).

### Document Manager Features

📤 **Upload Documents** - Drag-and-drop PDF, DOCX, or TXT files
🔍 **Automatic Duplicate Detection** - Prevents storing identical files
📋 **View & Manage** - See all uploaded documents with metadata
⚙️ **Process for RAG** - Extract text and build embeddings
📊 **Statistics Dashboard** - Monitor upload and processing status

### Quick Start with Document Manager

1. **Switch to Document Manager** (sidebar: 📋 Select Mode → 📚 Document Manager)
2. **Upload Documents** (Your past proposals, requirements, examples)
3. **Process for RAG** (Click button, system handles the rest)
4. **Generate Proposals** (Switch back to Proposal Generator)
5. **Better Results** (Proposals use your documents for better context)

**For detailed instructions, see:** [`DOCUMENT_MANAGER_GUIDE.md`](DOCUMENT_MANAGER_GUIDE.md)

## Usage Workflow

### Step 1: (Optional) Prepare Training Data
- Use **Document Manager** to upload past proposals
- System extracts text and builds RAG index
- Improves quality of generated proposals
- See `DOCUMENT_MANAGER_GUIDE.md` for details

### Step 2: Input Tender
- Upload a **PDF** tender document, OR
- Paste **plain text** from tender, OR
- Enter details via **structured form**

### Step 2: Organization Info
- Enter your organization name
- Select industry type
- Provide contact email

### Step 3: Extract Requirements
- System automatically extracts:
  - Fleet requirements
  - Technical specifications
  - Scope & deliverables
  - Timeline & milestones
  - Budget constraints
  - Compliance requirements
  - Evaluation criteria
- Review and edit if needed

### Step 4: Generate Proposal
- Click "Generate Proposal" to create all 7 sections:
  1. Executive Summary
  2. Technical Approach
  3. Fleet Details
  4. Implementation Timeline
  5. Pricing & Commercial
  6. Compliance Assurance
  7. Terms & Conditions

### Step 5: Refine & Export
- Review each section in tabs
- Refine individual sections (chat-based)
- Download as `.docx` file

## Project Structure

```
telematicsproposalsaf/
├── streamlit_app.py                 # Main Streamlit entry point
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
│
├── app/                             # Main application package
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── local_db_service.py      # SQLite database management
│   │   ├── tender_parser.py         # Tender document parsing (PDF/Text/Form)
│   │   └── document_exporter.py     # Word document generation with branding
│   │
│   ├── ai_service/
│   │   ├── __init__.py
│   │   ├── model_manager.py         # Ollama LLM lifecycle management
│   │   ├── rag_service.py           # Semantic search over past proposals
│   │   ├── requirement_extractor.py # LLM-based requirement extraction
│   │   └── proposal_generator.py    # Core proposal generation engine
│   │
│   ├── ai_models/
│   │   └── __init__.py              # Model registry and config
│   │
│   └── views/
│       ├── __init__.py
│       └── proposal_generator.py    # Main Streamlit UI (5-step workflow)
│
└── data/
    ├── training_proposals/          # Sample past proposals for RAG context
    └── proposals.db                 # SQLite database (auto-created)
```

## Configuration

Edit `.env` to customize:

```bash
# LLM Settings - Optimized Model Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Default model (used when no specific model requested)
DEFAULT_LLM_MODEL=llama3.1:8b

# Model for extraction (superior reasoning capabilities)
EXTRACTION_LLM_MODEL=deepseek-r1:8b

# Model for generation (best quality for proposals)
GENERATION_LLM_MODEL=llama3.1:8b

# Database
DATABASE_PATH=data/proposals.db

# Generation
GENERATION_TEMPERATURE=0.3           # Lower = more consistent
MAX_TOKENS_PER_SECTION=1000

# RAG
RAG_TOP_K=2                          # Retrieve top 2 similar proposals

# Logging
LOG_LEVEL=INFO
```

## Adding Training Data

To improve proposal quality with past examples:

1. Add sample proposals to `data/training_proposals/` as `.txt` files
2. Follow naming: `{industry}_{type}_{year}.txt`
3. Include all sections (Executive Summary, Technical Approach, etc.)
4. Restart the app

See [data/training_proposals/README.md](data/training_proposals/README.md) for detailed instructions.

## Troubleshooting

### "Connection refused" Error
```
Error: Failed to connect to Ollama at http://localhost:11434
```
**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Or check if it's running on different port:
# Edit .env: OLLAMA_BASE_URL=http://localhost:11434
```

### "Model not found" Error
```
Error: Model 'llama3.1:8b' not found
```
**Solution:**
```bash
# Pull the model
ollama pull llama3.1:8b

# Or use a faster alternative:
ollama pull mistral:7b-instruct

# List available models
ollama list
```

### Slow Generation
**Solution:**
- Use `mistral:7b-instruct` instead of `llama3.1:8b` (faster but slightly lower quality)
- Close other applications
- Ensure GPU support (if available)
- Reduce `MAX_TOKENS_PER_SECTION` in .env

### Streamlit App Crashes
**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check logs
tail -f .streamlit/logs/*.log

# Restart app
streamlit run streamlit_app.py --logger.level=debug
```

## Performance

| Model | Speed | Quality | VRAM | Best Use |
|-------|-------|---------|------|----------|
| **Llama 3.1 8B** | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ Excellent | 8GB | **PRIMARY** - Proposals |
| **Mistral 7B Instruct** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ Very Good | 8GB | Fast iteration |
| **DeepSeek R1 8B** | ⚡⚡ | ⭐⭐⭐⭐⭐ Excellent | 8GB | Requirement extraction |
| Mistral 7B (Base) | ⚡⚡⚡⚡ | ⭐⭐⭐ Good | 8GB | Fallback option |

**Generation time (per proposal):**
- Llama 3.1 8B: ~2-3 minutes
- Mistral 7B Instruct: ~1.5-2 minutes
- DeepSeek R1 8B: ~3-4 minutes (better reasoning)

## Architecture

### Components

**Database Layer** (`local_db_service.py`)
- Session-scoped SQLite
- CRUD for tenders & proposals
- Transaction management with rollback
- Connection pooling and validation

**Parser Layer** (`tender_parser.py`)
- 3-format input: PDF (pdfplumber), Text (regex), Form (structured)
- Factory pattern for automatic dispatch
- Extracts: overview, requirements, timeline, budget, evaluation criteria
- Fallback pattern matching on extraction failure

**AI/ML Layer**
- **Model Manager** (`model_manager.py`): Ollama integration with health checks
- **RAG Service** (`rag_service.py`): Semantic search using SentenceTransformers
- **Extractor** (`requirement_extractor.py`): LLM-based + pattern fallback
- **Generator** (`proposal_generator.py`): Multi-section generation with context

**Export Layer** (`document_exporter.py`)
- Python-docx for .docx generation
- Safaricom branding (orange/gray colors)
- Professional formatting with headers, footers
- Confidentiality notices

**UI Layer** (`views/proposal_generator.py`)
- 5-step interactive workflow
- Streamlit session state management
- Real-time AI model status
- Per-section refinement with chat

### Design Patterns

- **Singleton**: Service managers (DB, LLM, RAG) initialized once
- **Factory**: Parser selection based on input type
- **Abstract Base Class**: Parser interface for extensibility
- **Context Manager**: Database connection lifecycle
- **Dataclass**: Type-safe data models with serialization

## Documentation & Guides

### 📚 Available Guides

- **[DOCUMENT_MANAGER_GUIDE.md](DOCUMENT_MANAGER_GUIDE.md)** - Complete guide to uploading and managing training documents
  - Upload documents (PDF, DOCX, TXT)
  - Duplicate detection explained
  - Processing workflow
  - RAG integration
  - Statistics dashboard
  - Troubleshooting

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Technical details on document processing and RAG
  - Document extraction methods
  - RAG configuration
  - Best practices

- **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** - Summary of Document Manager integration
  - What's new
  - How it works
  - Testing instructions

## Limitations

- **MVP Scope**: Single-user, local-only (no multi-user/cloud)
- **Model Size**: 7B models require ~8GB VRAM (no quantized versions yet)
- **PDF Complexity**: Works with text PDFs; scanned/image PDFs require OCR
- **Long Documents**: Best for tenders < 50MB
- **Language**: English only (can extend to other languages)

## Future Enhancements

- [ ] Support for quantized models (Q4, Q5) for faster inference
- [ ] GPU optimization (cuBLAS, Metal support)
- [ ] Web UI deployment (FastAPI + React)
- [ ] Multi-user with cloud backend (Supabase)
- [ ] Proposal templates and customization
- [ ] Batch generation for multiple tenders
- [ ] OCR support for scanned PDFs
- [ ] Multilingual support
- [ ] Integration with email/CRM systems

## Development

### Running Tests
```bash
pytest tests/ -v --cov=app
```

### Code Quality
```bash
black app/                    # Format code
pylint app/                   # Lint
flake8 app/ --max-line-length=100  # Style check
```

### Debugging
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Run with debug output
streamlit run streamlit_app.py --logger.level=debug
```

## License

Proprietary - Telematics IQ (Safaricom)

## Support

For issues, questions, or feature requests:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs: `.streamlit/logs/`
3. Check Ollama status: `http://localhost:11434/api/tags`

## Contributors

- AI/ML Services: Advanced LLM integration and RAG
- Document Processing: Multi-format parsing and export
- UI/UX: Streamlit workflow design

---

**Version**: 1.0.0-MVP  
**Last Updated**: 2024  
**Status**: Production-Ready MVP ✅
