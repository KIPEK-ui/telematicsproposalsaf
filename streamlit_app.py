"""
Telematics IQ - AI-Assisted Proposal Generator for Safaricom
Production MVP - Proposal Generation with Local LLM
"""

import streamlit as st
import os
import importlib.util
import sys
import traceback

# Disable PaddleOCR model source check for faster startup
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from dotenv import load_dotenv
from app.services.local_db_service import get_db_service

# Load environment variables
load_dotenv()

# --- PAGE SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- UI Configuration ---
st.set_page_config(
    page_title="Telematics IQ - Proposal Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { max-width: 1400px; margin: 0 auto; }
    .stTitle { color: #FF6300; }
    .sidebar .sidebar-content { background: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR BRANDING ---
logo_path = os.path.join(current_dir, 'assets', 'logo.png')
if os.path.exists(logo_path):
    # show small logo in the sidebar if available
    st.sidebar.image(logo_path, width="content")  # Updated to use new width parameter

st.sidebar.markdown("# 📄✨ Telematics IQ")
st.sidebar.markdown("**Proposal Generator MVP**")
st.sidebar.markdown("AI-Powered proposal creation for tenders")
st.sidebar.markdown("---")

# Initialize local database service
@st.cache_resource
def init_db_service():
    """Initialize and cache database service"""
    try:
        return get_db_service()
    except Exception as e:
        st.error(f"Failed to initialize database service: {str(e)}")
        st.stop()

db_service = init_db_service()

# --- HELP & INFO ---
with st.sidebar.expander("ℹ️ About"):
    st.markdown("""
    **Telematics IQ Proposal Generator** is an MVP that helps sales teams create 
    professional proposals from tenders automatically.
    
    **Features:**
    - 📥 Accept tenders (PDF, text, or form)
    - 🤖 Extract requirements with AI
    - ✍️ Generate proposals with LLM
    - 📝 Refine sections interactively
    - 📄 Export as Word (.docx)
    
    **Requirements:**
    - Ollama running locally (http://localhost:11434)
    - Model loaded (llama3.1:8b, mistral:7b-instruct, or deepseek-r1:8b)
    """)

with st.sidebar.expander("🛠️ Model Status"):
    try:
        from app.ai_service.model_manager import get_model_manager
        manager = get_model_manager()
        
        if manager.check_ollama_availability():
            st.success("✅ Ollama: Available")
            models = manager.list_available_models()
            if models:
                st.success(f"✅ Models loaded: {len(models)}")
                for model in models[:3]:
                    st.caption(f"  • {model}")
            else:
                st.warning("⚠️ No models loaded. Run: ollama pull llama3.1:8b")
        else:
            st.error("❌ Ollama: Not running at http://localhost:11434")
            st.caption("Start Ollama with: `ollama serve`")
    except Exception as e:
        st.error(f"Model check failed: {str(e)}")

with st.sidebar.expander("📖 Quick Start"):
    st.markdown("""
    1. **Upload Tender** - PDF, text, or form
    2. **Enter Organization** - Your company info
    3. **Extract Requirements** - AI extracts key info
    4. **Generate Proposal** - Creates 7-section proposal
    5. **Refine & Export** - Edit & download .docx
    """)

st.sidebar.markdown("---")

# Navigation menu
app_mode = st.sidebar.radio(
    "📋 Select Mode",
    ["🚀 Proposal Generator", "📚 Document Manager"],
    help="Switch between proposal generation and document management"
)

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ for Safaricom")

# --- MAIN APP ---
# Route to appropriate view
if app_mode == "📚 Document Manager":
    # Document Manager View
    module_rel_path = "app/views/document_manager_view.py"
    module_path = os.path.join(current_dir, module_rel_path)
    
    # Page header
    st.title("📚 Document Manager")
    st.markdown("Upload and manage training documents for RAG system")
    st.markdown("---")
else:
    # Proposal Generator View
    module_rel_path = "app/views/proposal_generator.py"
    module_path = os.path.join(current_dir, module_rel_path)
    
    # Page header
    st.title("📄✨ Proposal Generator")
    st.markdown("Generate professional proposals from tenders using AI")
    st.markdown("---")
    
    # Performance tips
    with st.expander("⚡ **PERFORMANCE TIPS** - Make Steps 1-4 FASTER ⚡", expanded=False):
        st.markdown("""
        ### Why is it slow?
        - **LLM Inference**: AI model generation takes time (60-180s depending on model)
        - **Model Size**: Larger models (8B parameters) = slower but better quality
        - **Hardware**: GPU/RAM affects speed significantly
        
        ### Speed Up Steps 1-4:
        
        **✅ RECOMMENDED (Fastest):**
        - ⚡ **Use "Fast Mode" in Step 3** - Skips AI, uses pattern-based extraction (2-5 seconds)
        - 🚀 Use **Mistral 7B** model instead of DeepSeek (3-4x faster)
          - Install: `ollama pull mistral:7b`
          - Or use: `ollama pull mistral:7b-q5_K_M` (quantized, even faster)
        
        **🟡 MODERATE SPEED:**
        - Use "Detailed Mode" if you need AI analysis (60-120 seconds per step)
        - Use **DeepSeek R1 8B** for better reasoning (slower: 120-180 seconds)
        
        **🔧 HARDWARE OPTIMIZATION:**
        - Add GPU support (NVIDIA/AMD) in Ollama: https://ollama.ai
        - Increase RAM (minimum 8GB, better with 16GB+)
        - Close other applications while generating
        
        **💡 WORKFLOW TIPS:**
        1. Start with **Fast Mode** (Step 3) for quick results
        2. Review auto-extracted requirements
        3. Switch to **Detailed Mode** only if you need AI refinement
        4. Use pattern-based fallback (happens automatically if timeout occurs)
        
        **📊 EXPECTED TIMES:**
        - Fast Mode (Step 3): 2-5s ⚡
        - Mistral 7B (Step 3/4): 30-60s each 🟢
        - DeepSeek R1 8B (Step 3/4): 120-180s each 🟡
        - Local GPU: 5-15s per step ✨
        """)
    
    st.markdown("---")

# Load and execute the proposal generator view
if not os.path.exists(module_path):
    st.error(f"""
    ⚠️ **Module not found**: {module_path}
    
    Make sure you have the complete project structure:
    ```
    app/views/proposal_generator.py
    ```
    """)
else:
    try:
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        
        # Ensure the project directory is on sys.path
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        spec.loader.exec_module(module)

        if hasattr(module, "render"):
            module.render()
        else:
            st.warning("⚠️ Module loaded but no render() function found")
    except Exception as e:
        st.error("❌ Error loading proposal generator")
        st.error(str(e))
        with st.expander("📋 Details"):
            st.text(traceback.format_exc())




    