"""
Proposal Generator View
Main Streamlit interface for AI-assisted proposal generation.
Multi-step workflow: Tender Input → Organization Info → Extract Requirements → Generate → Refine → Export
"""

import os
# Disable PaddleOCR model source check for faster startup
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import streamlit as st
import logging
from typing import Optional, Dict, Any
import io
import hashlib

from app.services.local_db_service import get_db_service
from app.services.tender_parser import TenderParserFactory, TenderParsingError, TenderDocument
from app.services.temp_file_manager import TemporaryFileManager
from app.ai_service.model_manager import get_model_manager, ModelConnectionError
from app.ai_service.requirement_extractor import get_requirement_extractor, RequirementExtractionError, StructuredRequirements
from app.ai_service.dynamic_proposal_designer import TenderClassifier, DynamicProposalDesigner, TenderProfile
from app.ai_service.enhanced_proposal_generator import EnhancedProposalGenerator
from app.services.document_exporter import get_document_exporter, DocumentExportError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Proposal Generator",
    page_icon="📄✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Session-Level Caching (NEW)
# ========================

@st.cache_resource
def get_rag_service_cached():
    """Cache RAG service initialization."""
    from app.ai_service.rag_service import get_rag_service
    return get_rag_service()

# ========================
# Session State Initialization
# ========================

def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'step': 1,
        'tender_doc': None,
        'tender_id': None,
        'tender_file_path': None,  # Store original tender file path for form filling
        'tender_file_bytes': None,  # Store original tender file bytes for form filling
        'org_data': None,
        'requirements': None,
        'proposal_content': None,
        'proposal_id': None,
        'chat_history': [],
        'model_initialized': False,
        'tender_profile': None,  # Classification of tender
        'proposal_structure': None,  # Designed section structure
        'use_dynamic_proposal': True,  # Flag to use dynamic generator
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ========================
# Header & Navigation
# ========================

def render_header():
    """Render page header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📄✨ AI Proposal Generator")
    
    st.markdown("---")
    
    # Progress indicator
    step_labels = {
        1: "1️⃣ Tender Input",
        2: "2️⃣ Organization",
        3: "3️⃣ Structure Design",
        4: "4️⃣ Requirements",
        5: "5️⃣ Generate",
        6: "6️⃣ Refine & Export"
    }
    
    progress = st.progress(min(st.session_state.step / 6.0, 1.0))
    current_step_label = step_labels.get(st.session_state.step, "Unknown Step")
    st.caption(f"Step {st.session_state.step}: {current_step_label}")


# ========================
# Step 1: Tender Input
# ========================

def render_step_1_tender_input():
    """Render tender input step with 3 input methods."""
    st.subheader("Step 1: Upload Tender Document")
    st.write("Choose how you'd like to provide the tender information:")
    
    input_method = st.radio(
        "Input Method",
        ["📄 Upload PDF", "📝 Paste Text/Email", "📋 Use Form"],
        horizontal=True
    )
    
    tender_doc = None
    tender_title = None
    
    try:
        if input_method == "📄 Upload PDF":
            uploaded_file = st.file_uploader(
                "Upload tender document (PDF, DOCX, or TXT)", 
                type=['pdf', 'docx', 'doc', 'txt'],
                help="Supported formats: PDF, DOCX, DOC, TXT"
            )
            if uploaded_file:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                st.info(f"📤 Processing {file_ext.upper()} file...")
                file_bytes = uploaded_file.read()
                try:
                    tender_doc = TenderParserFactory.parse_file(file_bytes, uploaded_file.name)
                    tender_title = uploaded_file.name
                    
                    # Create temporary file for form filling operations
                    temp_file_path = TemporaryFileManager.create_temp_file(
                        file_bytes, 
                        uploaded_file.name
                    )
                    st.session_state.tender_file_bytes = file_bytes
                    st.session_state.tender_file_path = temp_file_path
                    
                    st.success(f"✅ {file_ext.upper()} parsed successfully: {tender_doc.title}")
                except Exception as e:
                    st.error(f"❌ Error parsing {file_ext.upper()}: {str(e)}")
        
        elif input_method == "📝 Paste Text/Email":
            st.write("Paste tender content from email or document:")
            text_content = st.text_area(
                "Tender content",
                height=300,
                placeholder="Paste tender text here...",
                key="tender_text_input"
            )
            tender_title_input = st.text_area("Tender Title (optional)", height=80, placeholder="Enter tender title")
            
            if text_content.strip():
                tender_doc = TenderParserFactory.parse_text(
                    text_content,
                    title=tender_title_input or None
                )
                tender_title = tender_title_input or tender_doc.title
                st.success(f"✅ Text parsed successfully")
        
        else:  # Form input
            st.write("Fill in the tender details:")
            
            form_data = {}
            
            # Tender title field (multi-line to capture full titles)
            form_data['title'] = st.text_area("Tender Title *", height=80, placeholder="e.g., Vehicle Fleet Management Services for 3 years")
            
            # Two-column layout for additional fields (all multi-line to capture full content)
            col1, col2 = st.columns(2)
            
            with col1:
                form_data['tender_no'] = st.text_area("Tender No.", height=60, placeholder="Enter tender reference number")
                form_data['email'] = st.text_area("Email", height=60, placeholder="Enter contact email(s)")
                form_data['timeline'] = st.text_area("Timeline/Deadline", height=60, placeholder="e.g., Closing date: Jan 15, 2025 at 11:00 am")
            
            with col2:
                form_data['address'] = st.text_area("Address", height=60, placeholder="Enter tender issuer address")
                form_data['bid_validity'] = st.text_area("Bid Validity", height=60, placeholder="e.g., 90 days from submission date")
                form_data['budget'] = st.text_area("Budget/Pricing Range", height=60, placeholder="e.g., KES 5,000,000 - 7,500,000")
            
            form_data['description'] = st.text_area("Description/Overview", height=100)
            form_data['requirements'] = st.text_area("Technical Requirements", height=100)
            form_data['fleet_details'] = st.text_area("Fleet & Equipment Details", height=100)
            
            if form_data['title']:
                tender_doc = TenderParserFactory.parse_form(form_data)
                tender_title = form_data['title']
                st.success("✅ Form data processed")
        
        # Display tender summary
        if tender_doc:
            with st.expander("📋 Tender Summary", expanded=True):
                # Full-width single column to allow multi-line text wrapping
                st.write(f"**Title:** {tender_doc.title}")
                st.write(f"**Source:** {tender_doc.source_type.value}")
                if tender_doc.tender_no:
                    st.write(f"**Tender No.:** {tender_doc.tender_no}")
                if tender_doc.address:
                    st.write(f"**Address:**")
                    st.markdown(tender_doc.address.replace('\n', '  \n'))  # Preserve line breaks in markdown
                if tender_doc.email:
                    st.write(f"**Email:** {tender_doc.email}")
                if tender_doc.phone_number:
                    st.write(f"**Phone:** {tender_doc.phone_number}")
                if tender_doc.bid_validity:
                    st.write(f"**Bid Validity:** {tender_doc.bid_validity}")
                if tender_doc.timeline:
                    st.write(f"**Timeline:** {tender_doc.timeline}")
                if tender_doc.budget_info:
                    st.write(f"**Budget:** {tender_doc.budget_info}")
            
            # Save to database and proceed
            if st.session_state.tender_doc is None:
                db_service = get_db_service()
                try:
                    tender_id = db_service.insert_tender(
                        tender_title=tender_title or tender_doc.title,
                        tender_source=tender_doc.source_type.value,
                        raw_content=tender_doc.raw_content,
                        parsed_content=tender_doc.to_dict(),
                        technical_requirements=tender_doc.technical_requirements
                    )
                    st.session_state.tender_doc = tender_doc
                    st.session_state.tender_id = tender_id
                except Exception as e:
                    st.error(f"❌ Failed to save tender: {str(e)}")
            
            # Navigation buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⬅️ Back", use_container_width=True):
                    st.session_state.step = max(1, st.session_state.step - 1)
                    st.rerun()
            with col2:
                if st.button("Next: Organization Info ➡️", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()
    
    except TenderParsingError as e:
        st.error(f"❌ Parsing Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"Step 1 error: {str(e)}")


# ========================
# Step 2: Organization Info
# ========================

def render_step_2_organization_info():
    """Render organization data capture."""
    if st.session_state.tender_doc is None:
        st.warning("⚠️ Please complete Step 1 first")
        return
    
    st.subheader("Step 2: Organization Information")
    st.write("Please provide Safaricom's details for this proposal:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        org_name = st.text_input(
            "Organization Full Name *",
            value=st.session_state.org_data.get('name', 'Safaricom') if st.session_state.org_data else 'Safaricom'
        )
        
        industry_options = [
            "Telecommunications",
            "Transportation",
            "Infrastructure",
            "Energy",
            "Healthcare",
            "Financial Services",
            "Retail",
            "Other"
        ]
        industry = st.selectbox(
            "Industry *",
            industry_options,
            index=0 if not st.session_state.org_data else industry_options.index(
                st.session_state.org_data.get('industry', 'Telecommunications')
            )
        )
    
    with col2:
        contact_email = st.text_area(
            "Primary Contact Email(s) *",
            height=80,
            placeholder="Enter email(s), one per line",
            value=st.session_state.org_data.get('contact_email', '') if st.session_state.org_data else ''
        )
        contact_phone = st.text_input(
            "Primary Contact Phone *",
            value=st.session_state.org_data.get('contact_phone', '') if st.session_state.org_data else ''
        )
    
    address = st.text_area(
        "Address",
        height=80,
        placeholder="Enter complete mailing address",
        value=st.session_state.org_data.get('address', '') if st.session_state.org_data else ''
    )
    
    # Validation and save
    if org_name and contact_email and contact_phone:
        org_data = {
            'name': org_name,
            'industry': industry,
            'contact_email': contact_email,
            'contact_phone': contact_phone,
            'address': address
        }
        st.session_state.org_data = org_data
        st.success("✅ Organization data saved")
    else:
        st.warning("⚠️ Please fill in all required fields (marked with *)")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, key="back_2"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next: Design Proposal Structure ➡️", use_container_width=True, key="next_2"):
            if org_name and contact_email and contact_phone:
                st.session_state.step = 3
                st.rerun()
            else:
                st.error("❌ Please fill in all required fields")


# ========================
# Step 3: Design Proposal Structure
# ========================

def render_step_3_structure_design():
    """Render proposal structure design step using dynamic designer."""
    if st.session_state.tender_doc is None or st.session_state.org_data is None:
        st.warning("⚠️ Please complete previous steps first")
        return
    
    st.subheader("Step 3: Design Proposal Structure (Dynamic)")
    st.write("AI will analyze your tender and design an optimal proposal structure...")
    
    if st.session_state.tender_profile is None or st.session_state.proposal_structure is None:
        # Progress tracking containers
        col1, col2 = st.columns([2, 1])
        with col1:
            status_text = st.empty()
        with col2:
            progress_placeholder = st.empty()
        
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # Define progress callback
            def update_structure_progress(current: int, total: int, step_name: str):
                """Update progress for structure design."""
                progress_pct = current / total
                display_name = step_name.replace('_', ' ').title()
                
                progress_bar.progress(
                    progress_pct,
                    text=f"Designing proposal... {display_name}..."
                )
                with status_text.container():
                    st.caption(f"Step: {current}/{total} | {display_name}")
                with progress_placeholder.container():
                    st.metric("Progress", f"{current}/{total}")
            
            # Step 1: Classify tender
            classifier = TenderClassifier()
            tender_profile = classifier.classify(st.session_state.tender_doc, progress_callback=update_structure_progress)
            st.session_state.tender_profile = tender_profile
            
            # Step 2: Design proposal structure
            designer = DynamicProposalDesigner()
            
            proposal_structure = designer.design_structure(
                tender_doc=st.session_state.tender_doc,
                tender_profile=tender_profile,
                progress_callback=update_structure_progress
            )
            st.session_state.proposal_structure = proposal_structure
            
            progress_bar.progress(1.0, text="✅ Structure designed successfully!")
            with status_text.container():
                st.success("✅ Tender analysis and structure design complete!")
        
        except Exception as e:
            st.error(f"❌ Design Error: {str(e)}")
            logger.error(f"Structure design error: {str(e)}")
            st.stop()
    
    # Display tender classification
    with st.expander("🎯 Tender Analysis", expanded=True):
        profile = st.session_state.tender_profile
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tender Type", profile.tender_type.replace('_', ' ').title())
        with col2:
            st.metric("Industry", profile.industry.replace('_', ' ').title())
        with col3:
            st.metric("Complexity", profile.complexity.title())
        
        st.write("**Key Themes:**", ", ".join(profile.key_themes) if profile.key_themes else "General")
        st.write("**Buyer Priorities:**", ", ".join(profile.priority_areas) if profile.priority_areas else "Standard")
    
    # Display proposed structure
    with st.expander("📋 Proposed Proposal Structure", expanded=True):
        structure = st.session_state.proposal_structure
        
        st.info(f"**Design Rationale:** {structure.design_rationale}")
        
        st.write(f"**Total Sections:** {len(structure.section_definitions)}")
        st.write(f"**Estimated Length:** {structure.estimated_word_count_min} - {structure.estimated_word_count_max} words")
        
        st.write("**Section Order & Emphasis:**")
        
        # Create a table of sections
        sections_data = []
        for i, section_name in enumerate(structure.section_order, 1):
            section_def = structure.section_definitions[section_name]
            sections_data.append({
                "Order": i,
                "Section": section_def.title,
                "Importance": section_def.importance,
                "Length": f"{section_def.target_length} words",
                "Key Points": ", ".join(section_def.key_points[:2]) if section_def.key_points else "—"
            })
        
        st.dataframe(sections_data, use_container_width=True)
    
    # Allow customization
    with st.expander("✏️ Customize Structure", expanded=False):
        st.write("Modify the proposed structure if desired:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Reorder Sections:**")
            structure = st.session_state.proposal_structure
            new_order = st.multiselect(
                "Section Order",
                structure.section_order,
                default=structure.section_order,
                help="Drag to reorder sections by importance"
            )
            
            if new_order and new_order != structure.section_order:
                structure.section_order = new_order
                st.session_state.proposal_structure = structure
                st.success("✅ Section order updated")
        
        with col2:
            st.write("**Adjust Section Emphasis:**")
            structure = st.session_state.proposal_structure
            
            selected_section = st.selectbox(
                "Select section to adjust",
                structure.section_order
            )
            
            if selected_section:
                section_def = structure.section_definitions[selected_section]
                new_importance = st.select_slider(
                    f"Importance of {section_def.title}",
                    options=["low", "medium", "high", "critical"],
                    value=section_def.importance,
                    help="Determines section emphasis and length"
                )
                
                if new_importance != section_def.importance:
                    section_def.importance = new_importance
                    # Recalculate target length based on importance
                    importance_multiplier = {
                        "low": 0.5,
                        "medium": 1.0,
                        "high": 1.5,
                        "critical": 2.0
                    }
                    section_def.target_length = int(400 * importance_multiplier[new_importance])
                    st.success(f"✅ {section_def.title} importance updated")
    
    st.success("✅ Proposal structure designed and ready!")
    st.info("ℹ️ Click 'Next' to extract requirements and generate the proposal with this custom structure.")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, key="back_3_structure"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Next: Extract Requirements ➡️", use_container_width=True, key="next_3_structure"):
            st.session_state.step = 4
            st.rerun()


# ========================
# Step 3: Requirements Extraction
# ========================

def render_step_3_extract_requirements():
    """Render requirement extraction step."""
    if st.session_state.tender_doc is None or st.session_state.org_data is None:
        st.warning("⚠️ Please complete previous steps first")
        return
    
    st.subheader("Step 3: Extract & Review Requirements")
    
    # Add Fast Mode option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Extraction Mode:**")
    with col2:
        use_fast_mode = st.checkbox(
            "⚡ Fast Mode",
            value=False,
            help="Skip LLM (use pattern-based extraction) for faster processing"
        )
    
    if st.session_state.requirements is None:
        # Extract requirements
        if use_fast_mode:
            st.info("⚡ Using fast pattern-based extraction...")
        else:
            st.info("🔄 Using LLM-based extraction (slower but more accurate)...")
        
        # Progress tracking containers
        col1, col2 = st.columns([2, 1])
        with col1:
            status_text = st.empty()
        with col2:
            progress_placeholder = st.empty()
        
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # Initialize model only if NOT using fast mode
            if not use_fast_mode:
                if not st.session_state.model_initialized:
                    model_manager = get_model_manager()
                    if not model_manager.check_ollama_availability():
                        st.error(
                            "❌ Ollama not available. Please install and start Ollama:\n\n"
                            "1. Download from https://ollama.ai\n"
                            "2. Run: `ollama serve`\n"
                            "3. In another terminal: `ollama pull mistral:7b` (fastest)"
                        )
                        st.stop()
                    model_manager.load_model()
                    st.session_state.model_initialized = True
            
            # Define progress callback
            def update_extraction_progress(current: int, total: int, step_name: str):
                """Update progress for requirement extraction."""
                progress_pct = current / total
                display_name = step_name.replace('_', ' ').title()
                
                progress_bar.progress(
                    progress_pct,
                    text=f"Extracting requirements... {display_name}..."
                )
                with status_text.container():
                    st.caption(f"Step: {current}/{total} | {display_name}")
                with progress_placeholder.container():
                    st.metric("Progress", f"{current}/{total}")
            
            # Extract requirements using tender context from Step 3
            extractor = get_requirement_extractor()
            
            # Pass tender profile metadata for dynamic, context-aware extraction
            profile = st.session_state.tender_profile
            
            requirements = extractor.extract_with_fallback(
                tender_doc=st.session_state.tender_doc,
                tender_type=profile.tender_type,
                industry=profile.industry,
                complexity=profile.complexity,
                priority_areas=profile.priority_areas,
                tender_no=st.session_state.tender_doc.tender_no,
                use_llm=not use_fast_mode,  # Skip LLM if fast mode enabled
                progress_callback=update_extraction_progress
            )
            
            st.session_state.requirements = requirements
            progress_bar.progress(1.0, text="✅ Requirements extracted successfully!")
            with status_text.container():
                st.success("✅ Requirements extracted successfully")
        
        except ModelConnectionError as e:
            st.error(f"❌ LLM Connection Error: {str(e)}\n\nPlease ensure Ollama is running.")
            st.stop()
        except RequirementExtractionError as e:
            st.error(f"❌ Extraction Error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Step 3 error: {str(e)}")
            st.stop()
    
    # Display and allow editing of requirements
    with st.expander("📋 Extracted Requirements", expanded=True):
        requirements = st.session_state.requirements
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Fleet",
            "Technical",
            "Scope",
            "Timeline",
            "Budget & Compliance"
        ])
        
        with tab1:
            if requirements.fleet_requirements:
                st.markdown("**Fleet Requirements:**")
                for key, value in requirements.fleet_requirements.items():
                    if value:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No fleet requirements extracted")
        
        with tab2:
            if requirements.technical_specifications:
                st.markdown("**Technical Specifications:**")
                for key, value in requirements.technical_specifications.items():
                    if value:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No technical specifications extracted")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                if requirements.scope_and_deliverables:
                    st.markdown("**Scope & Deliverables:**")
                    for key, value in requirements.scope_and_deliverables.items():
                        if value:
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No scope information extracted")
            
            with col2:
                if requirements.evaluation_criteria:
                    st.markdown("**Evaluation Criteria:**")
                    if isinstance(requirements.evaluation_criteria, dict):
                        for key, value in requirements.evaluation_criteria.items():
                            if value:
                                st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(requirements.evaluation_criteria)
        
        with tab4:
            if requirements.timeline_and_milestones:
                st.markdown("**Timeline & Milestones:**")
                for key, value in requirements.timeline_and_milestones.items():
                    if value:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No timeline information extracted")
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Budget Constraints:**")
                if requirements.budget_constraints:
                    for key, value in requirements.budget_constraints.items():
                        if value:
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No budget information extracted")
            
            with col2:
                st.markdown("**Compliance Requirements:**")
                if requirements.compliance_requirements:
                    for key, value in requirements.compliance_requirements.items():
                        if value:
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No compliance requirements extracted")
    
    st.info("ℹ️ Requirements have been extracted automatically. Click 'Next' to generate the proposal.")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, key="back_4"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Next: Generate Proposal ➡️", use_container_width=True, key="next_4"):
            st.session_state.step = 5
            st.rerun()


# ========================
# Step 5: Generate Proposal (Dynamic)
# ========================

def render_step_5_generate_proposal():
    """Render proposal generation step using dynamic generator."""
    if (st.session_state.tender_doc is None or 
        st.session_state.org_data is None or 
        st.session_state.requirements is None or
        st.session_state.proposal_structure is None):
        st.warning("⚠️ Please complete previous steps first")
        return
    
    st.subheader("Step 4: Generate Dynamic Proposal")
    
    if st.session_state.proposal_content is None:
        st.info("🔄 Generating proposal with dynamic structure...")
        
        # Progress tracking containers
        col1, col2 = st.columns([3, 1])
        with col1:
            status_text = st.empty()
        with col2:
            progress_placeholder = st.empty()
        
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # Use enhanced dynamic proposal generator
            generator = EnhancedProposalGenerator()
            
            structure = st.session_state.proposal_structure
            total_sections = len(structure.section_order) + 1  # +1 for cover page
            
            # Define progress callback
            def update_progress(current: int, total: int, section_name: str):
                """Update progress bar and status text for section generation."""
                progress_pct = current / total
                # Format section name for display
                display_name = section_name.replace('_', ' ').title()
                
                progress_bar.progress(
                    progress_pct,
                    text=f"Generating {display_name}..."
                )
                with status_text.container():
                    st.caption(f"Sections: {current}/{total} | Current: {display_name}")
                with progress_placeholder.container():
                    st.metric("Progress", f"{current}/{total}")
            
            # Generate using designed structure - reuse cached profile and structure from Step 3
            dynamic_proposal = generator.generate_dynamic_proposal(
                tender_doc=st.session_state.tender_doc,
                org_data=st.session_state.org_data,
                requirements=st.session_state.requirements,
                proposal_structure=st.session_state.proposal_structure,
                tender_profile=st.session_state.tender_profile,
                budget_context=st.session_state.tender_doc.budget_info,
                timeline_context=st.session_state.tender_doc.timeline,
                evaluation_criteria=st.session_state.requirements.evaluation_criteria.get('weighted_criteria', []) if st.session_state.requirements.evaluation_criteria else [],
                compliance_context=st.session_state.requirements.compliance_requirements if st.session_state.requirements else None,
                progress_callback=update_progress
            )
            
            st.session_state.proposal_content = dynamic_proposal
            progress_bar.progress(1.0, text="✅ Proposal generated successfully!")
            with status_text.container():
                st.success(f"✅ All {total_sections} sections generated successfully!")
            
        except Exception as e:
            import traceback
            st.error(f"❌ Generation Error: {str(e)}")
            logger.error(f"Proposal generation error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.stop()
    
    # Display generated dynamic proposal
    if st.session_state.proposal_content is None:
        st.warning("No proposal generated yet")
        return
    
    proposal = st.session_state.proposal_content
    
    # Verify proposal has required attributes
    if not hasattr(proposal, 'sections') or not hasattr(proposal, 'design'):
        st.error("❌ Proposal object is malformed or incomplete")
        logger.error(f"Proposal object missing attributes. Has sections: {hasattr(proposal, 'sections')}, Has design: {hasattr(proposal, 'design')}")
        return
    
    with st.expander("📄 Generated Dynamic Proposal", expanded=True):
        # Display cover page first if present
        if 'cover_page' in proposal.sections:
            st.subheader("📋 Cover Page")
            st.text(proposal.sections['cover_page'])
            st.divider()
        
        # Create tabs dynamically based on design sections (excluding cover_page)
        section_names = proposal.design.section_order
        
        # Create tabs for each section
        tabs = st.tabs([f"📋 {proposal.design.section_definitions[sec_name].title}" for sec_name in section_names])
        
        for tab, section_name in zip(tabs, section_names):
            with tab:
                section_content = proposal.sections.get(section_name, "")
                st.write(section_content)
                
                # Show section details
                section_def = proposal.design.section_definitions[section_name]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"**Importance:** {section_def.importance}")
                with col2:
                    st.caption(f"**Length:** {len(section_content.split())} words")
                with col3:
                    st.caption(f"**Key Points:** {', '.join(section_def.key_points[:2]) if section_def.key_points else '—'}")
    
    # Show design reasoning
    with st.expander("🎯 Design Rationale", expanded=False):
        st.write(f"**Why this structure?**\n\n{proposal.design_rationale}")
        st.write(f"\n**Success Factors in This Proposal:**\n")
        for i, factor in enumerate(proposal.success_factors, 1):
            st.write(f"{i}. {factor}")
    
    st.success("✅ Proposal ready for review and refinement!")
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, key="back_5"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("Next: Refine & Export ➡️", use_container_width=True, key="next_5"):
            st.session_state.step = 6
            st.rerun()


# ========================
# Step 5: Refine & Export
# ========================

def render_step_5_refine_export():
    """Render refinement and export step."""
    if st.session_state.proposal_content is None:
        st.warning("⚠️ Please generate proposal first")
        return
    
    st.subheader("Step 5: Refine & Export Proposal")
    
    proposal = st.session_state.proposal_content
    
    # Refinement section
    with st.expander("✏️ Refine Proposal Sections", expanded=False):
        st.write("Customize any section of the proposal:")
        
        # Get section choices from the dynamic proposal
        section_choices = proposal.design.section_order
        
        section_choice = st.selectbox(
            "Select section to refine",
            section_choices,
            format_func=lambda x: proposal.design.section_definitions[x].title
        )
        
        refinement_instruction = st.text_area(
            "What would you like to change?",
            placeholder="e.g., Make it more technical, Expand the pricing section, Focus on compliance",
            height=100
        )
        
        if st.button("🔄 Regenerate Section", use_container_width=True):
            if refinement_instruction.strip():
                st.info("🔄 Refining section...")
                try:
                    generator = EnhancedProposalGenerator()
                    
                    refined = generator.regenerate_section(
                        section_name=section_choice,
                        current_content=proposal.sections[section_choice],
                        refinement_instruction=refinement_instruction,
                        section_def=proposal.design.section_definitions[section_choice],
                        tender_doc=st.session_state.tender_doc,
                        tender_profile=st.session_state.tender_profile,
                        org_data=st.session_state.org_data,
                        requirements=st.session_state.requirements
                    )
                    
                    proposal.sections[section_choice] = refined
                    st.session_state.proposal_content = proposal
                    st.success("✅ Section refined successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Refinement failed: {str(e)}")
            else:
                st.warning("⚠️ Please provide refinement instructions")
    
    # Export section
    st.write("---")
    st.subheader("📥 Export Proposal")
    
    # Export format selection
    st.write("Choose how you'd like to export your proposal:")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        export_branded = st.checkbox("✨ Branded Proposal", value=True, help="Professional document with Safaricom branding")
    with col2:
        export_filled = st.checkbox(
            "📋 Filled Tender", 
            value=bool(st.session_state.tender_file_path),
            disabled=not st.session_state.tender_file_path,
            help="Original tender with proposal content filled into form fields (requires uploaded tender)"
        )
    with col3:
        export_zip = st.checkbox(
            "📦 Both (ZIP)", 
            value=False,
            disabled=not (st.session_state.tender_file_path and export_branded and export_filled),
            help="Download both formats in a ZIP archive"
        )
    
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        filename = st.text_input(
            "Filename (without extension)",
            value=f"Proposal_{st.session_state.org_data.get('name', 'Safaricom').replace(' ', '_')}"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        if st.button("📥 Generate & Download", use_container_width=True, type="primary"):
            if not (export_branded or export_filled):
                st.warning("⚠️ Please select at least one export format")
            else:
                try:
                    st.info("🔄 Generating documents...")
                    
                    exporter = get_document_exporter()
                    
                    # Convert dynamic proposal to export format
                    export_data = {
                        'title': st.session_state.tender_doc.title,
                        'organization': st.session_state.org_data.get('name', 'Safaricom'),
                        'sections': proposal.sections,
                        'section_order': proposal.design.section_order,
                        'design_rationale': proposal.design_rationale,
                        'success_factors': proposal.success_factors
                    }
                    
                    # Generate ZIP if both formats requested
                    if export_zip:
                        st.info("📦 Creating ZIP archive with both formats...")
                        
                        zip_bytes = exporter.export_dual_as_zip(
                            proposal_content=export_data,
                            org_data=st.session_state.org_data,
                            original_tender_path=st.session_state.tender_file_path,
                            tender_title=st.session_state.tender_doc.title
                        )
                        
                        # Save to database
                        db_service = get_db_service()
                        proposal_id = db_service.insert_proposal(
                            tender_id=st.session_state.tender_id,
                            proposal_version="v1_dynamic_dual_export",
                            content=export_data,
                            org_data=st.session_state.org_data,
                            status="final"
                        )
                        
                        st.success("✅ ZIP archive generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="⬇️ Download ZIP (Branded + Filled Tender)",
                            data=zip_bytes,
                            file_name=f"{filename}_Dual_Export.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    
                    else:
                        # Generate individual formats
                        if export_branded:
                            st.info("✨ Generating branded proposal...")
                            
                            docx_bytes = exporter.export_to_docx(
                                proposal_content=export_data,
                                org_data=st.session_state.org_data,
                                tender_title=st.session_state.tender_doc.title
                            )
                            
                            st.success("✅ Branded proposal generated!")
                            
                            st.download_button(
                                label="⬇️ Download Branded Proposal",
                                data=docx_bytes,
                                file_name=f"{filename}_Branded.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                        
                        if export_filled:
                            st.info("📋 Generating filled tender...")
                            
                            filled_bytes = exporter.export_as_filled_tender(
                                original_tender_path=st.session_state.tender_file_path,
                                proposal_content=export_data,
                                org_data=st.session_state.org_data
                            )
                            
                            st.success("✅ Filled tender generated!")
                            
                            st.download_button(
                                label="⬇️ Download Filled Tender",
                                data=filled_bytes,
                                file_name=f"{filename}_Filled_Tender.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True
                            )
                        
                        # Save to database
                        db_service = get_db_service()
                        export_version = "v1_dynamic_branded" if export_branded and not export_filled else "v1_dynamic_filled" if export_filled and not export_branded else "v1_dynamic_both"
                        proposal_id = db_service.insert_proposal(
                            tender_id=st.session_state.tender_id,
                            proposal_version=export_version,
                            content=export_data,
                            org_data=st.session_state.org_data,
                            status="final"
                        )
                        
                        st.success("✅ Export completed successfully!")
                
                except DocumentExportError as e:
                    st.error(f"❌ Export Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Failed to generate documents: {str(e)}")
                    logger.error(f"Export error: {str(e)}")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, key="back_5"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 New Proposal", use_container_width=True):
            for key in st.session_state:
                del st.session_state[key]
            init_session_state()
            st.rerun()
    with col3:
        st.write("")  # Spacing


# ========================
# Error Handling & Sidebar
# ========================

def render_sidebar():
    """Render sidebar with help and status."""
    with st.sidebar:
        st.write("---")
        st.subheader("ℹ️ Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", "Ready ✅" if st.session_state.model_initialized else "Not Ready")
        
        with col2:
            st.metric("Step", f"{st.session_state.step}/6")
        
        st.write("---")
        st.subheader("📚 Help")
        st.write("""
**Sequential Workflow:**
1️⃣ Upload tender (PDF, text, or form)
2️⃣ Enter organization details
3️⃣ AI designs optimal proposal structure
4️⃣ Auto-extract requirements (LLM-based)
5️⃣ Generate dynamic proposal with custom sections
6️⃣ Refine & download as .docx

**What's New:**
✨ Dynamic proposal structures based on tender type
✨ Each tender gets custom sections (5-12 sections)
✨ Sections ordered by buyer priorities
✨ Section emphasis adjusted by importance
✨ Learns from similar past proposals
✨ Better win rates through targeted content

**Tips:**
- Llama 3.1 8B: Best quality (2-3 min)
- Mistral 7B Instruct: Fastest (1.5-2 min)
- DeepSeek R1 8B: Best reasoning for extraction
- RAG uses past proposals for context
- All data stored locally
- Export as professional .docx
        """)
        
        st.write("---")
        st.subheader("🛠️ Model Info")
        
        model_manager = get_model_manager()
        if st.button("Check LLM Status"):
            if model_manager.check_ollama_availability():
                models = model_manager.list_available_models()
                st.success(f"Available models: {', '.join(models)}")
            else:
                st.error("Ollama not available. Please start Ollama service.")


# ========================
# Main Render Function
# ========================

def render():
    """Main render function."""
    init_session_state()
    
    render_header()
    render_sidebar()
    
    # Route to appropriate step
    if st.session_state.step == 1:
        render_step_1_tender_input()
    elif st.session_state.step == 2:
        render_step_2_organization_info()
    elif st.session_state.step == 3:
        render_step_3_structure_design()
    elif st.session_state.step == 4:
        render_step_4_extract_requirements()
    elif st.session_state.step == 5:
        render_step_5_generate_proposal()
    elif st.session_state.step == 6:
        render_step_6_refine_export()


if __name__ == "__main__":
    render()
