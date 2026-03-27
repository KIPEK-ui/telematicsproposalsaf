"""
Document Manager View
Streamlit interface for uploading, managing, and processing documents for RAG training.
Integrated into the main proposal generator workflow.
"""

import os
# Disable PaddleOCR model source check for faster startup
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import streamlit as st
import logging
from typing import Optional
from pathlib import Path

from app.services.document_manager_service import get_document_manager, DocumentManagerError
from app.services.document_processor import DocumentProcessor
from app.ai_service.rag_service import get_rag_service

logger = logging.getLogger(__name__)


def render_document_manager():
    """Main document manager interface."""
    st.subheader("📚 Training Document Management")
    st.write("Upload and manage documents for better proposal generation using RAG")
    
    # Initialize document manager once (singleton pattern)
    doc_manager = get_document_manager()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "📋 View Uploaded", "⚙️ Process for RAG", "📊 Statistics"])
    
    with tab1:
        render_upload_tab(doc_manager)
    
    with tab2:
        render_view_uploaded_tab(doc_manager)
    
    with tab3:
        render_process_tab(doc_manager)
    
    with tab4:
        render_statistics_tab(doc_manager)


# ========================
# Tab 1: Upload Documents
# ========================

def render_upload_tab(doc_manager):
    """Render file upload interface."""
    st.write("Upload PDF, DOCX, or TXT files to train the RAG system")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Max 50 MB per file. Upload multiple files at once."
        )
    
    with col2:
        st.write("")
        st.write("")
        allow_overwrite = st.checkbox("Allow overwriting duplicates?", value=False)
    
    if uploaded_files:
        st.markdown("---")
        st.write(f"**Processing {len(uploaded_files)} file(s)...**")
        
        progress_bar = st.progress(0)
        status_container = st.container()
        
        upload_results = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
                # Read file bytes
                file_bytes = uploaded_file.read()
                
                # Upload with duplicate check
                result = doc_manager.upload_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    overwrite=allow_overwrite
                )
                
                upload_results.append(result)
                
                # Display result
                with status_container:
                    if result.is_duplicate:
                        st.warning(f"⚠️ **{uploaded_file.name}**: {result.message}")
                    elif result.success:
                        st.success(f"✅ **{uploaded_file.name}**: {result.message}")
                    else:
                        st.error(f"❌ **{uploaded_file.name}**: {result.message}")
                        
            except Exception as e:
                logger.error(f"Upload error for {uploaded_file.name}: {str(e)}")
                with status_container:
                    st.error(f"❌ **{uploaded_file.name}**: {str(e)}")
        
        # Summary
        st.markdown("---")
        successful = sum(1 for r in upload_results if r.success)
        duplicates = sum(1 for r in upload_results if r.is_duplicate)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Uploaded", successful)
        with col2:
            st.metric("Duplicates", duplicates)
        with col3:
            st.metric("Errors", len(upload_results) - successful - duplicates)
        
        # Storage info
        st.info("""
        📁 **Storage Location**: `data/tenders_proposals/`
        
        Your uploaded files are stored locally and available for RAG processing. 
        Switch to the **"Process for RAG"** tab to extract text and build embeddings.
        """)


# ========================
# Tab 2: View Uploaded
# ========================

def render_view_uploaded_tab(doc_manager):
    """Render uploaded documents list."""
    documents = doc_manager.list_uploaded_documents()
    
    if not documents:
        st.info("📭 No documents uploaded yet. Use the **Upload Documents** tab to add files.")
        return
    
    st.write(f"**Total Documents**: {len(documents)}")
    
    # Display documents in a table
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1.5, 1, 1])
    
    with col1:
        st.write("**Filename**")
    with col2:
        st.write("**Type**")
    with col3:
        st.write("**Size**")
    with col4:
        st.write("**Uploaded**")
    with col5:
        st.write("**Status**")
    with col6:
        st.write("**Action**")
    
    st.divider()
    
    for doc in documents:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1.5, 1, 1])
        
        with col1:
            st.caption(doc['filename'])
        with col2:
            st.caption(doc['type'])
        with col3:
            st.caption(f"{doc['size_mb']} MB")
        with col4:
            st.caption(doc['uploaded_date'][:10])  # Date only
        with col5:
            if doc['processed']:
                st.caption("✅ Processed")
            else:
                st.caption("⏳ Pending")
        with col6:
            if st.button("🗑️", key=f"delete_{doc['filename']}", help="Delete this document"):
                success, message = doc_manager.delete_document(doc['filename'])
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    # Statistics
    st.divider()
    stats = doc_manager.get_document_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Size", f"{stats['total_size_mb']} MB")
    with col2:
        st.metric("Total Docs", stats['total_documents'])
    with col3:
        st.metric("Processed", stats['processed'])
    with col4:
        st.metric("Pending", stats['pending'])


# ========================
# Tab 3: Process for RAG
# ========================

def render_process_tab(doc_manager):
    """Render document processing interface."""
    pending = doc_manager.get_pending_documents()
    
    st.write("Process uploaded documents to extract text and build RAG embeddings")
    
    if not pending:
        all_docs = doc_manager.list_uploaded_documents()
        if not all_docs:
            st.info("📭 No documents to process. Upload files first in the **Upload Documents** tab.")
        else:
            st.success("✅ All documents have been processed!")
            st.info("Next step: Restart the app to use the processing documents in proposal generation.")
        return
    
    st.warning(f"⏳ {len(pending)} document(s) pending processing")
    
    # Show pending documents
    st.write("**Documents to Process:**")
    for doc in pending:
        st.caption(f"• {doc['filename']} ({doc['size_mb']} MB)")
    
    st.divider()
    
    # Processing options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Processing Settings:**")
        batch_process = st.checkbox(
            "Process all pending documents at once?",
            value=True,
            help="If unchecked, process one document at a time"
        )
    
    with col2:
        st.write("")
        st.write("")
        process_button = st.button(
            "🚀 Start Processing",
            use_container_width=True,
            type="primary"
        )
    
    if process_button:
        render_processing_workflow(doc_manager, batch_process)


def render_processing_workflow(doc_manager, batch_process: bool):
    """Execute document processing workflow."""
    processor = DocumentProcessor(output_dir=doc_manager.training_dir)
    pending = doc_manager.get_pending_documents()
    
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    successful_count = 0
    failed_count = 0
    
    for idx, doc in enumerate(pending):
        try:
            # Update progress
            current = idx + 1
            progress = current / len(pending)
            progress_bar.progress(progress)
            status_text.write(f"Processing {current}/{len(pending)}: {doc['filename']}")
            
            # Process file
            result = processor.process_file(doc['path'])
            
            # Update status
            if result.success:
                doc_manager.mark_processed(doc['filename'], result.output_path)
                successful_count += 1
                
                with results_container:
                    st.success(f"✅ {doc['filename']}")
                    st.caption(f"   {len(result.extracted_text)} characters extracted")
            else:
                failed_count += 1
                with results_container:
                    st.error(f"❌ {doc['filename']}")
                    st.caption(f"   Error: {result.error_message}")
                    
        except Exception as e:
            failed_count += 1
            logger.error(f"Processing error for {doc['filename']}: {str(e)}")
            with results_container:
                st.error(f"❌ {doc['filename']}: {str(e)}")
    
    # Final status
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Successful", successful_count)
    with col2:
        st.metric("Failed", failed_count)
    with col3:
        st.metric("Total", len(pending))
    
    # Rebuild RAG index
    if successful_count > 0:
        st.info("🔄 Rebuilding RAG index with processed documents...")
        try:
            rag = get_rag_service()
            rag.rebuild_index()
            st.success("✅ RAG index rebuilt successfully!")
            st.success("""
            📚 Training data is ready!
            
            The RAG system now has access to your documents and will use them to improve 
            proposal generation. Go to step 1 of the proposal generator to see the results.
            """)
        except Exception as e:
            st.warning(f"⚠️ Could not rebuild RAG index immediately: {str(e)}")
            st.info("You can restart the app to rebuild the index, or proceed to the proposal generator.")


# ========================
# Tab 4: Statistics
# ========================

def render_statistics_tab(doc_manager):
    """Render statistics and insights."""
    stats = doc_manager.get_document_stats()
    documents = doc_manager.list_uploaded_documents()
    
    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Documents", stats['total_documents'])
    with col2:
        st.metric("💾 Total Size", f"{stats['total_size_mb']} MB")
    with col3:
        st.metric("✅ Processed", stats['processed'])
    with col4:
        st.metric("⏳ Pending", stats['pending'])
    
    st.divider()
    
    # Documents by type
    st.subheader("Documents by Type")
    if stats['by_type']:
        type_data = stats['by_type']
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for doc_type, count in sorted(type_data.items()):
                st.write(f"**{doc_type}**: {count}")
        
        with col2:
            # Simple bar chart using columns
            max_count = max(type_data.values())
            for doc_type, count in sorted(type_data.items()):
                bar_width = int(count / max_count * 30) if max_count > 0 else 0
                st.write(f"{doc_type}: {'█' * bar_width} {count}")
    else:
        st.info("No documents uploaded yet.")
    
    st.divider()
    
    # Storage details
    st.subheader("Storage Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Upload Directory**")
        st.code(str(doc_manager.storage_dir))
        st.caption(f"{len(documents)} files stored")
    
    with col2:
        st.write("**Training Directory**")
        st.code(str(doc_manager.training_dir))
        processed_count = sum(1 for d in documents if d['processed'])
        st.caption(f"{processed_count} files processed")
    
    # RAG Status
    st.divider()
    st.subheader("RAG System Status")
    
    try:
        rag = get_rag_service()
        doc_count = len(rag.documents)
        
        # Get detailed training data status
        training_status = rag.check_training_data_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Loaded Proposals", training_status['loaded_proposals'])
        with col2:
            st.metric("⏳ Pending Files", training_status['pending_files_count'])
        with col3:
            status_label = training_status['status'].upper()
            status_emoji = "🟢" if training_status['status'] == 'ready' else "🟡" if training_status['status'] == 'pending' else "🔴"
            st.metric("Status", f"{status_emoji} {status_label}")
        
        st.divider()
        
        # Status details
        if training_status['status'] == 'ready':
            st.success(f"✅ RAG is active with {training_status['loaded_proposals']} indexed proposal(s)")
        elif training_status['status'] == 'pending':
            st.warning(f"⏳ {training_status['pending_files_count']} file(s) waiting to be processed")
            if training_status['pending_files']:
                st.caption("**Files to process:**")
                for fname in training_status['pending_files']:
                    st.caption(f"  • {fname}")
        else:
            st.info("📭 No training data available yet. Start by uploading documents.")
        
        # Next steps
        st.caption(f"**Next Step**: {training_status['next_steps']}")
        
        # Rebuild button
        if doc_count > 0 or training_status['pending_files_count'] > 0:
            if st.button("🔄 Refresh RAG Index", help="Reload and reindex all processed training documents"):
                with st.spinner("Rebuilding index..."):
                    try:
                        count = rag.rebuild_index()
                        st.success(f"✅ Index rebuilt with {count} document(s)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    except Exception as e:
        st.warning(f"⚠️ Could not check RAG status: {str(e)}")
        st.caption("Make sure sentence-transformers is installed: pip install sentence-transformers")


# ========================
# Public Functions
# ========================

def render():
    """Main render function for integration."""
    render_document_manager()
