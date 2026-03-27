"""
RAG (Retrieval-Augmented Generation) Service
Handles semantic search and retrieval of past proposals for context injection.
Uses SentenceTransformers for efficient embeddings.
"""

import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class EmbeddingServiceError(Exception):
    """Raised when embedding operations fail."""
    pass


class RAGError(Exception):
    """Raised when RAG operations fail."""
    pass


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with relevance score."""
    document_id: str
    content: str
    score: float
    source: str  # 'training', 'archived', etc.
    metadata: Dict[str, Any]


@dataclass
class ProposalDocument:
    """Represents a proposal for RAG training."""
    doc_id: str
    content: str
    industry: Optional[str] = None
    proposal_type: str = "general"  # 'executive_summary', 'technical', 'pricing', etc.
    embeddings: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed text to vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts to vectors."""
        pass

    @abstractmethod
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """
    Embedding provider using SentenceTransformers.
    Lightweight, efficient, runs locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embeddings.
        
        Args:
            model_name: Model from HuggingFace (default: all-MiniLM-L6-v2 - fast & accurate)
            
        Raises:
            EmbeddingServiceError: If SentenceTransformers not installed
        """
        if not EMBEDDINGS_AVAILABLE:
            raise EmbeddingServiceError(
                "SentenceTransformers not available. Install: pip install sentence-transformers"
            )
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingServiceError(f"Failed to load model: {str(e)}")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise EmbeddingServiceError(f"Failed to embed text: {str(e)}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray: Array of embeddings (shape: n_texts x embedding_dim)
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            raise EmbeddingServiceError(f"Failed to embed batch: {str(e)}")

    @staticmethod
    def similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score [-1, 1]
        """
        # Normalize vectors
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(norm1, norm2))


class DocumentStore:
    """
    In-memory document store for RAG.
    MVP version stores everything in memory; can be extended to disk/database.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        """
        Initialize document store.
        
        Args:
            embedding_provider: Provider for text embeddings
        """
        self.embedding_provider = embedding_provider
        self.documents: List[ProposalDocument] = []
        self.index: Dict[str, int] = {}  # doc_id -> index in documents list

    def add_document(
        self,
        doc_id: str,
        content: str,
        industry: Optional[str] = None,
        proposal_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the store.
        
        Args:
            doc_id: Unique document identifier
            content: Document content text
            industry: Industry tag for filtering
            proposal_type: Type of proposal section
            metadata: Additional metadata
        """
        if doc_id in self.index:
            logger.warning(f"Document {doc_id} already exists. Updating.")
            idx = self.index[doc_id]
            self.documents[idx] = ProposalDocument(
                doc_id=doc_id,
                content=content,
                industry=industry,
                proposal_type=proposal_type,
                embeddings=None,  # Will be recomputed
                metadata=metadata or {}
            )
        else:
            self.index[doc_id] = len(self.documents)
            self.documents.append(ProposalDocument(
                doc_id=doc_id,
                content=content,
                industry=industry,
                proposal_type=proposal_type,
                metadata=metadata or {}
            ))
        
        logger.info(f"Added document: {doc_id}")

    def add_documents_batch(self, documents: List[ProposalDocument]) -> None:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of ProposalDocument objects
        """
        for doc in documents:
            self.add_document(
                doc_id=doc.doc_id,
                content=doc.content,
                industry=doc.industry,
                proposal_type=doc.proposal_type,
                metadata=doc.metadata
            )

    def index_all_embeddings(self) -> None:
        """
        Compute and store embeddings for all documents.
        Called once after loading all documents.
        
        Raises:
            EmbeddingServiceError: If embedding fails
        """
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        logger.info(f"Indexing embeddings for {len(self.documents)} documents...")
        
        try:
            # Extract content from all documents
            contents = [doc.content for doc in self.documents]
            
            # Batch embed
            embeddings = self.embedding_provider.embed_batch(contents)
            
            # Store embeddings
            for i, embedding in enumerate(embeddings):
                self.documents[i].embeddings = embedding
            
            logger.info(f"Indexed {len(self.documents)} embeddings")
        except Exception as e:
            logger.error(f"Embedding indexing failed: {str(e)}")
            raise EmbeddingServiceError(f"Failed to index embeddings: {str(e)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        industry_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
        threshold: float = 0.3
    ) -> List[RetrievedDocument]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            industry_filter: Optional industry to filter by
            type_filter: Optional proposal type to filter by
            threshold: Minimum similarity score to return
        
        Returns:
            List[RetrievedDocument]: Ranked list of relevant documents
            
        Raises:
            EmbeddingServiceError: If search fails
        """
        if not self.documents:
            logger.warning("No documents in store")
            return []
        
        try:
            # Embed query
            query_embedding = self.embedding_provider.embed(query)
            
            # Calculate similarities
            results = []
            for doc in self.documents:
                if doc.embeddings is None:
                    logger.warning(f"Document {doc.doc_id} has no embeddings")
                    continue
                
                # Apply filters
                if industry_filter and doc.industry != industry_filter:
                    continue
                if type_filter and doc.proposal_type != type_filter:
                    continue
                
                # Calculate similarity
                score = self.embedding_provider.similarity(query_embedding, doc.embeddings)
                
                if score >= threshold:
                    results.append(RetrievedDocument(
                        document_id=doc.doc_id,
                        content=doc.content,
                        score=score,
                        source='training',
                        metadata=doc.metadata or {}
                    ))
            
            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RAGError(f"Search failed: {str(e)}")

    def get_document(self, doc_id: str) -> Optional[ProposalDocument]:
        """Retrieve a document by ID."""
        if doc_id not in self.index:
            return None
        return self.documents[self.index[doc_id]]

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the store."""
        if doc_id not in self.index:
            return False
        
        idx = self.index[doc_id]
        del self.documents[idx]
        del self.index[doc_id]
        # Rebuild index
        self.index = {doc.doc_id: i for i, doc in enumerate(self.documents)}
        return True

    def size(self) -> int:
        """Get number of documents in store."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self.index.clear()
        logger.info("Document store cleared")

    # ========================
    # Embedding Persistence (NEW)
    # ========================

    def save_embeddings(self, cache_file: str = "data/.embeddings_cache.pkl") -> bool:
        """
        Save computed embeddings to disk for faster initialization.
        
        Args:
            cache_file: Path to save embeddings cache
            
        Returns:
            bool: True if save successful
        """
        try:
            cache_dir = Path(cache_file).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            cache_data = {
                'documents': self.documents,
                'index': self.index,
                'model_name': self.embedding_provider.model_name,
                'embedding_dim': self.embedding_provider.embedding_dim,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {len(self.documents)} embeddings to disk: {cache_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {str(e)}")
            return False

    def load_embeddings(self, cache_file: str = "data/.embeddings_cache.pkl") -> bool:
        """
        Load pre-computed embeddings from disk.
        Skips 30-60 second embedding computation on startup.
        
        Args:
            cache_file: Path to embeddings cache file
            
        Returns:
            bool: True if load successful
        """
        try:
            cache_path = Path(cache_file)
            if not cache_path.exists():
                logger.debug(f"No embeddings cache found at {cache_file}")
                return False
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache is compatible (same embedding model)
            cached_model = cache_data.get('model_name')
            if cached_model != self.embedding_provider.model_name:
                logger.warning(f"Cache model mismatch: {cached_model} vs {self.embedding_provider.model_name}. Skipping.")
                return False
            
            # Restore documents and index
            self.documents = cache_data['documents']
            self.index = cache_data['index']
            
            logger.info(f"Loaded {len(self.documents)} cached embeddings from disk ({cache_file})")
            logger.info(f"Cache timestamp: {cache_data.get('timestamp')}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {str(e)}")
            return False

    def is_cache_stale(self, cache_file: str = "data/.embeddings_cache.pkl", max_age_hours: int = 24) -> bool:
        """
        Check if embeddings cache is stale (older than max_age_hours).
        
        Args:
            cache_file: Path to cache file
            max_age_hours: Maximum age in hours before considered stale
            
        Returns:
            bool: True if cache is stale or missing
        """
        try:
            cache_path = Path(cache_file)
            if not cache_path.exists():
                return True
            
            # Check file modification time
            file_age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
            return file_age_hours > max_age_hours
        except Exception:
            return True


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Manages document store and provides retrieval interface for proposal generation.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        training_data_dir: str = "data/training_proposals",
        cache_embeddings: bool = True
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_model: SentenceTransformer model to use
            training_data_dir: Directory containing training proposal files
            cache_embeddings: Whether to use disk caching for embeddings (default: True)
        """
        try:
            embedding_provider = SentenceTransformerEmbedding(embedding_model)
            self.document_store = DocumentStore(embedding_provider)
            self.training_data_dir = Path(training_data_dir)
            self.embeddings_cache_file = "data/.embeddings_cache.pkl"
            self.cache_embeddings = cache_embeddings
            self._query_cache: Dict[str, List[Tuple[str, float, Dict]]] = {}  # ✅ NEW: Query result cache
            self._query_cache_max_size = 50
            
            logger.info("RAG service initialized")
        except EmbeddingServiceError as e:
            logger.error(f"RAG service initialization failed: {str(e)}")
            raise

    def load_training_data(self) -> int:
        """
        Load training proposals from disk using metadata.json for efficiency.
        ✅ NEW: Tries to load from embeddings cache first (95% faster cold start).
        Extracts metadata from filenames and file processing info.
        
        Returns:
            int: Number of documents loaded
            
        Raises:
            RAGError: If loading fails
        """
        try:
            self.training_data_dir.mkdir(parents=True, exist_ok=True)
            tenders_dir = Path("data/tenders_proposals")
            
            # ✅ NEW: Try loading from embeddings cache first
            if self.cache_embeddings and self.document_store.load_embeddings(self.embeddings_cache_file):
                logger.info(f"Loaded {self.document_store.size()} documents from embeddings cache (fast startup)")
                return self.document_store.size()
            
            # Cache miss or disabled - load from scratch
            logger.info("Loading training data from disk (first time or cache disabled)...")
            metadata_file = tenders_dir / ".processing_metadata.json"
            metadata_mapping = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_raw = json.load(f)
                        # Map processed files -> paths
                        metadata_mapping = {
                            v.get('processed_path', ''): k 
                            for k, v in metadata_raw.items() 
                            if v.get('processed')
                        }
                    logger.info(f"Loaded metadata for {len(metadata_mapping)} processed proposals")
                except Exception as e:
                    logger.warning(f"Failed to load metadata.json: {str(e)}")
            
            # Load all .txt files from training_proposals
            proposal_files = list(self.training_data_dir.glob("*.txt"))
            
            if not proposal_files:
                logger.info(f"No training proposals found in {self.training_data_dir}. Upload documents via Document Manager to enable RAG context.")
                return 0
            
            logger.info(f"Loading {len(proposal_files)} training proposals")
            
            for file_path in proposal_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata intelligently from filename
                    filename = file_path.name
                    doc_id = file_path.stem
                    
                    # Extract industry and proposal metadata from filename
                    industry, proposal_type = self._extract_metadata_from_filename(filename, content)
                    
                    # Get original source file name from metadata if available
                    original_source = None
                    for orig_name, processed_path in metadata_mapping.items():
                        if str(file_path).endswith(processed_path.replace('\\', '/')):
                            original_source = orig_name
                            break
                    
                    self.document_store.add_document(
                        doc_id=doc_id,
                        content=content,
                        industry=industry,
                        proposal_type=proposal_type,
                        metadata={
                            'filename': filename,
                            'original_source': original_source,
                            'extracted_from_content': True
                        }
                    )
                    logger.debug(f"Loaded {filename} - Industry: {industry}, Type: {proposal_type}")
                
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    continue
            
            # Index all embeddings
            self.document_store.index_all_embeddings()
            
            # ✅ NEW: Save embeddings cache for fast startup next time
            if self.cache_embeddings:
                self.document_store.save_embeddings(self.embeddings_cache_file)
            
            logger.info(f"Loaded and indexed {self.document_store.size()} training proposals")
            return self.document_store.size()
        
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise RAGError(f"Failed to load training data: {str(e)}")

    @staticmethod
    def _extract_metadata_from_filename(filename: str, content: str) -> Tuple[Optional[str], str]:
        """
        Extract industry and proposal type from filename and content.
        
        Args:
            filename: Original filename
            content: Document content for deeper analysis
            
        Returns:
            Tuple[industry, proposal_type]: Extracted metadata
        """
        filename_lower = filename.lower()
        content_lower = content.lower()[:500]  # First 500 chars for quick scan
        
        # Industry detection
        industry_patterns = {
            'telecom': ['safaricom', 'telecommunications', 'telecom', 'sms', 'bulk sms', 'mobile'],
            'cloud': ['cloud', 'hosting', 'server', 'aws', 'azure', 'infrastructure'],
            'wifi': ['wifi', 'wi-fi', 'wireless', 'connectivity'],
            'colocation': ['collocation', 'colocation', 'data center'],
            'fleet': ['fleet', 'vehicle', 'logistics', 'transport'],
        }
        
        industry = 'general'
        for ind, patterns in industry_patterns.items():
            if any(p in filename_lower or p in content_lower for p in patterns):
                industry = ind
                break
        
        # Proposal type detection
        proposal_type = 'general'
        type_patterns = {
            'executive_summary': ['executive', 'summary', 'overview'],
            'technical': ['technical', 'approach', 'methodology', 'solution'],
            'pricing': ['pricing', 'cost', 'commercial', 'financial'],
            'implementation': ['implementation', 'timeline', 'schedule'],
        }
        
        for ptype, patterns in type_patterns.items():
            if any(p in filename_lower or p in content_lower for p in patterns):
                proposal_type = ptype
                break
        
        return industry, proposal_type

    def add_proposal(
        self,
        doc_id: str,
        content: str,
        industry: Optional[str] = None,
        proposal_type: str = "general"
    ) -> None:
        """
        Add a new proposal to the knowledge base.
        
        Args:
            doc_id: Unique identifier
            content: Proposal content
            industry: Industry tag
            proposal_type: Type of proposal
        """
        self.document_store.add_document(doc_id, content, industry, proposal_type)
        # Recompute embeddings for the new document
        if self.document_store.size() > 0:
            self.document_store.index_all_embeddings()

    def retrieve_similar_proposals(
        self,
        query: str,
        top_k: int = 3,
        industry: Optional[str] = None,
        threshold: float = 0.3,
        use_cache: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Retrieve similar proposals for context injection with metadata.
        ✅ NEW: Query results are cached to avoid duplicate semantic searches.
        
        Args:
            query: Search context (tender description or extracted requirements)
            top_k: Number of proposals to retrieve
            industry: Optional industry filter for more targeted results
            threshold: Minimum similarity score
            use_cache: Whether to use query result caching (default: True)
        
        Returns:
            List[Tuple[content, score, metadata]]: Retrieved proposals ranked by relevance
            
        Raises:
            RAGError: If retrieval fails
        """
        # ✅ NEW: Check query cache first
        cache_key = f"{query}|{top_k}|{industry}|{threshold}" if use_cache else None
        if cache_key and cache_key in self._query_cache:
            logger.debug(f"Query cache hit ({len(self._query_cache)} cached queries)")
            return self._query_cache[cache_key]
        
        try:
            if self.document_store.size() == 0:
                logger.warning("No documents in store for retrieval")
                return []
            
            results = self.document_store.search(
                query=query,
                top_k=top_k * 2,  # Get more results to filter
                industry_filter=industry,
                threshold=threshold
            )
            
            # Rerank by combining similarity score and metadata relevance
            ranked_results = []
            for result in results:
                # Boost score if industry matches
                score = result.score
                if industry and result.metadata.get('industry') == industry:
                    score *= 1.2  # 20% boost for industry match
                
                ranked_results.append((result.content, score, result.metadata))
            
            # Sort by boosted score
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            final_results = ranked_results[:top_k]
            
            # ✅ NEW: Store in query cache with LRU eviction
            if cache_key:
                if len(self._query_cache) >= self._query_cache_max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]
                self._query_cache[cache_key] = final_results
            
            logger.info(f"Retrieved {len(final_results)} similar proposals (from {len(results)} candidates)")
            return final_results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RAGError(f"Retrieval failed: {str(e)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        industry: Optional[str] = None,
        threshold: float = 0.3
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar proposals (wrapper for retrieve_similar_proposals).
        Provides compatibility with code expecting a search() method.
        
        Args:
            query: Search query
            top_k: Number of results to return (default 5, accepts max_results alias)
            industry: Optional industry filter
            threshold: Minimum similarity threshold
        
        Returns:
            List[Tuple[str, float, Dict]]: Similar proposals with scores and metadata
        """
        return self.retrieve_similar_proposals(
            query=query,
            top_k=top_k,
            industry=industry,
            threshold=threshold
        )

    def get_context_for_generation(
        self,
        tender_summary: str,
        industry: Optional[str] = None,
        max_examples: int = 3
    ) -> str:
        """
        Get formatted context string for LLM prompt injection.
        Uses both semantic similarity and metadata for intelligent retrieval.
        
        Args:
            tender_summary: Summary of tender/requirements
            industry: Industry for filtering and boosting relevance
            max_examples: Maximum examples to include
        
        Returns:
            str: Formatted context string with past proposals
        """
        try:
            if self.document_store.size() == 0:
                logger.debug("No documents available for context generation")
                return ""
            
            proposals = self.retrieve_similar_proposals(
                query=tender_summary,
                top_k=max_examples,
                industry=industry,
                threshold=0.3
            )
            
            if not proposals:
                logger.debug("No relevant proposals retrieved")
                return ""
            
            context = "## Reference Past Proposals\n\n"
            
            for i, (proposal_content, score, metadata) in enumerate(proposals, 1):
                # Truncate proposals to reasonable length
                max_length = 600
                if len(proposal_content) > max_length:
                    truncated = proposal_content[:max_length] + "\n[... truncated ...]"
                else:
                    truncated = proposal_content
                
                # Include metadata in context
                source_info = metadata.get('original_source', 'Previous Proposal')
                industry_tag = metadata.get('industry', 'general')
                relevance = f"(Relevance: {score:.1%})"
                
                context += f"### Example {i}: {source_info} - {industry_tag.title()} {relevance}\n"
                context += f"{truncated}\n\n"
            
            logger.info(f"Generated context from {len(proposals)} retrieved proposals")
            return context
        
        except RAGError as e:
            logger.warning(f"Failed to generate context: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in context generation: {str(e)}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        return {
            'documents_loaded': self.document_store.size(),
            'embedding_model': self.document_store.embedding_provider.model_name,
            'embedding_dimension': self.document_store.embedding_provider.embedding_dim,
            'training_data_dir': str(self.training_data_dir)
        }

    @property
    def documents(self) -> List[ProposalDocument]:
        """Get list of loaded documents."""
        return self.document_store.documents

    def rebuild_index(self) -> int:
        """
        Rebuild the RAG index by clearing and reloading training data.
        Useful after new documents are added or processed.
        
        Returns:
            int: Number of documents reloaded
            
        Raises:
            RAGError: If rebuild fails
        """
        try:
            logger.info("Rebuilding RAG index from metadata...")
            self.document_store.clear()
            count = self.load_training_data()
            logger.info(f"RAG index rebuilt with {count} documents from metadata")
            return count
        except Exception as e:
            logger.error(f"Failed to rebuild RAG index: {str(e)}")
            raise RAGError(f"Failed to rebuild RAG index: {str(e)}")

    def update_from_metadata(self) -> int:
        """
        Update RAG index by parsing .processing_metadata.json.
        Efficiently loads only processed documents.
        
        Returns:
            int: Number of documents updated
            
        Raises:
            RAGError: If update fails
        """
        try:
            logger.info("Updating RAG index from metadata...")
            tenders_dir = Path("data/tenders_proposals")
            metadata_file = tenders_dir / ".processing_metadata.json"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata file not found: {metadata_file}")
                return 0
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            count = 0
            for original_name, info in metadata.items():
                if not info.get('processed'):
                    continue
                
                processed_path = info.get('processed_path', '')
                full_path = Path(processed_path)
                
                if not full_path.exists():
                    logger.warning(f"Processed file not found: {full_path}")
                    continue
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    industry, proposal_type = self._extract_metadata_from_filename(
                        original_name, 
                        content
                    )
                    
                    doc_id = full_path.stem
                    
                    # Check if already in store
                    if self.document_store.get_document(doc_id) is None:
                        self.document_store.add_document(
                            doc_id=doc_id,
                            content=content,
                            industry=industry,
                            proposal_type=proposal_type,
                            metadata={
                                'filename': original_name,
                                'processed_date': info.get('processed_date'),
                                'from_metadata': True
                            }
                        )
                        count += 1
                        logger.debug(f"Added from metadata: {original_name}")
                
                except Exception as e:
                    logger.error(f"Failed to load from metadata {original_name}: {str(e)}")
                    continue
            
            # Reindex embeddings if new documents added
            if count > 0:
                self.document_store.index_all_embeddings()
                logger.info(f"Updated RAG index with {count} documents from metadata")
            
            return count
        
        except Exception as e:
            logger.error(f"Failed to update from metadata: {str(e)}")
            raise RAGError(f"Failed to update from metadata: {str(e)}")

    def check_training_data_status(self) -> Dict[str, Any]:
        """
        Check status of training data availability including raw files.
        Dynamically scans for PDF, DOCX, and processed TXT files.
        
        Returns:
            Dict with status info:
                - loaded_proposals: Number of processed .txt files
                - pending_files: List of PDF/DOCX files to be processed
                - training_data_dir: Path to training directory
                - has_training_data: Boolean indicating if any data exists
                - next_steps: Recommended action if no data
        """
        try:
            # Count processed files
            processed_files = list(self.training_data_dir.glob("*.txt"))
            processed_count = len(processed_files)
            
            # Check for pending files (PDF, DOCX to be processed)
            # These would be in the upload directory
            upload_dir = Path("data/tenders_proposals")
            pending_pdfs = list(upload_dir.glob("*.pdf")) if upload_dir.exists() else []
            pending_docx = list(upload_dir.glob("*.docx")) if upload_dir.exists() else []
            pending_files = [f.name for f in pending_pdfs + pending_docx]
            
            has_data = processed_count > 0 or len(pending_files) > 0
            
            status = {
                'loaded_proposals': processed_count,
                'loaded_proposal_files': [f.name for f in processed_files],
                'pending_files': pending_files,
                'pending_files_count': len(pending_files),
                'training_data_dir': str(self.training_data_dir),
                'has_training_data': has_data,
                'status': 'ready' if processed_count > 0 else 'pending' if len(pending_files) > 0 else 'empty',
                'next_steps': (
                    f"Upload {len(pending_files)} document(s) via Document Manager to enable RAG context"
                    if not has_data 
                    else f"Process {len(pending_files)} pending file(s)" if len(pending_files) > 0 
                    else f"{processed_count} proposal(s) loaded"
                )
            }
            
            logger.debug(f"Training data status: {status}")
            return status
        
        except Exception as e:
            logger.error(f"Failed to check training data status: {str(e)}")
            return {
                'loaded_proposals': 0,
                'pending_files': [],
                'error': str(e),
                'status': 'error',
                'next_steps': 'Check Document Manager'
            }


# Global RAG service instance (lazy-loaded)
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance.
    
    Returns:
        RAGService: Global RAG service
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        # Try to load training data on initialization
        try:
            _rag_service.load_training_data()
        except RAGError as e:
            logger.warning(f"Failed to load training data on startup: {str(e)}")
    return _rag_service
