"""
RAG (Retrieval-Augmented Generation) Service
Handles semantic search and retrieval of past proposals for context injection.
Uses SentenceTransformers for efficient embeddings.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
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


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Manages document store and provides retrieval interface for proposal generation.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        training_data_dir: str = "data/training_proposals"
    ):
        """
        Initialize RAG service.
        
        Args:
            embedding_model: SentenceTransformer model to use
            training_data_dir: Directory containing training proposal files
        """
        try:
            embedding_provider = SentenceTransformerEmbedding(embedding_model)
            self.document_store = DocumentStore(embedding_provider)
            self.training_data_dir = Path(training_data_dir)
            
            logger.info("RAG service initialized")
        except EmbeddingServiceError as e:
            logger.error(f"RAG service initialization failed: {str(e)}")
            raise

    def load_training_data(self) -> int:
        """
        Load training proposals from disk.
        Looks for .txt files in training_data_dir.
        
        Returns:
            int: Number of documents loaded
            
        Raises:
            RAGError: If loading fails
        """
        try:
            self.training_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for .txt files
            proposal_files = list(self.training_data_dir.glob("*.txt"))
            
            if not proposal_files:
                logger.warning(f"No training proposals found in {self.training_data_dir}")
                return 0
            
            logger.info(f"Loading {len(proposal_files)} training proposals")
            
            for file_path in proposal_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata from filename (e.g., proposal_telecom_executive_summary.txt)
                    parts = file_path.stem.split('_')
                    industry = parts[1] if len(parts) > 1 else None
                    proposal_type = parts[2] if len(parts) > 2 else "general"
                    
                    self.document_store.add_document(
                        doc_id=file_path.stem,
                        content=content,
                        industry=industry,
                        proposal_type=proposal_type,
                        metadata={'filename': file_path.name}
                    )
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
            
            # Index all embeddings
            self.document_store.index_all_embeddings()
            
            logger.info(f"Loaded {self.document_store.size()} training proposals")
            return self.document_store.size()
        
        except Exception as e:
            logger.error(f"Failed to load training data: {str(e)}")
            raise RAGError(f"Failed to load training data: {str(e)}")

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
        threshold: float = 0.3
    ) -> List[str]:
        """
        Retrieve similar proposals for context injection.
        
        Args:
            query: Search context (tender description or extracted requirements)
            top_k: Number of proposals to retrieve
            industry: Optional industry filter
            threshold: Minimum similarity score
        
        Returns:
            List[str]: Retrieved proposal contents, ranked by relevance
            
        Raises:
            RAGError: If retrieval fails
        """
        try:
            results = self.document_store.search(
                query=query,
                top_k=top_k,
                industry_filter=industry,
                threshold=threshold
            )
            
            logger.info(f"Retrieved {len(results)} similar proposals")
            return [result.content for result in results]
        
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise RAGError(f"Retrieval failed: {str(e)}")

    def get_context_for_generation(
        self,
        tender_summary: str,
        industry: Optional[str] = None,
        max_examples: int = 3
    ) -> str:
        """
        Get formatted context string for LLM prompt injection.
        
        Args:
            tender_summary: Summary of tender/requirements
            industry: Industry for filtering
            max_examples: Maximum examples to include
        
        Returns:
            str: Formatted context string with past proposals
        """
        try:
            proposals = self.retrieve_similar_proposals(
                query=tender_summary,
                top_k=max_examples,
                industry=industry
            )
            
            if not proposals:
                return ""
            
            context = "## Reference Past Proposals\n\n"
            for i, proposal in enumerate(proposals, 1):
                # Truncate very long proposals
                truncated = proposal[:500] + "..." if len(proposal) > 500 else proposal
                context += f"### Example {i}\n{truncated}\n\n"
            
            return context
        
        except RAGError:
            logger.warning("Failed to generate context")
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
            logger.info("Rebuilding RAG index...")
            self.document_store.clear()
            count = self.load_training_data()
            logger.info(f"RAG index rebuilt with {count} documents")
            return count
        except Exception as e:
            logger.error(f"Failed to rebuild RAG index: {str(e)}")
            raise RAGError(f"Failed to rebuild RAG index: {str(e)}")


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
