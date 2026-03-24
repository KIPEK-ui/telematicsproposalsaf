"""AI services module"""

from .model_manager import get_model_manager, LocalLMManager
from .rag_service import get_rag_service, RAGService
from .requirement_extractor import get_requirement_extractor, RequirementExtractor
from .proposal_generator import get_proposal_generator, ProposalGenerator
from .dynamic_proposal_designer import (
    DynamicProposalDesigner,
    TenderClassifier,
    DynamicProposalStructure,
    TenderProfile
)
from .enhanced_proposal_generator import EnhancedProposalGenerator, DynamicProposalContent

__all__ = [
    'get_model_manager',
    'LocalLMManager',
    'get_rag_service',
    'RAGService',
    'get_requirement_extractor',
    'RequirementExtractor',
    'get_proposal_generator',
    'ProposalGenerator',
    'DynamicProposalDesigner',
    'TenderClassifier',
    'DynamicProposalStructure',
    'TenderProfile',
    'EnhancedProposalGenerator',
    'DynamicProposalContent',
]
