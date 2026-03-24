"""
Requirement Extractor
Uses LLM to extract and structure requirements from parsed tenders.
Converts unstructured tender data into actionable requirements for proposal generation.
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

from app.services.tender_parser import TenderDocument
from app.ai_service.model_manager import get_model_manager, ModelConnectionError, ModelInferenceError


@dataclass
class StructuredRequirements:
    """Structured representation of extracted requirements."""
    fleet_requirements: Dict[str, Any]
    technical_specifications: Dict[str, Any]
    scope_and_deliverables: Dict[str, Any]
    timeline_and_milestones: Dict[str, Any]
    budget_constraints: Dict[str, Any]
    compliance_requirements: Dict[str, Any]
    evaluation_criteria: Dict[str, Any]
    additional_notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RequirementExtractionError(Exception):
    """Raised when requirement extraction fails."""
    pass


class RequirementExtractor:
    """
    Extracts and structures requirements from parsed tender documents.
    Uses LLM-based prompt engineering for intelligent extraction.
    """

    # Extraction prompt template
    EXTRACTION_PROMPT = """You are an expert proposal analyst for Safaricom, a telco/transport company.
Analyze the following tender document and extract structured requirements in JSON format.

TENDER DOCUMENT:
---
{tender_content}
---

Extract and structure the following information into a valid JSON object:
1. FLEET REQUIREMENTS: Vehicle types, quantities, specifications needed
2. TECHNICAL SPECIFICATIONS: Performance metrics, compliance standards, system requirements
3. SCOPE & DELIVERABLES: What needs to be delivered, project scope
4. TIMELINE & MILESTONES: Cutoff dates, project phases, key milestones
5. BUDGET CONSTRAINTS: Budget range, payment terms, cost considerations
6. COMPLIANCE REQUIREMENTS: Certifications, regulations, insurance needs
7. EVALUATION CRITERIA: How proposals will be evaluated

Return a valid JSON object with these keys (all required):
{{
    "fleet_requirements": {{"details": "...", "quantity": "...", "specifications": [...]}},
    "technical_specifications": {{"performance": [...], "compliance": [...], "systems": [...]}},
    "scope_and_deliverables": {{"scope": "...", "deliverables": [...]}},
    "timeline_and_milestones": {{"start_date": "...", "end_date": "...", "milestones": [...]}},
    "budget_constraints": {{"range": "...", "payment_terms": "...", "currency": "..."}},
    "compliance_requirements": {{"certifications": [...], "regulations": [...], "insurance": "..."}},
    "evaluation_criteria": {{"criteria": [...]}},
    "additional_notes": "..."
}}

Only return valid JSON, no other text."""

    def __init__(self):
        """Initialize requirement extractor."""
        self.model_manager = get_model_manager()
        logger.info("RequirementExtractor initialized")

    def extract(self, tender_doc: TenderDocument, use_reasoning_model: bool = True) -> StructuredRequirements:
        """
        Extract structured requirements from a tender document.
        
        Args:
            tender_doc: Parsed tender document
            use_reasoning_model: If True, uses deepseek-r1:8b for better reasoning (default True)
        
        Returns:
            StructuredRequirements: Structured extraction
            
        Raises:
            RequirementExtractionError: If extraction fails
            ModelConnectionError: If model is not available
        """
        try:
            # Prepare the tender content for extraction
            tender_content = self._prepare_tender_content(tender_doc)
            
            # Build extraction prompt
            prompt = self.EXTRACTION_PROMPT.format(tender_content=tender_content)
            
            # Use reasoning model for extraction if available (better for complex analysis)
            if use_reasoning_model:
                try:
                    logger.info("Loading DeepSeek R1 for requirement extraction (better reasoning)...")
                    self.model_manager.load_model("deepseek-r1:8b")
                except Exception as e:
                    logger.warning(f"Could not load deepseek-r1:8b: {e}. Using default model.")
            
            # Ensure a model is loaded
            if not self.model_manager.get_current_model():
                logger.info("Loading default model for extraction...")
                self.model_manager.load_model()
            
            # Generate extraction
            logger.info(f"Extracting requirements via LLM ({self.model_manager.get_current_model().name})...")
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=2000,
                system_prompt="You are a precise JSON extraction expert. Output only valid JSON."
            )
            
            # Parse response
            extracted_data = self._parse_json_response(response)
            
            # Create structured requirements
            requirements = StructuredRequirements(
                fleet_requirements=extracted_data.get('fleet_requirements', {}),
                technical_specifications=extracted_data.get('technical_specifications', {}),
                scope_and_deliverables=extracted_data.get('scope_and_deliverables', {}),
                timeline_and_milestones=extracted_data.get('timeline_and_milestones', {}),
                budget_constraints=extracted_data.get('budget_constraints', {}),
                compliance_requirements=extracted_data.get('compliance_requirements', {}),
                evaluation_criteria=extracted_data.get('evaluation_criteria', {}),
                additional_notes=extracted_data.get('additional_notes', '')
            )
            
            logger.info("Requirement extraction completed successfully")
            return requirements
        
        except ModelConnectionError as e:
            logger.error(f"Model connection error: {str(e)}")
            raise RequirementExtractionError(f"Model not available: {str(e)}")
        except ModelInferenceError as e:
            logger.error(f"LLM inference error: {str(e)}")
            raise RequirementExtractionError(f"LLM inference failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response: {str(e)}")
            raise RequirementExtractionError(f"Invalid response format: {str(e)}")
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise RequirementExtractionError(f"Extraction failed: {str(e)}")

    def extract_with_fallback(self, tender_doc: TenderDocument) -> StructuredRequirements:
        """
        Extract requirements with fallback to pattern-based extraction if LLM fails.
        Ensures MVP continues to work even if LLM is unavailable.
        
        Args:
            tender_doc: Parsed tender document
        
        Returns:
            StructuredRequirements: Extracted requirements (LLM or fallback)
        """
        try:
            return self.extract(tender_doc)
        except (RequirementExtractionError, ModelConnectionError) as e:
            logger.warning(f"LLM extraction failed, using fallback: {str(e)}")
            return self._extract_with_patterns(tender_doc)

    def _prepare_tender_content(self, tender_doc: TenderDocument) -> str:
        """
        Prepare tender content for extraction prompt.
        Includes title, sections, and metadata.
        
        Args:
            tender_doc: Parsed tender document
        
        Returns:
            str: Formatted tender content
        """
        parts = [f"Title: {tender_doc.title}\n"]
        
        # Add raw content (truncated if too long)
        content = tender_doc.raw_content
        if len(content) > 4000:
            content = content[:4000] + "\n[... content truncated ...]"
        parts.append(content)
        
        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.
        Handles cases where JSON is wrapped in markdown code blocks.
        
        Args:
            response: LLM response text
        
        Returns:
            Dict: Parsed JSON
            
        Raises:
            json.JSONDecodeError: If parsing fails
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end]
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end]
        
        # Extract JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        return json.loads(response)

    @staticmethod
    def _extract_with_patterns(tender_doc: TenderDocument) -> StructuredRequirements:
        """
        Fallback extraction using pattern matching.
        Used when LLM is unavailable.
        
        Args:
            tender_doc: Parsed tender document
        
        Returns:
            StructuredRequirements: Extracted requirements (pattern-based)
        """
        logger.info("Using pattern-based requirement extraction")
        
        return StructuredRequirements(
            fleet_requirements={
                'details': ', '.join(tender_doc.fleet_details.get('vehicles', [])) or 'Not specified',
                'specifications': tender_doc.fleet_details.get('specifications', [])
            },
            technical_specifications={
                'performance': tender_doc.technical_requirements.get('performance', []),
                'compliance': tender_doc.technical_requirements.get('compliance', []),
                'systems': []
            },
            scope_and_deliverables={
                'scope': tender_doc.sections.get('requirements', 'Not specified'),
                'deliverables': []
            },
            timeline_and_milestones={
                'timeline': tender_doc.timeline or 'Not specified',
                'milestones': []
            },
            budget_constraints={
                'budget': tender_doc.budget_info or 'Not specified',
                'terms': 'Not specified'
            },
            compliance_requirements={
                'certifications': [],
                'regulations': []
            },
            evaluation_criteria={
                'criteria': []
            },
            additional_notes=tender_doc.sections.get('evaluation', '')
        )


class RequirementRefiner:
    """
    Refines extracted requirements with user feedback.
    Allows iterative improvement of extracted requirements.
    """

    def __init__(self):
        """Initialize requirement refiner."""
        self.model_manager = get_model_manager()

    def refine_field(
        self,
        requirements: StructuredRequirements,
        field_name: str,
        instruction: str
    ) -> Any:
        """
        Refine a specific field in requirements based on instruction.
        
        Args:
            requirements: Current requirements
            field_name: Name of field to refine
            instruction: Instruction for refinement (e.g., "Expand with more details")
        
        Returns:
            Any: Refined field value
            
        Raises:
            RequirementExtractionError: If refinement fails
        """
        if not hasattr(requirements, field_name):
            raise RequirementExtractionError(f"Unknown field: {field_name}")
        
        current_value = getattr(requirements, field_name)
        
        prompt = f"""You are a business analyst. The user has provided the following extracted requirement:

CURRENT VALUE:
{json.dumps(current_value, indent=2)}

REFINEMENT INSTRUCTION:
{instruction}

Provide the refined version as valid JSON. Only output the refined value, no other text."""
        
        try:
            if not self.model_manager.get_current_model():
                self.model_manager.load_model()
            
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000
            )
            
            return RequirementExtractor._parse_json_response(response)
        except Exception as e:
            logger.error(f"Refinement failed: {str(e)}")
            raise RequirementExtractionError(f"Refinement failed: {str(e)}")


# Singleton instances
_extractor: Optional[RequirementExtractor] = None
_refiner: Optional[RequirementRefiner] = None


def get_requirement_extractor() -> RequirementExtractor:
    """Get or create requirement extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = RequirementExtractor()
    return _extractor


def get_requirement_refiner() -> RequirementRefiner:
    """Get or create requirement refiner instance."""
    global _refiner
    if _refiner is None:
        _refiner = RequirementRefiner()
    return _refiner
