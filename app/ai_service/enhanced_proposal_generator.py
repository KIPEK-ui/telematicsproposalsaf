"""
Enhanced Proposal Generator with Dynamic Sections
Generates proposals with intelligently designed structure based on tender analysis.
Learns from past proposals to improve accuracy and competitiveness.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

from app.ai_service.dynamic_proposal_designer import (
    DynamicProposalDesigner,
    TenderClassifier,
    DynamicProposalStructure
)
from app.ai_service.proposal_generator import ProposalGenerator
from app.services.tender_parser import TenderDocument
from app.ai_service.requirement_extractor import StructuredRequirements
from app.ai_service.model_manager import get_model_manager


@dataclass
class DynamicProposalContent:
    """Proposal content generated with dynamic sections."""
    sections: Dict[str, str]  # section_name -> content
    structure_design: DynamicProposalStructure
    reasoning: Dict[str, str]  # section_name -> generation reasoning
    success_factors: List[str]  # Why this structure should win
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sections': self.sections,
            'structure_design': {
                'base_sections': [asdict(s) for s in self.structure_design.base_sections],
                'custom_sections': [asdict(s) for s in self.structure_design.custom_sections],
                'optional_sections': [asdict(s) for s in self.structure_design.optional_sections],
                'section_order': self.structure_design.section_order,
                'design_rationale': self.structure_design.design_rationale,
                'estimated_total_length': self.structure_design.estimated_total_length
            },
            'reasoning': self.reasoning,
            'success_factors': self.success_factors
        }


class EnhancedProposalGenerator:
    """Generates proposals with dynamic, tender-optimized structure."""

    # Section generation prompts that adapt based on section definition
    ADAPTIVE_PROMPT_TEMPLATE = """Generate a proposal section for a Safaricom response.

SECTION: {section_title}
IMPORTANCE: {importance_level}
TARGET LENGTH: {suggested_length}
KEY POINTS TO COVER:
{key_points_list}

TENDER CONTEXT:
- Type: {tender_type}
- Industry: {industry}
- Buyer priorities: {priorities}
- Tender requirements: {requirements_summary}

ORGANIZATION CONTEXT:
- Name: {org_name}
- Industry: {org_industry}
- Key strengths: {org_strengths}

REFERENCE CONTEXT FROM PAST PROPOSALS:
{reference_context}

{additional_instructions}

Write a compelling {section_title} section that:
1. Directly addresses the key points listed above
2. Demonstrates understanding of buyer priorities
3. Shows Safaricom's competitive advantage
4. Is {importance_level} to winning this tender
5. Is approximately {suggested_length}

IMPORTANT: Start directly with the section content. Do NOT include section title, no preamble."""

    CUSTOM_SECTION_ADDITIONAL = {
        'security': """
- Explain security measures and frameworks
- Address compliance with security standards
- Show data protection approach""",
        
        'fleet_management': """
- Detail fleet composition and specifications
- Explain fleet management approach
- Show maintenance and support strategy""",
        
        'support_sla': """
- Define support hours and availability
- Specify SLA metrics and response times
- Explain escalation procedures""",
        
        'scalability': """
- Explain how solution scales with growth
- Show elasticity and performance capacity
- Describe upgrade path""",
        
        'innovation': """
- Highlight innovation approach and research
- Describe future enhancements planned
- Show commitment to continuous improvement""",
        
        'case_studies': """
- Provide relevant case study examples
- Show measurable success metrics
- Include client testimonials if available""",
        
        'risk_mitigation': """
- Identify key risks and mitigation strategies
- Explain contingency plans
- Show reliability through redundancy""",
    }

    def __init__(self):
        """Initialize enhanced proposal generator."""
        self.designer = DynamicProposalDesigner()
        self.generator = ProposalGenerator()
        self.model_manager = get_model_manager()
        logger.info("EnhancedProposalGenerator initialized")

    def generate_dynamic_proposal(
        self,
        tender_doc: TenderDocument,
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        tender_summary: Optional[str] = None
    ) -> DynamicProposalContent:
        """
        Generate a complete proposal with dynamically designed structure.
        
        Args:
            tender_doc: Parsed tender document
            org_data: Organization information
            requirements: Extracted requirements
            tender_summary: Optional tender summary
            
        Returns:
            DynamicProposalContent: Proposal with dynamic structure
        """
        try:
            logger.info("Starting dynamic proposal generation...")
            
            # Step 1: Design proposal structure
            logger.info("Step 1: Designing proposal structure...")
            structure = self.designer.design_structure(tender_doc)
            logger.info(f"  Structure designed: {len(structure.section_order)} sections")
            
            # Step 2: Prepare generation context
            logger.info("Step 2: Preparing generation context...")
            tender_summary = tender_summary or self._summarize_requirements(
                requirements,
                structure.tender_profile
            )
            
            # Step 3: Generate sections
            logger.info("Step 3: Generating proposal sections...")
            sections = {}
            reasoning = {}
            
            for section_name in structure.section_order:
                logger.info(f"  Generating {section_name}...")
                
                # Find section definition
                section_def = self._find_section_def(structure, section_name)
                if not section_def:
                    logger.warning(f"  Section {section_name} not found, skipping")
                    continue
                
                # Generate section
                content = self._generate_section(
                    section_def,
                    org_data,
                    requirements,
                    structure.tender_profile,
                    tender_summary
                )
                
                sections[section_name] = content
                reasoning[section_name] = f"Generated for {structure.tender_profile.tender_type} tender"
            
            # Step 4: Determine success factors
            success_factors = self._determine_success_factors(
                structure,
                requirements
            )
            
            # Create proposal content
            proposal = DynamicProposalContent(
                sections=sections,
                structure_design=structure,
                reasoning=reasoning,
                success_factors=success_factors
            )
            
            logger.info("Dynamic proposal generation completed successfully")
            return proposal
        
        except Exception as e:
            logger.error(f"Dynamic proposal generation failed: {str(e)}")
            raise

    def get_proposal_structure_preview(
        self,
        tender_doc: TenderDocument
    ) -> DynamicProposalStructure:
        """
        Get preview of proposed structure without generating content.
        Useful for UI to show structure before full generation.
        
        Args:
            tender_doc: Tender document
            
        Returns:
            DynamicProposalStructure: Designed structure
        """
        return self.designer.design_structure(tender_doc)

    def regenerate_section(
        self,
        section_name: str,
        section_def: Dict[str, Any],
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        tender_profile: Dict[str, Any],
        current_content: str,
        refinement_instruction: Optional[str] = None
    ) -> str:
        """
        Regenerate a single section with optional refinement.
        
        Args:
            section_name: Name of section
            section_def: Section definition (title, key points, etc.)
            org_data: Organization data
            requirements: Tender requirements
            tender_profile: Classified tender profile
            current_content: Current section content (for refinement)
            refinement_instruction: Optional refinement instruction
            
        Returns:
            str: Regenerated section content
        """
        if refinement_instruction:
            return self._refine_section(
                section_name,
                current_content,
                refinement_instruction,
                org_data
            )
        else:
            return self._generate_section(
                section_def,
                org_data,
                requirements,
                tender_profile,
                ""
            )

    def _generate_section(
        self,
        section_def: Dict[str, Any],
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        tender_profile: Dict[str, Any],
        tender_summary: str
    ) -> str:
        """Generate a single section with context."""
        try:
            # Get RAG context for this section
            rag_context = self.generator.rag_service.get_context_for_generation(
                f"{section_def.get('name')} for {tender_profile.get('tender_type')}",
                industry=tender_profile.get('industry'),
                max_examples=2
            )
            
            # Get additional instructions for custom sections
            additional = self.CUSTOM_SECTION_ADDITIONAL.get(
                section_def.get('name'),
                ""
            )
            
            # Build key points list
            key_points_list = '\n'.join([
                f"- {point}" for point in section_def.get('key_points', [])
            ])
            
            # Build prompt
            prompt = self.ADAPTIVE_PROMPT_TEMPLATE.format(
                section_title=section_def.get('title', 'Unknown'),
                importance_level=section_def.get('importance', 'medium'),
                suggested_length=section_def.get('suggested_length', '300 words'),
                key_points_list=key_points_list,
                tender_type=tender_profile.get('tender_type', 'unknown'),
                industry=tender_profile.get('industry', 'unknown'),
                priorities=', '.join(tender_profile.get('priority_areas', [])),
                requirements_summary=tender_summary,
                org_name=org_data.get('name', 'Our Company'),
                org_industry=org_data.get('industry', 'Telecom'),
                org_strengths=', '.join(org_data.get('key_strengths', [])),
                reference_context=rag_context or "No similar proposals found",
                additional_instructions=additional
            )
            
            # Generate
            if not self.model_manager.get_current_model():
                self.model_manager.load_model()
            
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=1500,
                system_prompt="You are a professional proposal writer for Safaricom. Write compelling, accurate proposal sections."
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate {section_def.get('name')}: {str(e)}")
            raise

    def _refine_section(
        self,
        section_name: str,
        current_content: str,
        instruction: str,
        org_data: Dict[str, Any]
    ) -> str:
        """Refine an existing section."""
        try:
            prompt = f"""You are a proposal refinement expert. Refine the following proposal section 
based on the instruction provided.

CURRENT SECTION ({section_name}):
{current_content}

REFINEMENT INSTRUCTION:
{instruction}

ORGANIZATION: {org_data.get('name', 'Safaricom')}

Provide the refined section maintaining professional tone and relevance.
Start directly with refined content, no preamble."""
            
            if not self.model_manager.get_current_model():
                self.model_manager.load_model()
            
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
                system_prompt="You are a proposal refinement expert."
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Failed to refine section: {str(e)}")
            raise

    def _summarize_requirements(
        self,
        requirements: StructuredRequirements,
        tender_profile: Dict[str, Any]
    ) -> str:
        """Create a summary of requirements for prompt context."""
        parts = []
        
        if requirements.scope_and_deliverables.get('scope'):
            parts.append(f"Scope: {requirements.scope_and_deliverables['scope']}")
        
        if requirements.technical_specifications:
            specs = requirements.technical_specifications
            if isinstance(specs, dict):
                if specs.get('performance'):
                    parts.append(f"Performance: {', '.join(specs['performance'][:3])}")
                if specs.get('compliance'):
                    parts.append(f"Compliance: {', '.join(specs['compliance'][:3])}")
        
        if requirements.budget_constraints.get('range'):
            parts.append(f"Budget: {requirements.budget_constraints['range']}")
        
        if tender_profile.get('priority_areas'):
            parts.append(f"Buyer priorities: {', '.join(tender_profile['priority_areas'])}")
        
        return '\n'.join(parts)

    @staticmethod
    def _find_section_def(structure: DynamicProposalStructure, section_name: str) -> Optional[Dict[str, Any]]:
        """Find section definition by name."""
        for section in structure.base_sections + structure.custom_sections + structure.optional_sections:
            if section.name == section_name:
                return asdict(section)
        return None

    @staticmethod
    def _determine_success_factors(
        structure: DynamicProposalStructure,
        requirements: StructuredRequirements
    ) -> List[str]:
        """Determine key success factors for this proposal."""
        factors = [
            "Structure matches tender evaluation criteria",
            "Custom sections address buyer priorities",
            f"Covers all {len(structure.section_order)} required sections",
        ]
        
        if structure.custom_sections:
            factors.append(
                f"Includes custom sections ({', '.join([s.title for s in structure.custom_sections])}) "
                "addressing specific tender requirements"
            )
        
        if structure.tender_profile.key_themes:
            factors.append(
                f"Emphasizes key themes: {', '.join(structure.tender_profile.key_themes[:3])}"
            )
        
        return factors
