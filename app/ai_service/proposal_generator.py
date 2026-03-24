"""
Proposal Generator Service
Orchestrates the proposal generation workflow using LLM and RAG context.
Generates multi-section proposals with iterative refinement capability.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

from app.ai_service.model_manager import get_model_manager, ModelInferenceError
from app.ai_service.rag_service import get_rag_service, RAGError
from app.ai_service.requirement_extractor import StructuredRequirements


class ProposalSection(Enum):
    """Standard proposal sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_APPROACH = "technical_approach"
    FLEET_DETAILS = "fleet_details"
    IMPLEMENTATION_TIMELINE = "implementation_timeline"
    PRICING = "pricing"
    COMPLIANCE_ASSURANCE = "compliance_assurance"
    TERMS_CONDITIONS = "terms_conditions"


@dataclass
class ProposalContent:
    """Complete proposal content."""
    executive_summary: str
    technical_approach: str
    fleet_details: str
    implementation_timeline: str
    pricing: str
    compliance_assurance: str
    terms_conditions: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_section(self, section: ProposalSection) -> str:
        """Get specific section content."""
        section_map = {
            ProposalSection.EXECUTIVE_SUMMARY: self.executive_summary,
            ProposalSection.TECHNICAL_APPROACH: self.technical_approach,
            ProposalSection.FLEET_DETAILS: self.fleet_details,
            ProposalSection.IMPLEMENTATION_TIMELINE: self.implementation_timeline,
            ProposalSection.PRICING: self.pricing,
            ProposalSection.COMPLIANCE_ASSURANCE: self.compliance_assurance,
            ProposalSection.TERMS_CONDITIONS: self.terms_conditions,
        }
        return section_map.get(section, "")
    
    def set_section(self, section: ProposalSection, content: str) -> None:
        """Update specific section content."""
        if section == ProposalSection.EXECUTIVE_SUMMARY:
            self.executive_summary = content
        elif section == ProposalSection.TECHNICAL_APPROACH:
            self.technical_approach = content
        elif section == ProposalSection.FLEET_DETAILS:
            self.fleet_details = content
        elif section == ProposalSection.IMPLEMENTATION_TIMELINE:
            self.implementation_timeline = content
        elif section == ProposalSection.PRICING:
            self.pricing = content
        elif section == ProposalSection.COMPLIANCE_ASSURANCE:
            self.compliance_assurance = content
        elif section == ProposalSection.TERMS_CONDITIONS:
            self.terms_conditions = content


class ProposalGenerationError(Exception):
    """Raised when proposal generation fails."""
    pass


class ProposalGenerator:
    """
    Generates proposals from tender requirements.
    Features:
    - Multi-section generation
    - RAG-enhanced context injection
    - Iterative refinement
    - Chat history for consistency
    """

    # Section generation prompts
    SECTION_PROMPTS = {
        ProposalSection.EXECUTIVE_SUMMARY: """Generate a compelling executive summary for a Safaricom proposal.

Organization: {org_name}
Industry: {org_industry}

TENDER REQUIREMENTS:
{requirements_summary}

REFERENCE PROPOSALS:
{reference_context}

The executive summary should:
1. Clearly state Safaricom's understanding of the tender
2. Highlight key value propositions (3-4 bullets)
3. Show competitive advantage
4. Be persuasive but professional
5. 150-200 words

Start directly with the summary, no preamble.""",

        ProposalSection.TECHNICAL_APPROACH: """Generate a technical approach section for a Safaricom proposal.

Organization: {org_name}
Technical Requirements: {tech_requirements}

IMPLEMENTATION STRATEGY:
- List the methodology and approach
- Describe technology stack/solutions
- Explain integration approach
- Address scalability and performance

REFERENCE PROPOSALS:
{reference_context}

Write a 300-400 word technical approach that:
1. Demonstrates technical expertise
2. Shows understanding of requirements
3. Provides detailed implementation steps
4. Addresses potential risks
5. Is specific and actionable

Start directly, no preamble.""",

        ProposalSection.FLEET_DETAILS: """Generate a fleet and equipment details section.

Fleet Requirements: {fleet_requirements}

SPECIFICATIONS:
- List all required vehicles/equipment
- Detail specifications per item
- Explain maintenance strategy
- Describe fleet management approach

REFERENCE PROPOSALS:
{reference_context}

Write a 250-300 word section that:
1. Details the exact fleet composition
2. Justifies each vehicle/equipment choice
3. Shows compliance with requirements
4. Explains maintenance and support
5. Is concrete and itemized

Start directly, no preamble.""",

        ProposalSection.IMPLEMENTATION_TIMELINE: """Generate an implementation timeline section.

Project Timeline Requirement: {timeline_requirement}
Key Milestones: {milestones}

Create a detailed timeline with:
1. Project phases and duration
2. Key milestones and deliverables
3. Critical path items
4. Resource allocation per phase
5. Risk mitigation timeline

REFERENCE PROPOSALS:
{reference_context}

Write a professional 200-250 word timeline section with a structured table/list format.
Start directly, no preamble.""",

        ProposalSection.PRICING: """Generate pricing and billing section.

Budget Context: {budget_context}
Pricing Considerations: {cost_factors}

Create a professional pricing section that:
1. Provides cost breakdown by category
2. Explains value for money
3. Details payment terms
4. Includes any volume discounts
5. Addresses cost efficiency

REFERENCE PROPOSALS:
{reference_context}

Write a 200-250 word section explaining pricing strategy.
NOTE: Use placeholder estimates like "Starting at $[X]" where appropriate.
Start directly, no preamble.""",

        ProposalSection.COMPLIANCE_ASSURANCE: """Generate a compliance and assurance section.

Compliance Requirements: {compliance_requirements}
Certifications/Standards: {standards}

Write a section demonstrating:
1. Full compliance with all requirements
2. Relevant certifications held
3. Industry standards adherence
4. Quality assurance processes
5. Audit and oversight mechanisms

REFERENCE PROPOSALS:
{reference_context}

Write a professional 250-300 word compliance section.
Start directly, no preamble.""",

        ProposalSection.TERMS_CONDITIONS: """Generate standard terms and conditions.

Key Terms Considerations: {terms_context}

Write a professional terms & conditions section covering:
1. Service levels and guarantees
2. Support hours and response times
3. Contract duration and renewal
4. Liability and insurance
5. Confidentiality and data handling

REFERENCE PROPOSALS:
{reference_context}

Write a 200-250 word terms & conditions section.
Note: Use standard industry terms where applicable.
Start directly, no preamble.""",
    }

    def __init__(self):
        """Initialize proposal generator."""
        self.model_manager = get_model_manager()
        self.rag_service = get_rag_service()
        self.chat_history: List[Dict[str, str]] = []
        logger.info("ProposalGenerator initialized")

    def generate_proposal(
        self,
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        tender_summary: Optional[str] = None
    ) -> ProposalContent:
        """
        Generate a complete proposal from requirements.
        
        Args:
            org_data: Organization information (name, industry, contact)
            requirements: Structured requirements from tender
            tender_summary: Optional summary of the tender for context
        
        Returns:
            ProposalContent: Generated proposal with all sections
            
        Raises:
            ProposalGenerationError: If generation fails
        """
        try:
            logger.info(f"Starting proposal generation for {org_data.get('name', 'Unknown')}")
            
            # Ensure model is loaded
            if not self.model_manager.get_current_model():
                logger.info("Loading model...")
                self.model_manager.load_model()
            
            # Get RAG context
            industry = org_data.get('industry')
            rag_context = self.rag_service.get_context_for_generation(
                tender_summary or self._summarize_requirements(requirements),
                industry=industry,
                max_examples=2
            )
            
            # Reset chat history for new proposal
            self.chat_history = []
            
            # Generate each section
            sections_data = {}
            for section in ProposalSection:
                logger.info(f"Generating {section.value}...")
                sections_data[section] = self._generate_section(
                    section,
                    org_data,
                    requirements,
                    rag_context
                )
            
            # Create proposal content
            proposal = ProposalContent(
                executive_summary=sections_data[ProposalSection.EXECUTIVE_SUMMARY],
                technical_approach=sections_data[ProposalSection.TECHNICAL_APPROACH],
                fleet_details=sections_data[ProposalSection.FLEET_DETAILS],
                implementation_timeline=sections_data[ProposalSection.IMPLEMENTATION_TIMELINE],
                pricing=sections_data[ProposalSection.PRICING],
                compliance_assurance=sections_data[ProposalSection.COMPLIANCE_ASSURANCE],
                terms_conditions=sections_data[ProposalSection.TERMS_CONDITIONS]
            )
            
            logger.info("Proposal generation completed successfully")
            return proposal
        
        except Exception as e:
            logger.error(f"Proposal generation failed: {str(e)}")
            raise ProposalGenerationError(f"Failed to generate proposal: {str(e)}")

    def refine_section(
        self,
        proposal: ProposalContent,
        section: ProposalSection,
        instruction: str,
        org_data: Dict[str, Any],
        requirements: Optional[StructuredRequirements] = None
    ) -> str:
        """
        Refine a specific section of the proposal.
        
        Args:
            proposal: Current proposal
            section: Section to refine
            instruction: Refinement instruction (e.g., "Make more technical")
            org_data: Organization data for context
            requirements: Optional requirements for context
        
        Returns:
            str: Refined section content
            
        Raises:
            ProposalGenerationError: If refinement fails
        """
        try:
            current_section = proposal.get_section(section)
            
            prompt = f"""You are a proposal writing expert. Refine the following proposal section.

CURRENT SECTION:
{current_section}

REFINEMENT INSTRUCTION:
{instruction}

ORGANIZATION CONTEXT:
- Name: {org_data.get('name', 'Unknown')}
- Industry: {org_data.get('industry', 'Unknown')}

Provide the refined section maintaining professional tone and relevance.
Start directly with refined content, no preamble."""
            
            if not self.model_manager.get_current_model():
                self.model_manager.load_model()
            
            refined_content = self.model_manager.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Update chat history for consistency
            self.chat_history.append({"role": "user", "content": instruction})
            self.chat_history.append({"role": "assistant", "content": refined_content})
            
            logger.info(f"Section {section.value} refined successfully")
            return refined_content
        
        except Exception as e:
            logger.error(f"Section refinement failed: {str(e)}")
            raise ProposalGenerationError(f"Refinement failed: {str(e)}")

    def _generate_section(
        self,
        section: ProposalSection,
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        rag_context: str
    ) -> str:
        """
        Generate a single proposal section.
        
        Args:
            section: Section to generate
            org_data: Organization data
            requirements: Structured requirements
            rag_context: RAG context from past proposals
        
        Returns:
            str: Generated section content
        """
        # Get section prompt
        prompt_template = self.SECTION_PROMPTS.get(section)
        if not prompt_template:
            logger.warning(f"No prompt template for {section.value}")
            return ""
        
        # Format prompt with data
        formatted_prompt = prompt_template.format(
            org_name=org_data.get('name', 'Safaricom'),
            org_industry=org_data.get('industry', 'Telecommunications'),
            requirements_summary=self._summarize_requirements(requirements),
            tech_requirements=self._format_tech_requirements(requirements.technical_specifications),
            fleet_requirements=self._format_fleet_requirements(requirements.fleet_requirements),
            timeline_requirement=requirements.timeline_and_milestones.get('timeline', 'Not specified'),
            milestones=self._format_milestones(requirements.timeline_and_milestones),
            budget_context=requirements.budget_constraints.get('range', 'Competitive market rate'),
            cost_factors=self._format_cost_factors(requirements.budget_constraints),
            compliance_requirements=self._format_compliance(requirements.compliance_requirements),
            standards=self._format_standards(requirements.compliance_requirements),
            terms_context=self._format_terms(requirements),
            reference_context=rag_context
        )
        
        try:
            # Generate section
            section_content = self.model_manager.generate(
                prompt=formatted_prompt,
                temperature=0.8,
                max_tokens=1200
            )
            
            # Update chat history
            self.chat_history.append({"role": "assistant", "content": section_content})
            
            return section_content.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate {section.value}: {str(e)}")
            return f"[Error generating {section.value}]\n{str(e)}"

    @staticmethod
    def _summarize_requirements(requirements: StructuredRequirements) -> str:
        """Create brief summary of requirements for prompt."""
        parts = []
        
        if requirements.scope_and_deliverables.get('scope'):
            parts.append(f"Scope: {requirements.scope_and_deliverables['scope']}")
        
        fleet = requirements.fleet_requirements.get('details', '')
        if fleet:
            parts.append(f"Fleet: {fleet}")
        
        timeline = requirements.timeline_and_milestones.get('timeline', '')
        if timeline:
            parts.append(f"Timeline: {timeline}")
        
        return "\n".join(parts) or "General proposal for Safaricom services"

    @staticmethod
    def _format_tech_requirements(tech_specs: Dict[str, Any]) -> str:
        """Format technical requirements for prompt."""
        parts = []
        for key, values in tech_specs.items():
            if values:
                parts.append(f"{key}: {', '.join(values) if isinstance(values, list) else str(values)}")
        return "\n".join(parts) or "Standard technical requirements"

    @staticmethod
    def _format_fleet_requirements(fleet: Dict[str, Any]) -> str:
        """Format fleet requirements for prompt."""
        parts = []
        if fleet.get('details'):
            parts.append(f"Details: {fleet['details']}")
        if fleet.get('specifications'):
            parts.append(f"Specs: {', '.join(fleet['specifications'])}")
        return "\n".join(parts) or "Standard fleet specifications"

    @staticmethod
    def _format_milestones(timeline: Dict[str, Any]) -> str:
        """Format milestones for prompt."""
        if timeline.get('milestones'):
            return ", ".join(timeline['milestones'])
        return "Key milestones to be defined in contract"

    @staticmethod
    def _format_cost_factors(budget: Dict[str, Any]) -> str:
        """Format cost factors for prompt."""
        parts = []
        if budget.get('payment_terms'):
            parts.append(f"Terms: {budget['payment_terms']}")
        if budget.get('range'):
            parts.append(f"Range: {budget['range']}")
        return "\n".join(parts) or "Competitive pricing based on scope"

    @staticmethod
    def _format_compliance(compliance: Dict[str, Any]) -> str:
        """Format compliance requirements."""
        parts = []
        if compliance.get('certifications'):
            parts.append(f"Certifications: {', '.join(compliance['certifications'])}")
        if compliance.get('regulations'):
            parts.append(f"Regulations: {', '.join(compliance['regulations'])}")
        return "\n".join(parts) or "Industry standard compliance"

    @staticmethod
    def _format_standards(compliance: Dict[str, Any]) -> str:
        """Format standards for prompt."""
        certs = compliance.get('certifications', [])
        if isinstance(certs, str):
            certs = [certs]
        return ", ".join(certs) if certs else "ISO 9001 and industry standards"

    @staticmethod
    def _format_terms(requirements: StructuredRequirements) -> str:
        """Format terms context."""
        timeline = requirements.timeline_and_milestones.get('timeline', '12 months')
        return f"Standard service agreement with {timeline} engagement period"


# Singleton instance
_proposal_generator: Optional[ProposalGenerator] = None


def get_proposal_generator() -> ProposalGenerator:
    """Get or create proposal generator instance."""
    global _proposal_generator
    if _proposal_generator is None:
        _proposal_generator = ProposalGenerator()
    return _proposal_generator
