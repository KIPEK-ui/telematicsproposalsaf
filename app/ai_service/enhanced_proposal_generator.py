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
from app.services.tender_parser import TenderDocument
from app.ai_service.requirement_extractor import StructuredRequirements
from app.ai_service.model_manager import get_model_manager


# Requirement-to-section mapping for dynamic content generation
REQUIREMENT_SECTION_MAPPING = {
    'executive_summary': {
        'requirement_fields': ['scope_and_deliverables', 'timeline_and_milestones'],
        'depth_multiplier': {'simple': 0.5, 'moderate': 1.0, 'complex': 1.5}
    },
    'technical_approach': {
        'requirement_fields': ['technical_specifications', 'scope_and_deliverables', 'compliance_requirements'],
        'depth_multiplier': {'simple': 0.7, 'moderate': 1.0, 'complex': 1.3}
    },
    'implementation': {
        'requirement_fields': ['timeline_and_milestones', 'scope_and_deliverables', 'fleet_requirements'],
        'depth_multiplier': {'simple': 0.6, 'moderate': 1.0, 'complex': 1.4}
    },
    'team_and_experience': {
        'requirement_fields': ['compliance_requirements', 'evaluation_criteria'],
        'depth_multiplier': {'simple': 0.5, 'moderate': 1.0, 'complex': 1.2}
    },
    'pricing': {
        'requirement_fields': ['budget_constraints', 'scope_and_deliverables'],
        'depth_multiplier': {'simple': 0.4, 'moderate': 1.0, 'complex': 1.1}
    },
    'support': {
        'requirement_fields': ['compliance_requirements', 'timeline_and_milestones'],
        'depth_multiplier': {'simple': 0.5, 'moderate': 1.0, 'complex': 1.3}
    },
    'timeline': {
        'requirement_fields': ['timeline_and_milestones', 'scope_and_deliverables'],
        'depth_multiplier': {'simple': 0.5, 'moderate': 1.0, 'complex': 1.4}
    }
}

COMPLEXITY_PROMPTS = {
    'simple': {
        'tone': 'Clear and concise',
        'detail': 'Focus on key benefits and deliverables',
        'approach': 'Executive summary style - high level, easy to understand',
        'evidence': 'Use selective case studies and proven track record',
        'length_guidance': 'Keep pointed and brief'
    },
    'moderate': {
        'tone': 'Professional and comprehensive',
        'detail': 'Balance strategic vision with technical details',
        'approach': 'Detailed explanation with supporting examples',
        'evidence': 'Include relevant case studies and metrics',
        'length_guidance': 'Adequate depth without overwhelming detail'
    },
    'complex': {
        'tone': 'Technical and authoritative',
        'detail': 'Comprehensive technical specifications and methodology',
        'approach': 'Deep dive into architecture, scalability, and resilience',
        'evidence': 'Extensive case studies, technical documentation, and certifications',
        'length_guidance': 'Thorough coverage of all technical aspects'
    }
}


@dataclass
class DynamicProposalContent:
    """Proposal content generated with dynamic sections."""
    sections: Dict[str, str]  # section_name -> content
    structure_design: DynamicProposalStructure
    reasoning: Dict[str, str]  # section_name -> generation reasoning
    success_factors: List[str]  # Why this structure should win
    
    @property
    def design(self) -> DynamicProposalStructure:
        """Alias for structure_design for backward compatibility."""
        return self.structure_design
    
    @property
    def design_rationale(self) -> str:
        """Get design rationale from structure."""
        return self.structure_design.design_rationale
    
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

    # Adaptive prompt that uses complexity and requirement-driven content
    ADAPTIVE_PROMPT_TEMPLATE = """Generate a proposal section for a Safaricom response.

SECTION: {section_title}
IMPORTANCE: {importance_level}
TARGET LENGTH: {suggested_length}
KEY POINTS TO COVER:
{key_points_list}

SECTION-SPECIFIC REQUIREMENTS (from tender analysis):
{section_requirements}

TENDER CONTEXT:
- Type: {tender_type}
- Industry: {industry}
- Complexity Level: {complexity}
- Buyer priorities: {priorities}
- Budget Constraints: {budget_context}
- Project Timeline: {timeline_context}

COMPLIANCE & EVALUATION CONTEXT:
- Required Compliance: {compliance_context}
- Evaluation Criteria: {evaluation_criteria}

TONE & APPROACH GUIDANCE (based on complexity level):
- Tone: {complexity_tone}
- Detail Level: {complexity_detail}
- Methodology: {complexity_approach}
- Evidence Style: {complexity_evidence}

ORGANIZATION CONTEXT:
- Name: {org_name}
- Industry: {org_industry}
- Key strengths: {org_strengths}

REFERENCE CONTEXT FROM PAST PROPOSALS (industry: {industry}):
{reference_context}

{additional_instructions}

CRITICAL REQUIREMENTS for this section:
1. EVERY statement must be grounded in the extracted tender requirements above
2. Address the section-specific requirements EXPLICITLY
3. RESPECT budget constraints: {budget_context}
4. ALIGN with project timeline: {timeline_context}
5. DEMONSTRATE compliance with: {compliance_context}
6. EMPHASIZE how this section meets evaluation criteria: {evaluation_criteria}
7. Match the complexity level ({complexity}): {complexity_detail}
8. Demonstrate how our solution meets the specific technical/operational requirements
9. Use {complexity_tone} tone and provide {complexity_evidence} level of detail
10. Avoid generic statements - everything must be tied to this specific tender

Write a compelling {section_title} section that:
1. Directly addresses EACH requirement listed above
2. Demonstrates understanding of buyer priorities and tender requirements
3. Shows Safaricom's competitive advantage relative to the specific requirements
4. Is {importance_level} to winning this tender
5. Respects and references the budget constraints
6. Aligns with the project timeline
7. Shows explicit compliance with all required standards
8. Is approximately {suggested_length}

IMPORTANT: Start directly with the section content. Do NOT include section title or preamble.
Every paragraph should reference or build on the specific requirements provided."""

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
        self.model_manager = get_model_manager()
        logger.info("EnhancedProposalGenerator initialized")

    def generate_dynamic_proposal(
        self,
        tender_doc: TenderDocument,
        org_data: Dict[str, Any],
        requirements: StructuredRequirements,
        tender_summary: Optional[str] = None,
        proposal_structure: Optional[DynamicProposalStructure] = None,
        tender_profile: Optional[Dict[str, Any]] = None,
        budget_context: Optional[str] = None,
        timeline_context: Optional[str] = None,
        evaluation_criteria: Optional[list] = None,
        compliance_context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> DynamicProposalContent:
        """
        Generate a complete proposal with dynamically designed structure.
        
        Args:
            tender_doc: Parsed tender document
            org_data: Organization information
            requirements: Extracted requirements
            tender_summary: Optional tender summary
            proposal_structure: Optional cached proposal structure (reuse from Step 3)
            tender_profile: Optional cached tender profile (reuse from Step 3)
            budget_context: Budget constraints from tender (e.g., "KES 5M - 7.5M")
            timeline_context: Project timeline/deadline from tender
            evaluation_criteria: Buyer evaluation criteria from requirements
            compliance_context: Required compliance from requirements
            progress_callback: Optional callback function(current, total, section_name)
            
        Returns:
            DynamicProposalContent: Proposal with dynamic structure
        """
        try:
            logger.info("Starting dynamic proposal generation...")
            
            # Store context for use in section generation
            self.budget_context = budget_context
            self.timeline_context = timeline_context
            self.evaluation_criteria = evaluation_criteria
            self.compliance_context = compliance_context
            
            # Step 1: Use cached structure or design proposal structure
            if proposal_structure is not None:
                logger.info("Step 1: Reusing cached proposal structure from Step 3...")
                structure = proposal_structure
                logger.info(f"  Structure reused: {len(structure.section_order)} sections")
            else:
                logger.info("Step 1: Designing proposal structure...")
                structure = self.designer.design_structure(tender_doc)
                logger.info(f"  Structure designed: {len(structure.section_order)} sections")
            
            # Step 2: Prepare generation context
            logger.info("Step 2: Preparing generation context...")
            
            # Use cached profile or structure's profile
            active_tender_profile = tender_profile if tender_profile is not None else structure.tender_profile
            
            # Convert TenderProfile to dict for compatibility
            if not isinstance(active_tender_profile, dict):
                active_tender_profile = asdict(active_tender_profile)
            
            tender_summary = tender_summary or self._summarize_requirements(
                requirements,
                active_tender_profile
            )
            
            # Step 3: Generate sections
            logger.info("Step 3: Generating proposal sections...")
            sections = {}
            reasoning = {}
            
            # Calculate total sections for progress callback
            total_sections = len(structure.section_order) + 1  # +1 for cover page
            current_section = 0
            
            # Generate cover page first
            current_section += 1
            logger.info("  Generating cover page...")
            if progress_callback:
                progress_callback(current_section, total_sections, "cover_page")
            
            cover_page = self._generate_cover_page(
                tender_doc=tender_doc,
                org_data=org_data,
                active_tender_profile=active_tender_profile,
                budget_context=budget_context,
                timeline_context=timeline_context
            )
            sections['cover_page'] = cover_page
            reasoning['cover_page'] = f"Professional cover page for {active_tender_profile.get('tender_type', 'unknown')} tender"
            
            for section_name in structure.section_order:
                current_section += 1
                logger.info(f"  Generating {section_name}...")
                if progress_callback:
                    progress_callback(current_section, total_sections, section_name)
                
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
                    active_tender_profile,
                    tender_summary
                )
                
                sections[section_name] = content
                reasoning[section_name] = f"Generated for {active_tender_profile.get('tender_type', 'unknown')} tender"
            
            # Step 4: Determine success factors
            success_factors = self._determine_success_factors(
                structure,
                requirements,
                active_tender_profile
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
            import traceback
            logger.error(f"Dynamic proposal generation failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
        tender_profile,
        tender_summary: str
    ) -> str:
        """Generate a single section with requirement-driven, complexity-aware content."""
        section_name = section_def.get('name', 'unknown')
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Convert TenderProfile to dict if needed
                if not isinstance(tender_profile, dict):
                    tender_profile = asdict(tender_profile)
                
                # Get complexity level and guidance
                complexity = tender_profile.get('complexity', 'moderate')
                complexity_info = self._get_complexity_info(complexity)
                
                # Extract section-specific requirements
                section_name_lower = section_name.lower()
                section_requirements = self._extract_section_requirements(section_name_lower, requirements)
                
                # Get RAG context for this section
                rag_context = self.designer.rag_service.get_context_for_generation(
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
                
                # Build requirement-driven prompt with complexity awareness
                prompt = self.ADAPTIVE_PROMPT_TEMPLATE.format(
                    section_title=section_def.get('title', 'Unknown'),
                    importance_level=section_def.get('importance', 'medium'),
                    suggested_length=section_def.get('suggested_length', '300 words'),
                    key_points_list=key_points_list,
                    section_requirements=section_requirements,  # ✅ Section-specific requirements
                    tender_type=tender_profile.get('tender_type', 'unknown'),
                    industry=tender_profile.get('industry', 'unknown'),
                    complexity=complexity,  # ✅ Complexity level
                    priorities=', '.join(tender_profile.get('priority_areas', [])),
                    budget_context=self.budget_context or "Not specified",
                    timeline_context=self.timeline_context or "Not specified",
                    compliance_context=str(self.compliance_context.get('compliance_standards', []) if self.compliance_context else "Not specified"),
                    evaluation_criteria=', '.join(self.evaluation_criteria) if self.evaluation_criteria else "Not specified",
                    complexity_tone=complexity_info.get('tone', 'Professional'),
                    complexity_detail=complexity_info.get('detail', 'Balance'),
                    complexity_approach=complexity_info.get('approach', 'Standard'),
                    complexity_evidence=complexity_info.get('evidence', 'Moderate'),
                    org_name=org_data.get('name', 'Our Company'),
                    org_industry=org_data.get('industry', 'Telecom'),
                    org_strengths=', '.join(org_data.get('key_strengths', [])),
                    reference_context=rag_context or "No similar proposals found",
                    additional_instructions=additional
                )
                
                # Adjust temperature based on complexity
                # Complex tenders = lower temperature (more focused), simple = slightly higher (more flexible)
                temperature = {'simple': 0.5, 'moderate': 0.4, 'complex': 0.3}.get(complexity, 0.4)
                
                # Adjust max tokens based on complexity
                max_tokens = {'simple': 1000, 'moderate': 1500, 'complex': 2000}.get(complexity, 1500)
                
                # Determine timeout based on section and complexity
                # Pricing and complex sections need more time
                section_timeout_map = {
                    'pricing': 1200,  # 20 minutes for pricing (most complex)
                    'technical_approach': 1200,  # 20 minutes for technical (complex)
                    'implementation': 1200,  # 20 minutes for implementation (complex)
                }
                base_timeout = section_timeout_map.get(section_name_lower, 900)  # 15 minutes default
                # Add extra time for retries
                timeout = base_timeout + (retry_count * 300)
                
                # Generate
                if not self.model_manager.get_current_model():
                    self.model_manager.load_model()
                
                logger.info(f"Generating {section_name} (attempt {retry_count + 1}/{max_retries + 1}, timeout: {timeout}s)...")
                
                response = self.model_manager.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    system_prompt="You are a professional proposal writer for Safaricom. Write compelling, requirement-driven proposal sections that directly address the extracted tender requirements. Every statement must be justified by the specific requirements provided."
                )
                
                logger.info(f"✅ Successfully generated {section_name}")
                return response.strip()
            
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"⚠️ {section_name} generation attempt {retry_count} failed: {str(e)}. Retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to generate {section_name} after {max_retries + 1} attempts: {str(e)}")
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
            
            logger.info(f"Refining {section_name}...")
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
                timeout=900,  # 15 minutes for section refinement
                system_prompt="You are a proposal refinement expert."
            )
            
            logger.info(f"✅ Successfully refined {section_name}")
            return response.strip()
        
        except Exception as e:
            logger.error(f"❌ Failed to refine {section_name}: {str(e)}")
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
        
        # Convert TenderProfile to dict if needed
        if not isinstance(tender_profile, dict):
            tender_profile = asdict(tender_profile)
        
        if tender_profile.get('priority_areas'):
            parts.append(f"Buyer priorities: {', '.join(tender_profile['priority_areas'])}")
        
        return '\n'.join(parts)

    def _extract_section_requirements(
        self,
        section_name: str,
        requirements: StructuredRequirements
    ) -> str:
        """
        Extract requirement fields relevant to a specific section.
        
        Args:
            section_name: Name of the section
            requirements: All extracted requirements
            
        Returns:
            str: Formatted section-specific requirements
        """
        # Get requirement fields that apply to this section
        mapping = REQUIREMENT_SECTION_MAPPING.get(
            section_name.lower(),
            {'requirement_fields': list(requirements.__dataclass_fields__.keys())}
        )
        
        relevant_reqs = []
        req_dict = asdict(requirements)
        
        for field_name in mapping['requirement_fields']:
            field_data = req_dict.get(field_name, {})
            if not field_data:
                continue
                
            # Format the requirement field
            if isinstance(field_data, dict):
                # For nested dicts like scope_and_deliverables
                for key, value in field_data.items():
                    if value and value != 'Not specified':
                        if isinstance(value, list):
                            relevant_reqs.append(f"• {key}: {', '.join(value)}")
                        else:
                            relevant_reqs.append(f"• {key}: {value}")
            elif isinstance(field_data, list):
                for item in field_data:
                    if item and item != 'Not specified':
                        relevant_reqs.append(f"• {item}")
        
        if not relevant_reqs:
            return "No specific requirements extracted for this section - use general best practices."
        
        return '\n'.join(relevant_reqs)

    def _get_complexity_info(self, complexity: str) -> Dict[str, str]:
        """Get complexity-based guidance for section generation."""
        return COMPLEXITY_PROMPTS.get(complexity, COMPLEXITY_PROMPTS['moderate'])

    def _find_section_def(
        self,
        structure: DynamicProposalStructure,
        section_name: str
    ) -> Optional[Dict[str, Any]]:
        """Find section definition by name."""
        for section in structure.base_sections + structure.custom_sections + structure.optional_sections:
            if section.name == section_name:
                return asdict(section)
        return None

    def _determine_success_factors(
        self,
        structure: DynamicProposalStructure,
        requirements: StructuredRequirements,
        tender_profile: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Determine key success factors based on structure, requirements, and tender profile."""
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
        
        # Use cached profile if available
        active_profile = tender_profile if tender_profile is not None else structure.tender_profile
        
        if isinstance(active_profile, dict):
            # Handle dict-based profile
            if active_profile.get('key_themes'):
                factors.append(
                    f"Emphasizes key themes: {', '.join(active_profile.get('key_themes', [])[:3])}"
                )
            if active_profile.get('priority_areas'):
                factors.append(
                    f"Addresses buyer priorities: {', '.join(active_profile.get('priority_areas', [])[:2])}"
                )
        else:
            # Handle object-based profile
            if hasattr(active_profile, 'key_themes') and active_profile.key_themes:
                factors.append(
                    f"Emphasizes key themes: {', '.join(active_profile.key_themes[:3])}"
                )
            if hasattr(active_profile, 'priority_areas') and active_profile.priority_areas:
                factors.append(
                    f"Addresses buyer priorities: {', '.join(active_profile.priority_areas[:2])}"
                )
        
        return factors

    def _generate_cover_page(
        self,
        tender_doc: TenderDocument,
        org_data: Dict[str, Any],
        active_tender_profile: Dict[str, Any],
        budget_context: Optional[str] = None,
        timeline_context: Optional[str] = None
    ) -> str:
        """
        Generate a professional cover page for the proposal.
        
        Args:
            tender_doc: Tender document with title and metadata
            org_data: Organization information
            active_tender_profile: Tender profile/classification
            budget_context: Budget constraints/range
            timeline_context: Project timeline/deadline
        
        Returns:
            str: Cover page content
        """
        try:
            # Ensure tender_profile is dict
            if not isinstance(active_tender_profile, dict):
                active_tender_profile = asdict(active_tender_profile)
            
            cover_page = f"""╔{'═' * 76}╗
║{' ' * 76}║
║{'PROPOSAL FOR TENDER'.center(76)}║
║{' ' * 76}║
║{'─' * 76}║
║{' ' * 76}║
║{tender_doc.title.center(76)}║
║{' ' * 76}║
║{'─' * 76}║
║{' ' * 76}║

Tender Reference: {tender_doc.tender_no or '[Not specified]'}

Submitted by:
{org_data.get('name', 'Safaricom')}
{org_data.get('address', '')}

Contact Information:
Email: {org_data.get('contact_email', '')}
Phone: {org_data.get('contact_phone', '')}

Submission Date: {self._get_submission_date()}

Service Category: {active_tender_profile.get('tender_type', 'Unknown').replace('_', ' ').title()}
Industry: {active_tender_profile.get('industry', 'Unknown').replace('_', ' ').title()}

─────────────────────────────────────────────────────────────────────────────

KEY TENDER CONSTRAINTS:
"""
            
            # Add budget if available
            if budget_context:
                cover_page += f"Budget Range: {budget_context}\n"
            
            # Add timeline if available
            if timeline_context:
                cover_page += f"Timeline: {timeline_context}\n"
            
            # Add bid validity if available
            if tender_doc.bid_validity:
                cover_page += f"Bid Validity: {tender_doc.bid_validity}\n"
            
            cover_page += f"""
╚{'═' * 76}╝
"""
            return cover_page.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate cover page: {str(e)}")
            return "PROPOSAL - COVER PAGE"

    def _get_submission_date(self) -> str:
        """Get current date in formatted string."""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")
