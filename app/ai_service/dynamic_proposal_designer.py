"""
Dynamic Proposal Designer
Intelligently designs proposal structure based on tender analysis and past proposals.
Enables learning from successful proposals to improve accuracy and competitiveness.
"""

import logging
import json
import hashlib
from functools import lru_cache
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

from app.services.tender_parser import TenderDocument
from app.ai_service.model_manager import get_model_manager
from app.ai_service.rag_service import get_rag_service


# ========================
# Data Classes (Moved Before Functions)
# ========================

@dataclass
class TenderProfile:
    """Classification of a tender document."""
    tender_type: str  # "cloud_services", "fleet_management", "tech_solutions", etc.
    industry: str  # "telecom", "transport", "logistics", etc.
    complexity: str  # "simple", "moderate", "complex"
    key_themes: List[str]  # ["security", "cost_efficiency", "scalability"]
    evaluation_focus: List[str]  # How proposal will be evaluated
    priority_areas: List[str]  # What buyer cares most about
    estimated_value: Optional[str] = None


# ========================
# Memoization Utilities
# ========================

@lru_cache(maxsize=128)
def _get_content_hash_classification(content_hash: str) -> Optional[TenderProfile]:
    """
    Helper to cache classification results by content hash.
    ✅ Uses lru_cache for deterministic results.
    
    Args:
        content_hash: MD5 hash of tender content
        
    Returns:
        Optional[TenderProfile]: Cached profile or None
    """
    # This is just a cache key holder; actual classification happens in classify()
    return None


def _hash_tender_content(content: str) -> str:
    """Generate consistent hash for tender content."""
    return hashlib.md5(content.encode()).hexdigest()


@dataclass
class ProposalSectionDef:
    """Definition of a proposal section."""
    name: str  # Section identifier
    title: str  # Display title
    importance: str  # "critical", "high", "medium", "low"
    suggested_length: str  # e.g., "300-400 words"
    key_points: List[str]  # Points to cover
    is_custom: bool = False  # Whether this is a custom section
    reason: Optional[str] = None  # Why this section was recommended
    similar_past_examples: Optional[List[str]] = None  # References to past proposals
    target_length: int = 400  # Target word count (derived from suggested_length or set dynamically)


@dataclass
class DynamicProposalStructure:
    """Dynamically designed proposal structure."""
    base_sections: List[ProposalSectionDef]  # Always include
    custom_sections: List[ProposalSectionDef]  # Added based on tender
    optional_sections: List[ProposalSectionDef]  # Nice to have
    section_order: List[str]  # Recommended reading order
    design_rationale: str  # Why this structure was chosen
    tender_profile: TenderProfile
    estimated_total_length: str  # e.g., "2500-3500 words"
    
    @property
    def section_definitions(self) -> Dict[str, ProposalSectionDef]:
        """Create dictionary mapping section names to definitions."""
        section_dict = {}
        for section in self.base_sections + self.custom_sections + self.optional_sections:
            section_dict[section.name] = section
        return section_dict
    
    @property
    def estimated_word_count_min(self) -> int:
        """Extract minimum word count from estimated_total_length."""
        try:
            parts = self.estimated_total_length.replace('words', '').replace('-', ',').split(',')
            return int(parts[0].strip())
        except (ValueError, IndexError):
            return 2000
    
    @property
    def estimated_word_count_max(self) -> int:
        """Extract maximum word count from estimated_total_length."""
        try:
            parts = self.estimated_total_length.replace('words', '').replace('-', ',').split(',')
            return int(parts[1].strip()) if len(parts) > 1 else int(parts[0].strip())
        except (ValueError, IndexError):
            return 3500


class TenderClassifier:
    """Classifies tender documents by type, industry, and complexity."""

    CLASSIFIER_PROMPT = """You are an expert tender analyst. Analyze this tender document and classify it precisely.

TENDER DOCUMENT:
---
{tender_content}
---

INSTRUCTIONS:
1. Extract the primary service/product type being tendered
2. Identify the industry or sector
3. Assess scope and complexity
4. List key themes mentioned (security, cost, innovation, etc.)
5. Identify evaluation criteria from requirements
6. Infer buyer priorities from the tender language
7. Estimate project budget if mentioned

Return ONLY a valid JSON object (no markdown, no code blocks, no extra text):
{{
    "tender_type": "Select from: service_management, fleet_management, cloud_services, telecommunications, infrastructure, consulting, managed_services, logistics, supply_chain, healthcare, government",
    "industry": "Select from: telecom, transport, logistics, government, finance, healthcare, retail, energy, manufacturing",
    "complexity": "simple|moderate|complex",
    "key_themes": ["theme1", "theme2", "theme3"],
    "evaluation_focus": ["criterion1", "criterion2", "criterion3"],
    "priority_areas": ["priority1", "priority2", "priority3"],
    "estimated_value": "budget or \"not specified\""
}}"""

    # Keyword patterns for fallback classification
    INDUSTRY_KEYWORDS = {
        'telecom': ['telecommunications', 'mobile', 'network', 'signal', 'wireless', 'safaricom', 'vodafone'],
        'transport': ['fleet', 'vehicles', 'logistics', 'transportation', 'vehicle', 'drivers', 'tracking'],
        'logistics': ['logistics', 'supply chain', 'warehousing', 'distribution', 'shipping'],
        'government': ['government', 'public', 'ministry', 'agency', 'state'],
        'healthcare': ['healthcare', 'medical', 'hospital', 'health', 'clinic'],
        'finance': ['finance', 'banking', 'financial', 'insurance'],
        'energy': ['energy', 'power', 'electricity', 'solar', 'fuel']
    }
    
    TYPE_KEYWORDS = {
        'fleet_management': ['fleet', 'vehicles', 'tracking', 'driver'],
        'cloud_services': ['cloud', 'aws', 'azure', 'hosting', 'server'],
        'telecommunications': ['telecom', 'mobile', 'network', 'connectivity'],
        'managed_services': ['managed', 'managed service', 'msp'],
        'consulting': ['consulting', 'advisory', 'expertise', 'specialist'],
        'infrastructure': ['infrastructure', 'build', 'construction', 'deploy']
    }

    @staticmethod
    def classify(tender_doc: TenderDocument, progress_callback: Optional[callable] = None) -> TenderProfile:
        """
        Classify a tender document with LLM and fallback to pattern matching.
        
        Args:
            tender_doc: TenderDocument to classify
            progress_callback: Optional callback function(step, total, step_name)
            
        Returns:
            TenderProfile: Classification results
        """
        try:
            model_manager = get_model_manager()
            if not model_manager.get_current_model():
                model_manager.load_model()
            
            # Prepare content (truncate if too long)
            content = tender_doc.raw_content
            if len(content) > 4000:
                content = content[:4000] + "\n[... truncated ...]"
            
            prompt = TenderClassifier.CLASSIFIER_PROMPT.format(tender_content=content)
            
            logger.info("Classifying tender document with LLM...")
            if progress_callback:
                progress_callback(1, 3, "classifying_tender")
            
            response = model_manager.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=500,
                system_prompt="You are a tender classification expert. Return ONLY valid JSON, no other text."
            )
            
            logger.debug(f"LLM Response: {response[:200]}...")
            
            # Parse JSON response
            data = TenderClassifier._parse_json_response(response)
            
            # If parsing failed or returned empty, use pattern-based fallback
            if not data or data.get('tender_type') == 'unknown':
                logger.warning("LLM classification returned unknown, using pattern-based fallback...")
                data = TenderClassifier._fallback_classification(tender_doc.raw_content, data or {})
            
            profile = TenderProfile(
                tender_type=data.get('tender_type', 'unknown'),
                industry=data.get('industry', 'unknown'),
                complexity=data.get('complexity', 'moderate'),
                key_themes=data.get('key_themes', []),
                evaluation_focus=data.get('evaluation_focus', []),
                priority_areas=data.get('priority_areas', []),
                estimated_value=data.get('estimated_value')
            )
            
            logger.info(f"Tender classified: {profile.tender_type} ({profile.industry}) - Complexity: {profile.complexity}")
            return profile
        
        except Exception as e:
            logger.error(f"Tender classification failed: {str(e)}", exc_info=True)
            # Return default profile with fallback pattern matching
            logger.info("Using pattern-based fallback classification...")
            try:
                return TenderClassifier._create_default_profile(tender_doc.raw_content)
            except:
                return TenderProfile(
                    tender_type="unknown",
                    industry="unknown",
                    complexity="moderate",
                    key_themes=[],
                    evaluation_focus=[],
                    priority_areas=[]
                )

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with enhanced error handling."""
        try:
            # Remove markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()
            
            # Find JSON object boundaries
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                response = response[start:end].strip()
                logger.debug(f"Extracted JSON: {response[:100]}...")
                return json.loads(response)
            else:
                logger.warning("No JSON object found in response")
                return {}
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.debug(f"Response text: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {}

    @staticmethod
    def _fallback_classification(content: str, partial_data: Dict) -> Dict[str, Any]:
        """Fallback pattern-based classification using keywords."""
        content_lower = content.lower()
        
        # Detect industry
        industry = 'unknown'
        for ind, keywords in TenderClassifier.INDUSTRY_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                industry = ind
                break
        
        # Detect tender type
        tender_type = 'unknown'
        for ttype, keywords in TenderClassifier.TYPE_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                tender_type = ttype
                break
        
        # Detect complexity
        complexity_dict = {'simple': 0, 'moderate': 1, 'complex': 2}
        complexity = 'moderate'
        for level, keywords in {'complex': ['complex', 'advanced', 'sophisticated'], 'simple': ['basic', 'simple', 'straightforward']}.items():
            if any(kw in content_lower for kw in keywords):
                complexity = level
        
        # Extract themes from content
        themes = []
        theme_keywords = {
            'security': ['security', 'secure', 'protection', 'encrypted'],
            'cost_efficiency': ['cost', 'budget', 'economical', 'savings'],
            'scalability': ['scalable', 'scale', 'growth', 'expand'],
            'reliability': ['reliable', 'uptime', 'availability', 'redundant'],
            'support': ['support', 'maintenance', 'service', 'helpdesk'],
            'innovation': ['innovative', 'cutting-edge', 'modern', 'latest']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(kw in content_lower for kw in keywords):
                themes.append(theme)
        
        return {
            'tender_type': partial_data.get('tender_type', tender_type),
            'industry': partial_data.get('industry', industry),
            'complexity': partial_data.get('complexity', complexity),
            'key_themes': partial_data.get('key_themes', themes[:5]),
            'evaluation_focus': partial_data.get('evaluation_focus', ['Technical capability', 'Cost', 'Experience']),
            'priority_areas': partial_data.get('priority_areas', ['Reliability', 'Support', 'Quality']),
            'estimated_value': partial_data.get('estimated_value', 'Not specified')
        }

    @staticmethod
    def _create_default_profile(content: str) -> TenderProfile:
        """Create a profile using pattern-based classification."""
        data = TenderClassifier._fallback_classification(content, {})
        
        return TenderProfile(
            tender_type=data.get('tender_type', 'unknown'),
            industry=data.get('industry', 'unknown'),
            complexity=data.get('complexity', 'moderate'),
            key_themes=data.get('key_themes', ['Cost', 'Reliability', 'Support']),
            evaluation_focus=data.get('evaluation_focus', ['Technical capability', 'Cost', 'Experience']),
            priority_areas=data.get('priority_areas', ['Reliability', 'Support', 'Quality'])
        )


class DynamicProposalDesigner:
    """Designs proposal structure based on tender analysis and past proposals."""

    # Base sections that should always be included
    BASE_SECTIONS = {
        'executive_summary': ProposalSectionDef(
            name='executive_summary',
            title='Executive Summary',
            importance='critical',
            suggested_length='250-350 words',
            key_points=['Value proposition', 'Key differentiators', 'Expected outcomes']
        ),
        'technical_approach': ProposalSectionDef(
            name='technical_approach',
            title='Technical Approach & Methodology',
            importance='critical',
            suggested_length='400-500 words',
            key_points=['Implementation approach', 'Technology stack', 'Integration strategy']
        ),
        'implementation_timeline': ProposalSectionDef(
            name='implementation_timeline',
            title='Implementation Timeline',
            importance='high',
            suggested_length='200-300 words',
            key_points=['Project phases', 'Key milestones', 'Critical path']
        ),
        'pricing': ProposalSectionDef(
            name='pricing',
            title='Pricing & Commercial Terms',
            importance='high',
            suggested_length='150-250 words',
            key_points=['Cost breakdown', 'Payment terms', 'Value analysis']
        ),
        'terms_conditions': ProposalSectionDef(
            name='terms_conditions',
            title='Terms & Conditions',
            importance='medium',
            suggested_length='150-200 words',
            key_points=['Service levels', 'Support', 'Contract terms']
        ),
    }

    # Custom section templates for different scenarios
    CUSTOM_SECTION_TEMPLATES = {
        'security': {
            'title': 'Data Security & Compliance',
            'importance': 'high',
            'suggested_length': '300-400 words',
            'key_points': ['Security framework', 'Data protection', 'Compliance standards'],
            'triggers': ['security', 'data protection', 'compliance', 'encryption', 'audit']
        },
        'fleet_management': {
            'title': 'Fleet & Equipment Details',
            'importance': 'high',
            'suggested_length': '300-400 words',
            'key_points': ['Fleet composition', 'Equipment specifications', 'Maintenance'],
            'triggers': ['fleet', 'vehicle', 'equipment', 'logistics', 'transport']
        },
        'support_sla': {
            'title': 'Support & Service Level Agreements',
            'importance': 'high',
            'suggested_length': '250-350 words',
            'key_points': ['Support hours', 'Response times', 'SLA metrics'],
            'triggers': ['support', 'sla', 'response time', '24/7', 'availability']
        },
        'scalability': {
            'title': 'Scalability & Growth Strategy',
            'importance': 'medium',
            'suggested_length': '250-300 words',
            'key_points': ['Growth capacity', 'Elasticity', 'Performance at scale'],
            'triggers': ['scale', 'growth', 'capacity', 'expansion', 'performance']
        },
        'innovation': {
            'title': 'Innovation & Future Roadmap',
            'importance': 'medium',
            'suggested_length': '200-300 words',
            'key_points': ['Innovation approach', 'Future enhancements', 'Research & development'],
            'triggers': ['innovation', 'future', 'roadmap', 'r&d', 'emerging']
        },
        'case_studies': {
            'title': 'Case Studies & References',
            'importance': 'medium',
            'suggested_length': '300-400 words',
            'key_points': ['Relevant case studies', 'Success metrics', 'Client testimonials'],
            'triggers': ['case study', 'reference', 'proven', 'success', 'similar']
        },
        'risk_mitigation': {
            'title': 'Risk Management & Mitigation',
            'importance': 'medium',
            'suggested_length': '250-350 words',
            'key_points': ['Risk identification', 'Mitigation strategies', 'Contingency plans'],
            'triggers': ['risk', 'mitigation', 'contingency', 'failover', 'backup']
        },
    }

    def __init__(self):
        """Initialize designer."""
        self.model_manager = get_model_manager()
        self.rag_service = get_rag_service()
        logger.info("DynamicProposalDesigner initialized")

    def design_structure(
        self,
        tender_doc: TenderDocument,
        tender_profile: Optional[TenderProfile] = None,
        progress_callback: Optional[callable] = None
    ) -> DynamicProposalStructure:
        """
        Design proposal structure for a tender.
        
        Args:
            tender_doc: Tender document
            tender_profile: Optional pre-classified tender profile
            progress_callback: Optional callback function(step, total, step_name)
            
        Returns:
            DynamicProposalStructure: Designed proposal structure
        """
        try:
            # Classify tender if not provided
            if not tender_profile:
                if progress_callback:
                    progress_callback(1, 4, "classifying_tender")
                tender_profile = TenderClassifier.classify(tender_doc, progress_callback)
            
            # Find similar past proposals in RAG
            if progress_callback:
                progress_callback(2, 4, "loading_references")
            similar_proposals = self._find_similar_proposals(tender_profile)
            
            # Identify needed custom sections
            if progress_callback:
                progress_callback(3, 4, "designing_structure")
            custom_sections = self._identify_custom_sections(tender_profile, tender_doc)
            
            # Determine section ordering and emphasis
            section_order = self._determine_section_order(tender_profile, custom_sections)
            
            # Create structure
            structure = DynamicProposalStructure(
                base_sections=list(self.BASE_SECTIONS.values()),
                custom_sections=custom_sections,
                optional_sections=[],  # Can be added based on analysis
                section_order=section_order,
                design_rationale=self._generate_design_rationale(
                    tender_profile,
                    custom_sections,
                    section_order
                ),
                tender_profile=tender_profile,
                estimated_total_length=self._estimate_total_length(custom_sections)
            )
            
            logger.info(f"Proposal structure designed: {len(section_order)} sections")
            return structure
        
        except Exception as e:
            logger.error(f"Failed to design proposal structure: {str(e)}")
            raise

    def _find_similar_proposals(self, profile: TenderProfile) -> List[str]:
        """
        Find similar past proposals using RAG.
        
        Args:
            profile: Tender profile
            
        Returns:
            List of similar proposal references
        """
        try:
            query = f"Proposals for {profile.tender_type} in {profile.industry} industry"
            results = self.rag_service.search(query, max_results=3)
            return [r.get('id', 'unknown') for r in results]
        except Exception as e:
            logger.warning(f"Failed to find similar proposals: {str(e)}")
            return []

    def _identify_custom_sections(
        self,
        profile: TenderProfile,
        tender_doc: TenderDocument
    ) -> List[ProposalSectionDef]:
        """
        Identify custom sections needed based on tender requirements.
        
        Args:
            profile: Tender profile
            tender_doc: Tender document
            
        Returns:
            List of custom sections to include
        """
        custom_sections = []
        
        # Combine themes and evaluation focus
        all_topics = profile.key_themes + profile.priority_areas + profile.evaluation_focus
        all_topics_lower = [t.lower() for t in all_topics]
        
        # Check each custom section template
        for section_key, template in self.CUSTOM_SECTION_TEMPLATES.items():
            # See if any trigger words match
            if any(trigger in ' '.join(all_topics_lower) for trigger in template['triggers']):
                custom_sections.append(ProposalSectionDef(
                    name=section_key,
                    title=template['title'],
                    importance=template['importance'],
                    suggested_length=template['suggested_length'],
                    key_points=template['key_points'],
                    is_custom=True,
                    reason=f"Identified from tender focus on {', '.join(profile.priority_areas)}"
                ))
        
        return custom_sections

    def _determine_section_order(
        self,
        profile: TenderProfile,
        custom_sections: List[ProposalSectionDef]
    ) -> List[str]:
        """
        Determine optimal section order based on tender evaluation criteria.
        
        Args:
            profile: Tender profile
            custom_sections: Custom sections identified
            
        Returns:
            List of section names in recommended order
        """
        order = ['executive_summary']  # Always first
        
        # Add high-importance custom sections before technical
        high_importance = [s.name for s in custom_sections if s.importance in ['critical', 'high']]
        order.extend(high_importance)
        
        # Add technical sections
        order.append('technical_approach')
        
        # Add medium-importance custom sections
        medium_importance = [s.name for s in custom_sections if s.importance == 'medium']
        order.extend(medium_importance)
        
        # Add timeline and pricing
        order.extend(['implementation_timeline', 'pricing'])
        
        # Add T&C last
        order.append('terms_conditions')
        
        return order

    def _generate_design_rationale(
        self,
        profile: TenderProfile,
        custom_sections: List[ProposalSectionDef],
        section_order: List[str]
    ) -> str:
        """Generate explanation of design choices."""
        rationale = f"""
Proposal structure designed for {profile.tender_type.replace('_', ' ')} tender:

TENDER ANALYSIS:
- Industry: {profile.industry}
- Complexity: {profile.complexity}
- Key themes: {', '.join(profile.key_themes)}
- Buyer priorities: {', '.join(profile.priority_areas)}
- Evaluation focus: {', '.join(profile.evaluation_focus)}

STRUCTURE:
- Base sections: 5
- Custom sections added: {len(custom_sections)} ({', '.join([s.name for s in custom_sections])})
- Total estimated sections: {len(section_order)}

RATIONALE:
1. Executive summary leads to establish context and value proposition
2. Custom sections {', '.join([f'"{s.title}"' for s in custom_sections])} added based on buyer priorities
3. Technical approach and implementation timeline follow to show capability
4. Pricing and terms close the proposal with commercial details

This structure aligns with {profile.industry} industry best practices and emphasizes 
the buyer's stated priorities in {', '.join(profile.priority_areas)}.
""".strip()
        return rationale

    def _estimate_total_length(self, custom_sections: List[ProposalSectionDef]) -> str:
        """Estimate total proposal length."""
        base_count = 5  # Executive summary, technical, timeline, pricing, T&C
        total_sections = base_count + len(custom_sections)
        
        # Rough estimate: 250-500 words per section
        min_words = total_sections * 250
        max_words = total_sections * 500
        
        return f"{min_words}-{max_words} words"
