"""
Requirement Extractor
Uses LLM to extract and structure requirements from parsed tenders.
Converts unstructured tender data into actionable requirements for proposal generation.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

from app.services.tender_parser import TenderDocument
from app.ai_service.model_manager import get_model_manager, ModelConnectionError, ModelInferenceError


@dataclass
class RequirementCategory:
    """Definition of a dynamic requirement category."""
    name: str  # Identifier (e.g., "security_requirements")
    title: str  # Display title (e.g., "Security Requirements")
    description: str  # What to look for in this category
    importance: str  # "critical", "high", "medium", "low"
    key_focus_areas: List[str]  # Specific areas to extract


@dataclass
class DynamicRequirementCategories:
    """Dynamically designed requirement categories based on tender profile."""
    categories: List[RequirementCategory]
    category_order: List[str]  # Order to display categories
    extraction_guidance: str  # Guidance for extraction
    tender_type: str
    industry: str
    complexity: str
    
    @property
    def category_names(self) -> List[str]:
        """Get list of category names in order."""
        return self.category_order
    
    @property
    def categories_dict(self) -> Dict[str, RequirementCategory]:
        """Create dictionary mapping names to categories."""
        return {cat.name: cat for cat in self.categories}


@dataclass
class StructuredRequirements:
    """Structured representation of extracted requirements."""
    # Support both static and dynamic categories
    requirements_dict: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    categories: Optional[DynamicRequirementCategories] = None
    additional_notes: str = ""
    
    # Legacy fields for backward compatibility
    fleet_requirements: Dict[str, Any] = field(default_factory=dict)
    technical_specifications: Dict[str, Any] = field(default_factory=dict)
    scope_and_deliverables: Dict[str, Any] = field(default_factory=dict)
    timeline_and_milestones: Dict[str, Any] = field(default_factory=dict)
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = {
            'requirements_dict': self.requirements_dict,
            'additional_notes': self.additional_notes,
            'fleet_requirements': self.fleet_requirements,
            'technical_specifications': self.technical_specifications,
            'scope_and_deliverables': self.scope_and_deliverables,
            'timeline_and_milestones': self.timeline_and_milestones,
            'budget_constraints': self.budget_constraints,
            'compliance_requirements': self.compliance_requirements,
            'evaluation_criteria': self.evaluation_criteria,
        }
        if self.categories:
            base_dict['categories'] = {
                'categories': [asdict(c) for c in self.categories.categories],
                'category_order': self.categories.category_order,
                'extraction_guidance': self.categories.extraction_guidance,
                'tender_type': self.categories.tender_type,
                'industry': self.categories.industry,
                'complexity': self.categories.complexity,
            }
        return base_dict


class RequirementExtractionError(Exception):
    """Raised when requirement extraction fails."""
    pass


class RequirementExtractor:
    """
    Extracts and structures requirements from parsed tender documents.
    Uses LLM-based prompt engineering for intelligent extraction.
    """

    # Industry-specific extraction guidance
    INDUSTRY_EXTRACTION_GUIDANCE = {
        'telecom': {
            'simple': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Service type (SMS, voice, data), coverage areas, service quality (uptime %, availability), support response SLA.
EXAMPLES TO EXTRACT: "SMS delivery confirmation within 30 seconds", "99.5% service availability", "24/7 support center"
EXAMPLES TO IGNORE: "[indicate coverage areas]", "_________ (customer contact)", "Third party liability insurance", "Closing date: xxx"
Pattern: If it starts with [, ends with ], or has excessive underscores/dashes, it's a form template.""",
            
            'moderate': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Service specs (throughput, latency, delivery rates), compliance (telecom regulations, data protection), scalability (concurrent users, message volume), API/interface specs, support SLA with metrics.
EXAMPLES TO EXTRACT: "SMS throughput: 1000 messages/second", "API support: REST and SOAP", "GDPR and local telecommunications compliance", "Support response: 1 hour for critical issues"
EXAMPLES TO IGNORE: "[indicate main requirement]", "_____ (specify)", "Bank guarantee form", "Tender no. 2024/xxx", "(authorized signature)", "Subject to conditions"
Template red flags: Brackets [like this], underscores _____, parenthetical instructions (like this), form header text.""",
            
            'complex': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Detailed service architecture, performance specs (bandwidth, latency, jitter, message delivery guarantee %), security (encryption algorithms, authentication methods, DDoS protection), redundancy/disaster recovery with RTO/RPO, scalability limits, detailed SLA with penalty clauses, regulatory compliance (telecom licensing, spectrum, GDPR, local laws), testing/validation requirements.
EXAMPLES TO EXTRACT: "End-to-end encryption with AES-256", "RTO: 4 hours, RPO: 1 hour", "Multi-region redundancy with automatic failover", "Industry compliance: NIST Cybersecurity Framework, ISO 27001"
EXAMPLES TO IGNORE: "Please indicate security requirements", "[insert compliance standard]", "Insurance/Guarantee form section", "Tender administrative procedures", "Signatory line: _______", "Date: _______"
Critical distinction: Real requirements specify WHAT and HOW. Templates specify [WHERE TO FILL IN]."""
        },
        'cloud': {
            'simple': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Services needed (compute, storage, database), deployment model (public/private/hybrid), basic security (firewalls, access control), uptime SLA.
EXAMPLES TO EXTRACT: "AWS EC2 instances", "99.9% uptime SLA", "TLS encryption for data in transit"
EXAMPLES TO IGNORE: "[indicate service type]", "_______ (specify region)", "Form template fields", "Insurance requirements section".""",
            
            'moderate': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Infrastructure tier/specifications, multi-region/availability zone requirements, disaster recovery concepts, security standards (encryption, IAM), compliance frameworks (vendor-specific), data retention policies, performance metrics (response time, throughput).
EXAMPLES TO EXTRACT: "Availability across 3 regions", "Automated daily backups with 7-day retention", "Users managed via IAM roles"
EXAMPLES TO IGNORE: "[specify cloud provider]", "[insert SLA requirement]", "Bank guarantee", "Tender procedures", "(signature required)".""",
            
            'complex': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Infrastructure stack (specific services and versions), advanced security (encryption key management, VPC isolation, WAF rules), compliance frameworks (SOC2 Type II, ISO 27001, HIPAA, GDPR), disaster recovery with specific RTO/RPO, auto-scaling parameters, monitoring requirements (metrics, alert thresholds), API specification details, performance benchmarks.
EXAMPLES TO EXTRACT: "Secrets encrypted with customer-managed KMS keys", "Compliance: SOC2 Type II audit", "RTO: 2 hours, RPO: 30 minutes", "Application performance monitoring with <100ms response time"
EXAMPLES TO IGNORE: "[Please specify compliance needs]", "Form fields with underscores", "Insurance/guarantee documents", "Tender closing dates", "Authorized by: _______"."""
        },
        'wifi': {
            'simple': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Coverage area/map, bandwidth requirement, number of access points, basic security (WPA2/WPA3), user density estimate.
EXAMPLES TO EXTRACT: "Coverage: Building A-F, 50,000 sqm", "Support 500 concurrent users", "WPA3 encryption standard"
EXAMPLES TO IGNORE: "[indicate coverage]", "_______ sqm", "Template form sections", "[please specify]".""",
            
            'moderate': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Detailed coverage map, capacity metrics (concurrent users, data throughput per AP), RF design requirements, security standards and testing, roaming capability (same network across sites), guest access requirements, performance SLA.
EXAMPLES TO EXTRACT: "1Gbps per access point minimum", "Roaming within site without disconnection", "RF survey and heatmap required", "Support 1000 concurrent users"
EXAMPLES TO IGNORE: "[insert capacity requirement]", "_______ users", "Invoice form section", "Tender reference: xxx", "(customer authorized)".""",
            
            'complex': """Extract ACTUAL SERVICE REQUIREMENTS ONLY.
Focus on: Advanced RF design specs, interference mitigation strategy, detailed capacity planning (users, throughput, concurrent connections), advanced security (WPA3 Enterprise, 802.1X, RADIUS integration), analytics/monitoring requirements (AP usage, client behavior), roaming agreements with failover specs, site survey and design requirements, network QoS specifications.
EXAMPLES TO EXTRACT: "Interference analysis with 2.4/5GHz optimization", "Enterprise 802.1X with LDAP integration", "Monthly analytics dashboard with client band steering reports", "Seamless roaming with <500ms handoff"
EXAMPLES TO IGNORE: "[Please specify RF design]", "_______ interference mitigation", "Form template with [indicate...]", "Tender procedures and dates", "Insurance form"."""
        },
        'infrastructure': {
            'simple': """Extract ACTUAL EQUIPMENT REQUIREMENTS ONLY.
Focus on: Equipment types and models, quantities required, basic specifications, delivery timeline, basic warranty.
EXAMPLES TO EXTRACT: "50x Cisco switches model C9300", "Delivery within 30 days", "1-year warranty included"
EXAMPLES TO IGNORE: "[indicate equipment type]", "Qty: _______", "Form field markers", "Tender administration dates".""",
            
            'moderate': """Extract ACTUAL EQUIPMENT REQUIREMENTS ONLY.
Focus on: Equipment specifications (detailed), performance metrics (throughput, latency, capacity), installation requirements, warranty/support terms, compliance standards, testing requirements.
EXAMPLES TO EXTRACT: "Throughput: 100Gbps, Latency: <100μs", "On-site installation and configuration", "3-year hardware warranty with 4-hour replacement"
EXAMPLES TO IGNORE: "[specify throughput]", "Performance: _______", "Insurance section", "Tender procedures".""",
            
            'complex': """Extract ACTUAL EQUIPMENT REQUIREMENTS ONLY.
Focus on: Detailed technical specs (all parameters), performance benchmarks (full load testing specs), redundancy/failover configuration, sophisticated installation methodology, advanced testing requirements, maintenance/support SLA with specific escalation levels.
EXAMPLES TO EXTRACT: "Load testing: 1M concurrent connections", "Redundant power/cooling with automatic failover", "Support Tier 1: 4-hour response, Tier 2: 1-hour response", "Installation: phase 1-4 with testing between phases"
EXAMPLES TO IGNORE: "[Please specify performance requirements]", "Technical specs: _______", "Form fields and templates", "Insurance guarantees", "Tender closing information"."""
        },
        'fleet': {
            'simple': """Extract ACTUAL FLEET REQUIREMENTS ONLY.
Focus on: Vehicle types and specifications, batch quantities, basic maintenance expectations, fueling/energy type.
EXAMPLES TO EXTRACT: "20x Toyota Hilux pickup trucks", "Monthly maintenance schedule", "Diesel fuel vehicles"
EXAMPLES TO IGNORE: "[indicate vehicle type]", "Qty: _______", "Form template sections", "[please fill in]".""",
            
            'moderate': """Extract ACTUAL FLEET REQUIREMENTS ONLY.
Focus on: Vehicle specs and variants, maintenance requirements and frequency, fuel/energy type and consumption targets, insurance requirements, basic tracking/telematics capabilities, driver support needs.
EXAMPLES TO EXTRACT: "Vehicle specs: 4WD, payload 800kg, AC cabin", "Preventive maintenance every 5000km", "GPS tracking with real-time location"
EXAMPLES TO IGNORE: "[specify tracking needs]", "Vehicle type: _______", "Insurance form section", "Tender procedures", "(signature): _______".""",
            
            'complex': """Extract ACTUAL FLEET REQUIREMENTS ONLY.
Focus on: Detailed vehicle specifications (all parameters), telematics specifications (GPS accuracy, reporting frequency, alert thresholds), maintenance schedule with detailed procedures, fuel/energy efficiency metrics and targets, comprehensive insurance and liability terms, driver training and certification requirements, compliance with safety standards.
EXAMPLES TO EXTRACT: "Telematics: GPS accuracy <10m, reports every 60 seconds, alerts at >100km/h", "Maintenance: every 5000km or 3 months, includes filters/oils/inspection", "Insurance: 3rd party liability 50M KES, fire and theft 80%", "Safety compliance: Euro 5 emissions, ABS/stability control required"
EXAMPLES TO IGNORE: "[Please specify vehicle requirements]", "Vehicles: _______, Qty: _______", "Insurance form with placeholders", "Tender administrative sections", "Authorized by: _______"."""
        }
    }

    # Extraction prompt template - DYNAMIC based on tender metadata
    EXTRACTION_PROMPT = """You are an expert proposal analyst for Safaricom, a telco/transport company.
Your task: Extract ONLY ACTUAL BUSINESS REQUIREMENTS from the tender document.
Critical: Exclude ALL form templates, instructions, placeholder text, and boilerplate.

TENDER CONTEXT (Use this to focus extraction):
- Tender Type: {tender_type}
- Industry: {industry}
- Complexity Level: {complexity}
- Buyer Priorities: {priorities}

=== STEP 1: UNDERSTAND WHAT TO EXTRACT ===

GOOD EXTRACTION (Include these):
- Concrete specifications: "SMS throughput: 1000 messages/second"
- Performance metrics: "Delivery confirmation within 30 seconds"
- Technical requirements: "Support for Unicode and binary SMS"
- Compliance: "Must comply with telecom regulations"
- SLA commitments: "99.5% uptime SLA"
- Clear scope: "Provide SMS gateway for 3 years"

BAD EXTRACTION (Exclude these - these are form/template text):
- Placeholders: "[indicate required throughput]", "(please specify)"
- Form fields: "________", "___________", "[Insert name here]"
- Form instructions: "(indicate main reason)", "(authorized by)", "(stamp below)"
- Boilerplate: "In accordance with", "Subject to conditions", "As per contract"
- Insurance forms: "Third party liability insurance", "Demand guarantee", "Bank guarantee"
- Tender procedure: "Closing date: xxx", "Bid validity: xxx", "Tender reference: xxx"
- Generic text: "Not specified", "To be determined", "TBD"

=== STEP 2: INDUSTRY & COMPLEXITY-SPECIFIC GUIDANCE ===

{extraction_guidance}

=== STEP 3: EXTRACTION RULES (Non-negotiable) ===

Rule 1: IGNORE FORM TEMPLATES
- Exclude: [indicate...], [insert...], [please...], [main reason], etc.
- Exclude: __________, ______, _____, ___________
- Exclude: (indicate xyz), (authorized by), (signatory)
- These are NEVER real requirements

Rule 2: IGNORE TENDER PROCEDURES & ADMINISTRATION
- Exclude: "Closing date:", "Bid validity:", "Tender no:", "Ref:"
- Exclude: "Document submission", "Opening ceremony", "Evaluation timeline"
- These describe HOW to apply, not WHAT is needed

Rule 3: IGNORE CONTRACT BOILERPLATE
- Exclude: "In accordance with", "As per contract", "Subject to conditions"
- Exclude: Signature blocks, letterhead mentions, stamp/seal references
- These are legal template language, not technical requirements

Rule 4: IGNORE INSURANCE/SECURITY FORM SECTIONS
- Exclude: "Third party liability", "Employer's liability", "Professional indemnity"
- Exclude: Insurance forms, security deposit requirements, guarantee forms
- These are administrative, not service requirements

Rule 5: EXTRACT ONLY EXPLICIT MENTIONS
- Only include requirements explicitly stated in the tender
- If a section is heavily templated, mark as "Not specified" rather than guessing
- Prefer exact quotes or close paraphrasing for technical requirements

=== STEP 4: BEFORE YOU RETURN JSON, VERIFY ===

Self-check each extracted requirement:
□ Is this a REAL business requirement (not form text)?
□ Is this explicitly mentioned in the tender (or clearly implied)?
□ Would a vendor need to know this to propose correctly?
□ Does it NOT contain placeholder language like [indicate], _________, (please), etc.?

If any answer is NO, remove it or mark as "Not specified".

TENDER DOCUMENT:
---
{tender_content}
---

=== STEP 5: EXTRACT AND RETURN JSON ===

Extract and structure ONLY ACTUAL BUSINESS REQUIREMENTS:

1. FLEET REQUIREMENTS: Actual vehicle types/quantities (cars, trucks, etc.)
2. TECHNICAL SPECIFICATIONS: Real performance metrics and technical standards
3. SCOPE & DELIVERABLES: Concrete deliverables and what's included/excluded
4. TIMELINE & MILESTONES: Actual dates, milestones, phase timelines
5. BUDGET CONSTRAINTS: Budget amounts, payment terms, currency
6. COMPLIANCE REQUIREMENTS: Certifications, regulations, security standards
7. EVALUATION CRITERIA: How proposals will be scored/evaluated

Return ONLY valid JSON (no markdown, no code blocks, no explanations):
{{
    "fleet_requirements": {{
        "vehicle_types": ["list of actual vehicle types if mentioned"],
        "quantities": "specific quantity or 'not specified' if not mentioned",
        "specifications": ["actual technical specs from tender, or 'not specified'"]
    }},
    "technical_specifications": {{
        "performance_metrics": ["metric: value (e.g., 'throughput: 1000 msg/sec')", "or 'not specified'"],
        "standards": ["list of standards (e.g., 'ISO 27001')", "or 'not specified'"],
        "required_systems": ["list of required integrations/systems", "or 'not specified'"]
    }},
    "scope_and_deliverables": {{
        "scope": "clear statement of what is being procured",
        "deliverables": ["concrete deliverable1", "deliverable2"],
        "key_exclusions": ["what is explicitly NOT included"]
    }},
    "timeline_and_milestones": {{
        "project_start": "actual start date mentioned or 'not specified'",
        "project_end": "actual end date mentioned or 'not specified'",
        "key_dates": ["Date X: purpose/milestone"],
        "milestones": ["Milestone: timeframe"]
    }},
    "budget_constraints": {{
        "budget_range": "specific budget amount mentioned or 'not specified'",
        "payment_terms": "payment schedule and terms from tender",
        "currency": "currency (KES, USD, etc.) or 'not specified'"
    }},
    "compliance_requirements": {{
        "required_certifications": ["cert1: details", "or 'not specified'"],
        "regulations": ["specific law or regulation mentioned"],
        "insurance_requirements": "specific insurance needed or 'not specified'",
        "compliance_standards": ["standard1 (e.g., ISO, NIST)"]
    }},
    "evaluation_criteria": {{
        "evaluation_method": "how proposals will be evaluated",
        "weighted_criteria": ["criterion: weight% (if specified)"],
        "pass_fail_criteria": ["minimum requirement to qualify"]
    }},
    "additional_notes": "Any other important actual requirements not covered above"
}}

FINAL VERIFICATION - Before returning:
✓ Every extracted value is from the tender document, not placeholder text
✓ No [indicate], [insert], _______, or (instruction) patterns remain
✓ No form field markers or boilerplate language
✓ All values are either concrete requirements or explicitly marked "not specified"

CRITICAL: Return ONLY valid JSON. No markdown. No code blocks. No explanations.
Remember: Form templates and instructions are NEVER actual requirements."""

    def __init__(self):
        """Initialize requirement extractor."""
        self.model_manager = get_model_manager()
        logger.info("RequirementExtractor initialized")

    def extract(
        self,
        tender_doc: TenderDocument,
        tender_type: str = None,
        industry: str = None,
        complexity: str = None,
        priority_areas: list = None,
        tender_no: str = None,
        use_llm: bool = True,
        use_fast_model: bool = True,
        progress_callback: Optional[callable] = None
    ) -> StructuredRequirements:
        """
        Extract structured requirements from a tender document with context awareness.
        
        Args:
            tender_doc: Parsed tender document
            tender_type: Type of tender (e.g., 'sms_services', 'cloud_hosting'). If None, extracts generically
            industry: Industry context (e.g., 'telecom', 'cloud'). If None, extracts generically
            complexity: Tender complexity ('simple', 'moderate', 'complex'). Default 'moderate'
            priority_areas: List of buyer priorities to emphasize. If None, treats all equally
            tender_no: Tender reference number for context
            use_llm: If False, uses pattern-based extraction only (faster). Default True
            use_fast_model: If True, uses Mistral 7B instead of DeepSeek R1 (faster). Default True
            progress_callback: Optional callback function(step, total, step_name)
        
        Returns:
            StructuredRequirements: Structured extraction
            
        Raises:
            RequirementExtractionError: If extraction fails
            ModelConnectionError: If model is not available
        """
        try:
            # If LLM extraction is disabled, use patterns only
            if not use_llm:
                logger.info("Using pattern-based requirement extraction (fast mode)")
                if progress_callback:
                    progress_callback(1, 2, "parsing_document")
                requirements = self._extract_with_patterns(tender_doc)
                if progress_callback:
                    progress_callback(2, 2, "extraction_complete")
                return requirements
            
            # Prepare the tender content for extraction (truncated for speed)
            if progress_callback:
                progress_callback(1, 6, "parsing_document")
            tender_content = self._prepare_tender_content(tender_doc, tender_no)
            
            # Truncate to first 1500 chars to speed up LLM inference
            max_content_length = 1500
            if len(tender_content) > max_content_length:
                tender_content = tender_content[:max_content_length] + "\n[... document truncated for processing ...]" 
            
            # Normalize inputs
            tender_type = tender_type or 'unknown'
            industry = industry or 'general'
            complexity = complexity or 'moderate'
            priority_areas = priority_areas or []
            
            # Design dynamic requirement categories for this tender
            if progress_callback:
                progress_callback(3, 6, "designing_categories")
            
            category_designer = get_requirement_category_designer()
            dynamic_categories = category_designer.design_categories(
                tender_type=tender_type,
                industry=industry,
                complexity=complexity,
                key_themes=priority_areas,  # Use priority areas as themes
                priority_areas=priority_areas
            )
            
            logger.info(f"Designed {len(dynamic_categories.categories)} requirement categories")
            
            # Get extraction guidance based on industry and complexity
            extraction_guidance = self._get_extraction_guidance(industry, complexity, priority_areas)
            
            # Build extraction prompt with tender context
            prompt = self.EXTRACTION_PROMPT.format(
                tender_content=tender_content,
                tender_type=tender_type.replace('_', ' ').title(),
                industry=industry.replace('_', ' ').title(),
                complexity=complexity.title(),
                priorities=', '.join(priority_areas) if priority_areas else 'Standard',
                extraction_guidance=extraction_guidance
            )
            
            # Use fast model for speed (Mistral 7B instead of DeepSeek R1)
            if use_fast_model:
                try:
                    if progress_callback:
                        progress_callback(2, 6, "loading_model")
                    logger.info("Loading Mistral 7B for fast requirement extraction...")
                    self.model_manager.load_model("mistral:7b")
                except Exception as e:
                    logger.info(f"Mistral not available, using default model: {e}")
            
            # Ensure a model is loaded
            if not self.model_manager.get_current_model():
                if progress_callback:
                    progress_callback(2, 6, "loading_model")
                logger.info("Loading default model for extraction...")
                self.model_manager.load_model()
            
            # Generate extraction with SHORTER TIMEOUT (60s) to fail fast and use fallback sooner
            if progress_callback:
                progress_callback(3, 6, "extracting_requirements")
            logger.info(f"Extracting requirements via LLM ({self.model_manager.get_current_model().name})...")
            response = self.model_manager.generate(
                prompt=prompt,
                temperature=0.2,  # Low temperature for consistent extraction
                max_tokens=2000,
                timeout=60,  # SHORT TIMEOUT for extraction - fail fast and use pattern-based fallback
                system_prompt="You are a precise JSON extraction expert. Output only valid JSON."
            )
            
            # Parse response
            extracted_data = self._parse_json_response(response)
            
            # Clean extracted data - remove template/placeholder text
            extracted_data = self._clean_extracted_data(extracted_data)
            
            # Build dynamic requirements dictionary from extracted data
            requirements_dict = {}
            for cat in dynamic_categories.categories:
                # Try to find matching data from extraction
                # Map old category names to new ones if needed
                cat_name = cat.name
                requirements_dict[cat_name] = extracted_data.get(cat_name, {})
                
                # also try common aliases
                if not requirements_dict[cat_name]:
                    if 'scope' in cat_name.lower():
                        requirements_dict[cat_name] = extracted_data.get('scope_and_deliverables', {})
                    elif 'timeline' in cat_name.lower():
                        requirements_dict[cat_name] = extracted_data.get('timeline_and_milestones', {})
                    elif 'budget' in cat_name.lower() or 'pricing' in cat_name.lower():
                        requirements_dict[cat_name] = extracted_data.get('budget_constraints', {})
                    elif 'technical' in cat_name.lower():
                        requirements_dict[cat_name] = extracted_data.get('technical_specifications', {})
                    elif 'compliance' in cat_name.lower():
                        requirements_dict[cat_name] = extracted_data.get('compliance_requirements', {})
            
            # Create structured requirements with both legacy and dynamic fields
            requirements = StructuredRequirements(
                # New dynamic fields
                requirements_dict=requirements_dict,
                categories=dynamic_categories,
                additional_notes=extracted_data.get('additional_notes', ''),
                # Legacy fields for backward compatibility
                fleet_requirements=extracted_data.get('fleet_requirements', {}),
                technical_specifications=extracted_data.get('technical_specifications', {}),
                scope_and_deliverables=extracted_data.get('scope_and_deliverables', {}),
                timeline_and_milestones=extracted_data.get('timeline_and_milestones', {}),
                budget_constraints=extracted_data.get('budget_constraints', {}),
                compliance_requirements=extracted_data.get('compliance_requirements', {}),
                evaluation_criteria=extracted_data.get('evaluation_criteria', {}),
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

    def extract_with_fallback(
        self,
        tender_doc: TenderDocument,
        tender_type: str = None,
        industry: str = None,
        complexity: str = None,
        priority_areas: list = None,
        tender_no: str = None,
        use_llm: bool = True,
        progress_callback: Optional[callable] = None
    ) -> StructuredRequirements:
        """
        Extract requirements with fallback to pattern-based extraction if LLM fails.
        Ensures MVP continues to work even if LLM is unavailable.
        Passes tender context to LLM for dynamic extraction.
        
        Args:
            tender_doc: Parsed tender document
            tender_type: Type of tender (e.g., 'sms_services'). If None, extracts generically
            industry: Industry context (e.g., 'telecom'). If None, extracts generically
            complexity: Tender complexity ('simple', 'moderate', 'complex'). Default 'moderate'
            priority_areas: List of buyer priorities. If None, treats all equally
            tender_no: Tender reference number for context
            use_llm: If False, skips LLM and uses pattern-based extraction (faster). Default True
            progress_callback: Optional callback function(step, total, step_name)
        
        Returns:
            StructuredRequirements: Extracted requirements (LLM or fallback)
        """
        try:
            return self.extract(
                tender_doc=tender_doc,
                tender_type=tender_type,
                industry=industry,
                complexity=complexity,
                priority_areas=priority_areas,
                tender_no=tender_no,
                use_llm=use_llm,
                progress_callback=progress_callback
            )
        except (RequirementExtractionError, ModelConnectionError) as e:
            logger.info(f"LLM extraction timed out or unavailable, using fast pattern-based extraction: {str(e)}")
            return self._extract_with_patterns(tender_doc)
        except Exception as e:
            logger.warning(f"Unexpected error in requirement extraction, falling back to patterns: {str(e)}")
            return self._extract_with_patterns(tender_doc)

    def _get_extraction_guidance(self, industry: str, complexity: str, priority_areas: list) -> str:
        """
        Get industry and complexity-specific extraction guidance.
        
        Args:
            industry: Industry type (e.g., 'telecom', 'cloud')
            complexity: Complexity level ('simple', 'moderate', 'complex')
            priority_areas: Buyer priorities
        
        Returns:
            str: Extraction guidance text
        """
        # Get base guidance from industry-complexity matrix
        industry_lower = industry.lower()
        complexity_lower = complexity.lower()
        
        guidance = self.INDUSTRY_EXTRACTION_GUIDANCE.get(industry_lower, {}).get(
            complexity_lower,
            "Extract all requirements mentioned in the tender with appropriate detail."
        )
        
        # Add priority-based guidance
        if priority_areas:
            guidance += f"\n\nPrioritize these buyer priorities in extraction: {', '.join(priority_areas)}."
        
        return guidance

    def _prepare_tender_content(self, tender_doc: TenderDocument, tender_no: str = None) -> str:
        """
        Prepare tender content for extraction prompt.
        Includes title, tender reference, and content.
        
        Args:
            tender_doc: Parsed tender document
            tender_no: Optional tender reference number
        
        Returns:
            str: Formatted tender content
        """
        parts = [f"Title: {tender_doc.title}\n"]
        
        # Add tender reference if available
        if tender_no:
            parts.append(f"Tender Reference: {tender_no}\n")
        
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
    def _clean_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggressively clean extracted data by removing template placeholders, boilerplate, and form text.
        Filters out common form templates, placeholder text, and tender instructions.
        
        Args:
            data: Raw extracted data from LLM
        
        Returns:
            Dict: Cleaned data with template text removed
        """
        # Comprehensive template/boilerplate/form patterns to filter
        template_patterns = [
            # Form field placeholders
            '[indicate', '[insert', '[please', '[main reason', '[insert name', '[indicate main',
            # Underscores and dashes (form blanks)
            '________', '______', '____',
            # Common form instructions
            'if our tender is accepted', 'performance security', 'guarantor:', 'letterhead',
            'define broadly', 'provisions', 'deﬁnes broadly', 'name and address',
            # Contract/legal boilerplate
            'as per contract', 'as per conditions', 'in accordance with', 'subject to',
            # Tender/procurement specific
            'closing date:', 'bid validity', 'tender validity', 'tender no.', 'ref:',
            # Insurance/security terms (often form-related)
            'third party liability', 'demand guarantee', 'bank guarantee',
            # Instructions in extraction output
            'leave blank', 'to be filled', 'to be provided', 'not to be filled',
            # Very generic form responses
            'not specified', 'not applicable', 'to be determined', 'tbd',
            # Other common form markers
            'signature:', 'date:', 'company:', 'authorized by:', 'stamp', 'seal'
        ]
        
        def is_template_text(text: str) -> bool:
            """Check if text is template/placeholder/boilerplate."""
            if not isinstance(text, str):
                return False
            
            text_lower = text.lower().strip()
            
            # Empty or very short
            if len(text_lower) < 5:
                return True
            
            # Matches known template patterns
            if any(pattern in text_lower for pattern in template_patterns):
                return True
            
            # Check for form-like patterns: mostly parenthetical instructions
            # e.g., "some text (indicate something)" or "text _______ (some instruction)"
            if text_lower.count('(') > 0 and text_lower.count(')') > 0:
                # If more than 50% is parenthetical content, it's probably a form
                open_paren = text_lower.rfind('(')
                close_paren = text_lower.rfind(')')
                if close_paren > open_paren and (close_paren - open_paren) > len(text_lower) * 0.4:
                    return True
            
            # Check for excessive whitespace/underscores (form blanks)
            if text_lower.count('_') > 3 or text_lower.count('-') > 5:
                return True
            
            return False
        
        def clean_list(items: Any) -> list:
            """Clean a list by filtering out template text."""
            if not isinstance(items, list):
                return []
            cleaned = []
            for item in items:
                if isinstance(item, str):
                    if not is_template_text(item):
                        cleaned_item = item.strip()
                        if cleaned_item:  # Double-check after stripping
                            cleaned.append(cleaned_item)
                elif isinstance(item, dict):
                    # Recursively clean nested dicts
                    cleaned_dict = {k: clean_value(v) for k, v in item.items()}
                    # Only include dict if it has non-template values
                    if any(v and v != "Not specified" for v in cleaned_dict.values()):
                        cleaned.append(cleaned_dict)
            return cleaned
        
        def clean_value(value: Any) -> Any:
            """Clean a value based on its type."""
            if isinstance(value, list):
                return clean_list(value)
            elif isinstance(value, str):
                if is_template_text(value):
                    return "Not specified"
                return value.strip()
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            return value
        
        # Clean entire data structure
        cleaned_data = {}
        for key, value in data.items():
            cleaned_data[key] = clean_value(value)
        
        logger.debug("Aggressively cleaned extracted data of template and form text")
        return cleaned_data

    @staticmethod
    def _extract_with_patterns(tender_doc: TenderDocument) -> StructuredRequirements:
        """
        Fallback extraction using pattern matching.
        Used when LLM is unavailable.
        Uses intelligent keyword matching to extract requirements.
        
        Args:
            tender_doc: Parsed tender document
        
        Returns:
            StructuredRequirements: Extracted requirements (pattern-based)
        """
        logger.info("Using pattern-based requirement extraction with intelligent filtering")
        
        # Extract fleet requirements
        fleet_vehicles = tender_doc.fleet_details.get('vehicles', [])
        fleet_specs = tender_doc.fleet_details.get('specifications', [])
        fleet_qty = tender_doc.fleet_details.get('quantity', 'Not specified')
        
        # Extract technical specifications
        tech_performance = tender_doc.technical_requirements.get('performance', [])
        tech_compliance = tender_doc.technical_requirements.get('compliance', [])
        
        # Try to identify standards/systems from content
        tech_systems = []
        if isinstance(tech_performance, list) and len(tech_performance) > 0:
            tech_systems = [item for item in tech_performance if isinstance(item, str) and len(item) > 10][:3]
        
        # Extract scope - look for descriptions
        scope = tender_doc.sections.get('description', 'Not specified')
        if not scope or scope == 'Not specified':
            scope = tender_doc.sections.get('overview', 'Not specified')
        
        # Extract timeline - clean up common date formats
        timeline = tender_doc.timeline or 'Not specified'
        
        # Extract budget
        budget = tender_doc.budget_info or 'Not specified'
        
        # Extract compliance from found requirements
        compliance_certs = []
        if tender_doc.technical_requirements:
            for key, val in tender_doc.technical_requirements.items():
                if 'cert' in key.lower() or 'license' in key.lower():
                    if isinstance(val, list):
                        compliance_certs.extend(val)
        
        return StructuredRequirements(
            fleet_requirements={
                'vehicle_types': fleet_vehicles,
                'quantities': fleet_qty,
                'specifications': fleet_specs
            },
            technical_specifications={
                'performance_metrics': tech_performance,
                'standards': tech_compliance,
                'required_systems': tech_systems
            },
            scope_and_deliverables={
                'scope': scope,
                'deliverables': [],
                'key_exclusions': []
            },
            timeline_and_milestones={
                'project_start': 'Not specified',
                'project_end': timeline,
                'key_dates': [timeline] if timeline != 'Not specified' else [],
                'milestones': []
            },
            budget_constraints={
                'budget_range': budget,
                'payment_terms': 'Not specified',
                'currency': 'Not specified'
            },
            compliance_requirements={
                'required_certifications': compliance_certs,
                'regulations': [],
                'insurance_requirements': 'Not specified',
                'compliance_standards': tech_compliance
            },
            evaluation_criteria={
                'evaluation_method': 'Not specified',
                'weighted_criteria': [],
                'pass_fail_criteria': []
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


# ========================
# Dynamic Requirement Category Designer
# ========================

class RequirementCategoryDesigner:
    """
    Designs dynamic requirement categories based on tender profile.
    Similar to DynamicProposalDesigner but for requirements extraction.
    Creates tailored extraction guidance and category structure.
    """
    
    def design_categories(
        self,
        tender_type: str,
        industry: str,
        complexity: str,
        key_themes: List[str],
        priority_areas: List[str]
    ) -> DynamicRequirementCategories:
        """
        Design dynamic requirement categories for a tender.
        
        Args:
            tender_type: Type of tender (e.g., 'sms_services', 'cloud_hosting')
            industry: Industry context (e.g., 'telecom', 'cloud')
            complexity: Tender complexity ('simple', 'moderate', 'complex')
            key_themes: Key themes in the tender (e.g., ['security', 'cost_efficiency'])
            priority_areas: Areas buyer cares about most
            
        Returns:
            DynamicRequirementCategories: Dynamically designed categories
        """
        logger.info(f"Designing requirement categories for {tender_type} ({complexity})")
        
        # Base categories that apply to all tenders
        base_categories = {
            'scope_deliverables': RequirementCategory(
                name='scope_deliverables',
                title='Scope & Deliverables',
                description='What needs to be delivered and when',
                importance='critical',
                key_focus_areas=['Services', 'Deliverables', 'Scope', 'Outputs']
            ),
            'timeline_milestones': RequirementCategory(
                name='timeline_milestones',
                title='Timeline & Milestones',
                description='Key dates, phases, and milestones',
                importance='critical',
                key_focus_areas=['Timeline', 'Start date', 'End date', 'Milestones', 'Phases']
            ),
            'budget_pricing': RequirementCategory(
                name='budget_pricing',
                title='Budget & Pricing',
                description='Cost constraints, budget limits, pricing models',
                importance='high',
                key_focus_areas=['Budget', 'Cost', 'Price', 'Amount', 'Financial']
            ),
        }
        
        # Add complexity-aware categories
        additional_categories = {}
        
        if complexity in ['moderate', 'complex']:
            additional_categories['technical_specs'] = RequirementCategory(
                name='technical_specs',
                title='Technical Specifications',
                description='Technical requirements and specifications',
                importance='high',
                key_focus_areas=['Technical', 'Specifications', 'Technical specs', 'Architecture', 'System']
            )
        
        if complexity == 'complex':
            additional_categories['compliance_security'] = RequirementCategory(
                name='compliance_security',
                title='Compliance & Security',
                description='Security, compliance, and regulatory requirements',
                importance='critical',
                key_focus_areas=['Security', 'Compliance', 'Regulations', 'Standards', 'Encryption', 'Audit']
            )
        
        # Add industry-specific categories
        industry_categories = self._get_industry_specific_categories(industry, complexity)
        
        # Add theme-specific categories
        theme_categories = self._get_theme_specific_categories(key_themes, priority_areas, complexity)
        
        # Combine all categories
        all_categories = {**base_categories, **additional_categories, **industry_categories, **theme_categories}
        
        # Sort by importance and priority
        category_order = self._order_categories(all_categories, priority_areas)
        
        # Create extraction guidance
        extraction_guidance = self._create_extraction_guidance(tender_type, industry, complexity, key_themes)
        
        return DynamicRequirementCategories(
            categories=list(all_categories.values()),
            category_order=category_order,
            extraction_guidance=extraction_guidance,
            tender_type=tender_type,
            industry=industry,
            complexity=complexity
        )
    
    def _get_industry_specific_categories(
        self,
        industry: str,
        complexity: str
    ) -> Dict[str, RequirementCategory]:
        """Get industry-specific requirement categories."""
        categories = {}
        
        if industry == 'telecom':
            categories['service_quality'] = RequirementCategory(
                name='service_quality',
                title='Service Quality & SLAs',
                description='Service level agreements, uptime, performance metrics',
                importance='critical' if complexity == 'complex' else 'high',
                key_focus_areas=['SLA', 'Uptime', 'Availability', 'Performance', 'Response time']
            )
            categories['coverage'] = RequirementCategory(
                name='coverage',
                title='Coverage & Infrastructure',
                description='Coverage areas, network infrastructure, capacity',
                importance='high',
                key_focus_areas=['Coverage', 'Geographic', 'Regional', 'Capacity', 'Infrastructure']
            )
        
        elif industry == 'cloud':
            categories['cloud_infrastructure'] = RequirementCategory(
                name='cloud_infrastructure',
                title='Cloud Infrastructure',
                description='Cloud services, deployment model, regions',
                importance='high',
                key_focus_areas=['AWS', 'Azure', 'GCP', 'Deployment', 'Instance', 'Virtual', 'Regions']
            )
            categories['disaster_recovery'] = RequirementCategory(
                name='disaster_recovery',
                title='Disaster Recovery & Redundancy',
                description='Backup, failover, disaster recovery, data replication',
                importance='high',
                key_focus_areas=['Disaster', 'Recovery', 'Backup', 'Failover', 'Redundancy', 'Replication']
            )
        
        elif industry == 'wifi':
            categories['coverage_capacity'] = RequirementCategory(
                name='coverage_capacity',
                title='Coverage & Capacity',
                description='Coverage area, user capacity, bandwidth requirements',
                importance='critical',
                key_focus_areas=['Coverage', 'Capacity', 'Concurrent users', 'Bandwidth', 'Access points']
            )
        
        elif industry == 'fleet' or industry == 'transport':
            categories['fleet_specs'] = RequirementCategory(
                name='fleet_specs',
                title='Fleet Requirements',
                description='Fleet size, vehicle types, specifications',
                importance='critical',
                key_focus_areas=['Fleet', 'Vehicles', 'Units', 'Type', 'Specifications']
            )
            categories['maintenance'] = RequirementCategory(
                name='maintenance',
                title='Maintenance & Support',
                description='Maintenance schedule, support requirements',
                importance='high',
                key_focus_areas=['Maintenance', 'Support', 'Service', 'Repair', 'Availability']
            )
        
        return categories
    
    def _get_theme_specific_categories(
        self,
        key_themes: List[str],
        priority_areas: List[str],
        complexity: str
    ) -> Dict[str, RequirementCategory]:
        """Get categories based on tender themes and priorities."""
        categories = {}
        all_themes = set(key_themes + priority_areas)
        
        theme_mapping = {
            'security': RequirementCategory(
                name='security_requirements',
                title='Security Requirements',
                description='Security measures, encryption, authentication, access control',
                importance='critical',
                key_focus_areas=['Security', 'Encryption', 'Authentication', 'Firewall', 'Access control']
            ),
            'cost_efficiency': RequirementCategory(
                name='cost_optimization',
                title='Cost Optimization',
                description='Cost reduction strategies, efficiency metrics, ROI',
                importance='high',
                key_focus_areas=['Cost', 'Efficiency', 'Savings', 'ROI', 'Optimization']
            ),
            'scalability': RequirementCategory(
                name='scalability_growth',
                title='Scalability & Growth',
                description='Scalability, growth capacity, future expansion',
                importance='high',
                key_focus_areas=['Scalability', 'Growth', 'Expansion', 'Capacity increase', 'Upgrade']
            ),
            'integration': RequirementCategory(
                name='integration_compatibility',
                title='Integration & Compatibility',
                description='System integration, API compatibility, third-party integration',
                importance='high',
                key_focus_areas=['Integration', 'API', 'Compatibility', 'Third-party', 'Interface']
            ),
            'innovation': RequirementCategory(
                name='innovation_features',
                title='Innovation & Features',
                description='New features, innovative solutions, technology advancements',
                importance='medium',
                key_focus_areas=['Innovation', 'Features', 'Technology', 'New', 'Advanced']
            ),
        }
        
        for theme in all_themes:
            if theme.lower() in theme_mapping:
                categories[theme.lower()] = theme_mapping[theme.lower()]
        
        return categories
    
    def _order_categories(
        self,
        categories: Dict[str, RequirementCategory],
        priority_areas: List[str]
    ) -> List[str]:
        """Order categories by importance and priority."""
        # Sort categories: critical first, then high, then medium, then low
        # Within same importance, prioritize by priority_areas
        
        importance_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        def sort_key(cat_name: str) -> tuple:
            cat = categories[cat_name]
            importance_rank = importance_order.get(cat.importance, 4)
            
            # Check if this category matches a priority area
            priority_rank = 99
            for i, priority in enumerate(priority_areas):
                if priority.lower() in cat.title.lower() or priority.lower() in cat.name.lower():
                    priority_rank = i
                    break
            
            return (importance_rank, priority_rank)
        
        return sorted(categories.keys(), key=sort_key)
    
    def _create_extraction_guidance(
        self,
        tender_type: str,
        industry: str,
        complexity: str,
        key_themes: List[str]
    ) -> str:
        """Create extraction guidance based on tender profile."""
        guidance = f"""Extract requirements specific to this {industry.title()} tender ({complexity} complexity).
Focus on: {', '.join(key_themes) if key_themes else 'general requirements'}
Priority: Extract ACTUAL requirements only, not template placeholders or form fields.
Template indicators to ignore: [...], [indicate ...], _____, (specify), (optional), parenthetical instructions"""
        
        return guidance


# Singleton instances
_extractor: Optional[RequirementExtractor] = None
_refiner: Optional[RequirementRefiner] = None
_category_designer: Optional[RequirementCategoryDesigner] = None


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


def get_requirement_category_designer() -> RequirementCategoryDesigner:
    """Get or create requirement category designer instance."""
    global _category_designer
    if _category_designer is None:
        _category_designer = RequirementCategoryDesigner()
    return _category_designer
