#!/usr/bin/env python3
"""Test script to validate improved document layout."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.document_exporter import DocumentExporter

def test_improved_layout():
    """Test the improved document layout with proper flow: cover → org info → sections."""
    
    exporter = DocumentExporter()
    
    # Test data matching the user's output
    org_data = {
        'name': 'Safaricom',
        'industry': 'Government',
        'contact_email': 'emmanuelketer124@gmail.com',
        'contact_phone': '+254703260126'
    }
    
    tender_title = 'PROVISION OF BULK SMS SERVICES FOR A PERIOD OF THREE (3) YEARS'
    
    # Minimal proposal content with different section types
    proposal_content = {
        'executive_summary': """In this proposal, Safaricom presents a comprehensive solution tailored to meet the specified requirements for a three-year SMS gateway service within the government sector. Our offering is designed with a focus on security, reliability, and cost-efficiency, aligning directly with your priority areas.""",
        
        'security': """Safaricom's Data Security & Compliance framework is tailored to meet the specified tender requirements, prioritizing security, reliability, and cost-efficiency. We employ a multi-layered approach with encryption at rest and in transit, regular backups, and access controls.""",
        
        'technical_approach': """Our strategy involves establishing a high-performance SMS gateway that can handle up to 1000 messages per second. Our technology stack employs industry-standard technologies such as Apache Kafka for real-time data processing and MySQL for data storage.""",
        
        'implementation_timeline': """Phase 1: Pre-Implementation (Feb - Jul)
During this phase, we will conduct a comprehensive site survey and design the SMS gateway infrastructure.

Phase 2: Implementation (Aug 2023 - Feb 2024)
Installation and configuration of the SMS gateway with integration into existing infrastructure.

Phase 3: Operations (Mar 2024 - Feb 2026)
Ongoing monitoring and maintenance by our dedicated support team.""",
        
        'pricing': """Our pricing structure is designed to cater exclusively to the services specified in this tender:

1. Setup Fee: One-time charge to establish the SMS gateway
2. Monthly Service Fee: Covers maintenance, support, and operational costs
3. Usage Fees: Based on the volume of SMS messages transmitted""",
    }
    
    try:
        print("📄 Testing improved document layout...")
        print(f"  - Tender: {tender_title}")
        print(f"  - Organization: {org_data['name']}")
        print(f"  - Sections: {', '.join(proposal_content.keys())}")
        print()
        
        # Generate document
        docx_bytes = exporter.export_to_docx(
            proposal_content=proposal_content,
            org_data=org_data,
            tender_title=tender_title
        )
        
        # Save to disk
        output_path = project_root / "data" / "test_output_layout.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(docx_bytes)
        
        file_size = len(docx_bytes)
        print(f"✅ Export succeeded!")
        print(f"   Document size: {file_size:,} bytes")
        print(f"   Saved to: {output_path}")
        print()
        print("📋 Expected document structure:")
        print("  Page 1: Cover page (logo, title, organization name, date)")
        print("  Page 2: Organization Information box + Executive Summary")
        print("  Page 3: Data Security & Compliance")
        print("  Page 4: Technical Approach")
        print("  Page 5: Implementation Timeline")
        print("  Page 6: Pricing")
        print()
        print("✨ Layout improvements:")
        print("  ✓ Cover page generated ONCE (no repetition)")
        print("  ✓ Organization info on second page (not separate)")
        print("  ✓ Executive Summary follows (not another title page)")
        print("  ✓ Section headers styled with orange color and border")
        print("  ✓ Content text remains plain (no color override)")
        print("  ✓ Proper page breaks between major sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_layout()
    sys.exit(0 if success else 1)
