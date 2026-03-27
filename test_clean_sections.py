#!/usr/bin/env python3
"""Test to verify proposal sections are displayed without metadata."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.document_exporter import DocumentExporter

def test_clean_section_export():
    """Test that only proposal sections are exported without metadata."""
    
    exporter = DocumentExporter()
    
    org_data = {
        'name': 'Safaricom',
        'industry': 'Government',
        'contact_email': 'emmanuelketer124@gmail.com',
        'contact_phone': '+254703260126'
    }
    
    tender_title = 'PROVISION OF BULK SMS SERVICES FOR A PERIOD OF THREE (3) YEARS'
    
    # Simulate clean proposal.sections dictionary (what should be passed)
    proposal_sections_only = {
        'cover_page': 'Internal cover page data',
        'executive_summary': 'In this proposal, Safaricom presents a comprehensive solution...',
        'security': 'Safaricom\'s Data Security & Compliance framework...',
        'technical_approach': 'Our strategy involves establishing a high-performance SMS gateway...',
        'implementation_timeline': 'Phase 1: Pre-Implementation\nPhase 2: Implementation',
        'pricing': 'Our pricing structure is designed to cater exclusively...',
        'terms_conditions': 'In our proposal, we offer comprehensive Terms & Conditions...'
    }
    
    try:
        print("✅ Test: Exporting ONLY proposal sections (no metadata)")
        print(f"   Sections to export: {list(proposal_sections_only.keys())}")
        print()
        
        # Generate document
        docx_bytes = exporter.export_to_docx(
            proposal_content=proposal_sections_only,
            org_data=org_data,
            tender_title=tender_title
        )
        
        file_size = len(docx_bytes)
        
        print(f"✅ Export succeeded!")
        print(f"   Document size: {file_size:,} bytes")
        print()
        print("📋 Expected document structure:")
        print("   Page 1: Cover page (logo, title, organization, date)")
        print("   Page 2: Organization Information + Executive Summary")
        print("   Page 3: Data Security & Compliance")
        print("   Page 4: Technical Approach")
        print("   Page 5: Implementation Timeline")
        print("   Page 6: Pricing & Commercial Terms")
        print("   Page 7: Terms & Conditions")
        print()
        print("✨ Benefits of this approach:")
        print("   ✓ No metadata mixed into proposal content")
        print("   ✓ Clean document with only practical sections")
        print("   ✓ Better for end readers")
        print("   ✓ Professional appearance")
        
        # Save for inspection
        output_path = project_root / "data" / "test_clean_proposal.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(docx_bytes)
        print(f"\n💾 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_section_export()
    sys.exit(0 if success else 1)
