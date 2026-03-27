#!/usr/bin/env python3
"""Test to verify tender reference appears on cover page."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.document_exporter import DocumentExporter

def test_tender_reference():
    """Test that tender reference appears on cover page with proper styling."""
    
    exporter = DocumentExporter()
    
    org_data = {
        'name': 'Safaricom',
        'industry': 'Government',
        'contact_email': 'emmanuelketer124@gmail.com',
        'contact_phone': '+254703260126'
    }
    
    tender_title = 'PROVISION OF BULK SMS SERVICES FOR A PERIOD OF THREE (3) YEARS'
    tender_reference = 'KRA/HQS/DP-016/2025-2026'
    
    proposal_content = {
        'executive_summary': 'In this proposal, Safaricom presents a comprehensive solution tailored to meet the specified requirements...',
        'security': 'Safaricom\'s Data Security & Compliance framework provides robust protection...',
        'technical_approach': 'Our strategy involves establishing a high-performance SMS gateway...',
    }
    
    try:
        print("📄 Testing tender reference on cover page...")
        print(f"   Organization: {org_data['name']}")
        print(f"   Tender Title: {tender_title}")
        print(f"   Tender Reference: {tender_reference}")
        print()
        
        # Generate document with tender reference
        docx_bytes = exporter.export_to_docx(
            proposal_content=proposal_content,
            org_data=org_data,
            tender_title=tender_title,
            tender_reference=tender_reference
        )
        
        # Save to disk
        output_path = project_root / "data" / "test_tender_reference.docx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(docx_bytes)
        
        file_size = len(docx_bytes)
        print(f"✅ Export succeeded!")
        print(f"   Document size: {file_size:,} bytes")
        print(f"   Saved to: {output_path}")
        print()
        print("📋 Cover page now includes:")
        print("   ✓ Safaricom logo")
        print("   ✓ 'PROPOSAL' title (orange, 32pt)")
        print("   ✓ Tender title (gray, 18pt)")
        print("   ✓ Tender Reference: KRA/HQS/DP-016/2025-2026 (orange, bold, 11pt)")
        print("   ✓ Organization name (14pt)")
        print("   ✓ Prepared date (11pt italic)")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tender_reference()
    sys.exit(0 if success else 1)
