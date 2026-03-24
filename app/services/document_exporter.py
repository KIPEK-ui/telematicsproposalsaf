"""
Document Exporter Service
Converts proposal content to professionally formatted .docx files.
Includes Safaricom branding, styling, and structure.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentExportError(Exception):
    """Raised when document export fails."""
    pass


class DocumentExporter:
    """
    Exports proposal content to professionally formatted .docx documents.
    Features:
    - Safaricom branding (logo, colors)
    - Professional styling
    - Structured sections with TOC
    - Multi-page support
    """

    # Safaricom brand colors
    BRAND_COLORS = {
        'primary': RGBColor(255, 99, 0),      # Safaricom orange
        'secondary': RGBColor(51, 51, 51),   # Dark gray
        'accent': RGBColor(200, 200, 200)    # Light gray
    }

    SECTION_HEADINGS = {
        'executive_summary': ('Executive Summary', 1),
        'technical_approach': ('Technical Approach', 1),
        'fleet_details': ('Fleet & Equipment Details', 1),
        'implementation_timeline': ('Implementation Timeline', 1),
        'pricing': ('Pricing & Commercial Terms', 1),
        'compliance_assurance': ('Compliance & Quality Assurance', 1),
        'terms_conditions': ('Terms & Conditions', 1),
    }

    def __init__(self):
        """Initialize document exporter."""
        if not DOCX_AVAILABLE:
            raise DocumentExportError(
                "python-docx not available. Install: pip install python-docx"
            )
        logger.info("DocumentExporter initialized")

    def export_to_docx(
        self,
        proposal_content: Dict[str, str],
        org_data: Dict[str, Any],
        tender_title: str = "Proposal Document"
    ) -> bytes:
        """
        Export proposal to .docx format.
        
        Args:
            proposal_content: Dictionary with proposal sections
            org_data: Organization information (name, industry, contact)
            tender_title: Title of the tender/proposal
        
        Returns:
            bytes: .docx file content
            
        Raises:
            DocumentExportError: If export fails
        """
        try:
            doc = Document()
            
            # Set document margins
            sections = doc.sections
            for section in sections:
                section.top_margin = Cm(2)
                section.bottom_margin = Cm(2)
                section.left_margin = Cm(2)
                section.right_margin = Cm(2)
            
            # Add header/title section
            self._add_header(doc, org_data, tender_title)
            
            # Add proposal_content sections
            self._add_sections(doc, proposal_content)
            
            # Add footer
            self._add_footer(doc, org_data)
            
            # Convert to bytes
            output = BytesIO()
            doc.save(output)
            output.seek(0)
            
            logger.info("Document exported successfully to .docx")
            return output.getvalue()
        
        except Exception as e:
            logger.error(f"Document export failed: {str(e)}")
            raise DocumentExportError(f"Failed to export document: {str(e)}")

    def export_to_file(
        self,
        proposal_content: Dict[str, str],
        org_data: Dict[str, Any],
        output_path: str,
        tender_title: str = "Proposal Document"
    ) -> str:
        """
        Export proposal to .docx file on disk.
        
        Args:
            proposal_content: Dictionary with proposal sections
            org_data: Organization information
            output_path: Path to save .docx file
            tender_title: Title of the proposal
        
        Returns:
            str: Path to saved file
            
        Raises:
            DocumentExportError: If export fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            docx_bytes = self.export_to_docx(proposal_content, org_data, tender_title)
            
            with open(output_path, 'wb') as f:
                f.write(docx_bytes)
            
            logger.info(f"Document saved to {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"File export failed: {str(e)}")
            raise DocumentExportError(f"Failed to save file: {str(e)}")

    # ========================
    # Document Building Methods
    # ========================

    def _add_header(
        self,
        doc: Document,
        org_data: Dict[str, Any],
        tender_title: str
    ) -> None:
        """Add title and header section."""
        # Title
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.add_run("PROPOSAL")
        title_run.font.size = Pt(28)
        title_run.font.bold = True
        title_run.font.color.rgb = self.BRAND_COLORS['primary']
        
        # Tender title
        subtitle = doc.add_paragraph(tender_title)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_run = subtitle.runs[0]
        subtitle_run.font.size = Pt(16)
        subtitle_run.font.color.rgb = self.BRAND_COLORS['secondary']
        
        # Organization name
        org_para = doc.add_paragraph(org_data.get('name', 'Organization'))
        org_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        org_run = org_para.runs[0]
        org_run.font.size = Pt(12)
        org_run.font.bold = True
        
        # Date
        date_para = doc.add_paragraph(f"Prepared: {datetime.now().strftime('%B %d, %Y')}")
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.runs[0]
        date_run.font.size = Pt(10)
        date_run.font.italic = True
        
        # Add spacing
        doc.add_paragraph()
        
        # Organization info box
        self._add_organization_info_box(doc, org_data)
        
        # Add page break after header
        doc.add_page_break()

    def _add_organization_info_box(
        self,
        doc: Document,
        org_data: Dict[str, Any]
    ) -> None:
        """Add organization information box."""
        table = doc.add_table(rows=1, cols=2)
        table.autofit = False
        table.allow_autofit = False
        
        # Set table width
        table.width = Inches(6)
        
        # Header row
        header_cells = table.rows[0].cells
        header_cells[0].text = "Organization Details"
        header_cells[1].text = "Contact Information"
        
        # Shade header
        for cell in header_cells:
            self._shade_cell(cell, self.BRAND_COLORS['accent'])
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
        
        # Data row
        data_row_cells = table.add_row().cells
        
        # Left column - organization details
        details = [
            f"Name: {org_data.get('name', 'N/A')}",
            f"Industry: {org_data.get('industry', 'N/A')}",
        ]
        data_row_cells[0].text = "\n".join(details)
        
        # Right column - contact info
        contact = [
            f"Email: {org_data.get('contact_email', 'N/A')}",
            f"Phone: {org_data.get('contact_phone', 'N/A')}",
        ]
        data_row_cells[1].text = "\n".join(contact)

    def _add_sections(
        self,
        doc: Document,
        proposal_content: Dict[str, str]
    ) -> None:
        """Add proposal sections to document."""
        for key, content in proposal_content.items():
            if key in self.SECTION_HEADINGS:
                heading, level = self.SECTION_HEADINGS[key]
                
                # Add heading
                heading_para = doc.add_heading(heading, level=level)
                heading_run = heading_para.runs[0]
                heading_run.font.color.rgb = self.BRAND_COLORS['primary']
                
                # Add content
                # Split content into paragraphs for better formatting
                paragraphs = content.split('\n\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        para = doc.add_paragraph(para_text.strip())
                        para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
                        para.paragraph_format.space_before = Pt(6)
                        para.paragraph_format.space_after = Pt(6)
                
                # Add spacing between sections
                doc.add_paragraph()

    def _add_footer(
        self,
        doc: Document,
        org_data: Dict[str, Any]
    ) -> None:
        """Add footer with organization information."""
        # Add horizontal line
        section = doc.sections[0]
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.text = f"{org_data.get('name', 'Safaricom')} | Confidential | {datetime.now().strftime('%Y-%m-%d')}"
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in footer_para.runs:
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(128, 128, 128)

    @staticmethod
    def _shade_cell(cell, color):
        """Shade a table cell with background color."""
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:fill'), f'{color.rgb:06x}')
        cell._element.get_or_add_tcPr().append(shading_elm)

    # ========================
    # Utility Methods
    # ========================

    def generate_filename(
        self,
        org_name: str,
        tender_title: str
    ) -> str:
        """
        Generate a safe filename for the proposal.
        
        Args:
            org_name: Organization name
            tender_title: Tender title
        
        Returns:
            str: Safe filename with timestamp
        """
        # Sanitize names
        org_safe = "".join(c if c.isalnum() or c == " " else "" for c in org_name)[:20]
        tender_safe = "".join(c if c.isalnum() or c == " " else "" for c in tender_title)[:20]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        filename = f"Proposal_{org_safe}_{tender_safe}_{timestamp}.docx"
        
        return filename.replace(" ", "_")


# Singleton instance
_document_exporter: Optional[DocumentExporter] = None


def get_document_exporter() -> DocumentExporter:
    """Get or create document exporter instance."""
    global _document_exporter
    if _document_exporter is None:
        _document_exporter = DocumentExporter()
    return _document_exporter
