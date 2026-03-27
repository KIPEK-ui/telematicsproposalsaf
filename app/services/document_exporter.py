"""
Document Exporter Service
Converts proposal content to professionally formatted .docx files.
Includes Safaricom branding, styling, and structure.
Also supports filling tender templates with proposal content.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

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
        # Set branding paths
        branding_dir = Path(__file__).parent.parent.parent / "data" / "branding"
        self.logo_path = branding_dir / "LOGO.jpg"
        self.details_path = branding_dir / "DETAILS.jpg"
        logger.info("DocumentExporter initialized with branding assets")

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

    def export_as_filled_tender(
        self,
        original_tender_path: str,
        proposal_content: Dict[str, str],
        org_data: Dict[str, Any]
    ) -> bytes:
        """
        Export proposal as filled tender template (preserving original structure).
        
        Args:
            original_tender_path: Path to original tender document
            proposal_content: Dictionary with proposal sections
            org_data: Organization information
        
        Returns:
            bytes: Filled tender document content
            
        Raises:
            DocumentExportError: If filling fails
        """
        try:
            from .form_detector import FormDetector
            from .form_filler import FormFiller

            # Detect form structure in original tender
            detector = FormDetector()
            form_structure = detector.detect_form_structure(original_tender_path)

            # Fill the form with proposal content
            filler = FormFiller()
            filled_docx = filler.fill_form(
                original_tender_path,
                form_structure,
                proposal_content,
                org_data
            )

            logger.info("Filled tender exported successfully")
            return filled_docx

        except Exception as e:
            logger.error(f"Tender filling failed: {str(e)}")
            raise DocumentExportError(f"Failed to fill tender: {str(e)}")

    def export_dual(
        self,
        proposal_content: Dict[str, str],
        org_data: Dict[str, Any],
        original_tender_path: Optional[str] = None,
        tender_title: str = "Proposal Document"
    ) -> Tuple[bytes, Optional[bytes]]:
        """
        Export both branded proposal and filled tender (if original provided).
        
        Args:
            proposal_content: Dictionary with proposal sections
            org_data: Organization information
            original_tender_path: Path to original tender (optional)
            tender_title: Title of the proposal
        
        Returns:
            Tuple[bytes, Optional[bytes]]: (branded_docx, filled_tender_docx or None)
        """
        try:
            # Always generate branded proposal
            branded = self.export_to_docx(proposal_content, org_data, tender_title)

            # Optionally fill original tender if provided
            filled_tender = None
            if original_tender_path and Path(original_tender_path).exists():
                try:
                    filled_tender = self.export_as_filled_tender(
                        original_tender_path,
                        proposal_content,
                        org_data
                    )
                    logger.info("Both branded and filled tender generated")
                except Exception as e:
                    logger.warning(f"Could not generate filled tender: {e}. Returning branded only.")

            return (branded, filled_tender)

        except Exception as e:
            logger.error(f"Dual export failed: {str(e)}")
            raise DocumentExportError(f"Failed to export documents: {str(e)}")

    def export_dual_as_zip(
        self,
        proposal_content: Dict[str, str],
        org_data: Dict[str, Any],
        original_tender_path: Optional[str] = None,
        tender_title: str = "Proposal Document"
    ) -> bytes:
        """
        Export both branded and filled tender as a ZIP archive.
        
        Args:
            proposal_content: Dictionary with proposal sections
            org_data: Organization information
            original_tender_path: Path to original tender (optional)
            tender_title: Title of the proposal
        
        Returns:
            bytes: ZIP file content containing both documents
        """
        try:
            branded, filled_tender = self.export_dual(
                proposal_content,
                org_data,
                original_tender_path,
                tender_title
            )

            # Create ZIP archive
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, 'w') as zip_file:
                # Add branded proposal
                org_safe = "".join(c if c.isalnum() or c == " " else "" for c in org_data.get('name', 'Org'))[:20].replace(" ", "_")
                tender_safe = "".join(c if c.isalnum() or c == " " else "" for c in tender_title)[:20].replace(" ", "_")
                
                branded_filename = f"Proposal_Branded_{org_safe}_{tender_safe}.docx"
                zip_file.writestr(branded_filename, branded)

                # Add filled tender if available
                if filled_tender:
                    filled_filename = f"Proposal_Filled_Tender_{org_safe}_{tender_safe}.docx"
                    zip_file.writestr(filled_filename, filled_tender)

            zip_buffer.seek(0)
            logger.info("Dual export as ZIP completed")
            return zip_buffer.getvalue()

        except Exception as e:
            logger.error(f"ZIP export failed: {str(e)}")
            raise DocumentExportError(f"Failed to create ZIP export: {str(e)}")

    # ========================
    # Document Building Methods
    # ========================

    def _add_header(
        self,
        doc: Document,
        org_data: Dict[str, Any],
        tender_title: str
    ) -> None:
        """Add title and header section with Safaricom branding."""
        # Add logo if available
        if self.logo_path.exists():
            try:
                logo_para = doc.add_paragraph()
                logo_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                logo_run = logo_para.add_run()
                logo_run.add_picture(str(self.logo_path), width=Inches(3))
            except Exception as e:
                logger.warning(f"Could not add logo: {e}")
        
        # Add spacing
        doc.add_paragraph()
        
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
        """Add proposal sections to document with dynamic titles."""
        for key, content in proposal_content.items():
            if content and content.strip():
                # Generate heading from key (dynamic section headers)
                if key in self.SECTION_HEADINGS:
                    heading, level = self.SECTION_HEADINGS[key]
                else:
                    # Convert key to title case for dynamic sections
                    heading = key.replace('_', ' ').title()
                    level = 1
                
                # Add heading with Safaricom branding
                heading_para = doc.add_heading(heading, level=level)
                if heading_para.runs:
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
        """Add footer with Safaricom branding details and contact information."""
        # Add footer with details image if available
        section = doc.sections[0]
        footer = section.footer
        
        # Add details image if available
        if self.details_path.exists():
            try:
                # Clear default footer paragraphs
                for para in footer.paragraphs:
                    p = para._element
                    p.getparent().remove(p)
                
                # Add new paragraph with image
                footer_para = footer.add_paragraph()
                footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                footer_run = footer_para.add_run()
                footer_run.add_picture(str(self.details_path), width=Inches(6))
                logger.info("Added Safaricom details image to footer")
            except Exception as e:
                logger.warning(f"Could not add footer image: {e}")
                # Fallback to text footer
                footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
                footer_para.text = f"{org_data.get('name', 'Safaricom')} | Confidential | {datetime.now().strftime('%Y-%m-%d')}"
                footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in footer_para.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(128, 128, 128)
        else:
            # Fallback text footer if image not available
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
