"""evaluation/generate_pdf_report.py
-----------------------------------
Builds publication-grade IEEE-style PDF document PUBLICATION_RESEARCH_REPORT.pdf
from PUBLICATION_RESEARCH_REPORT.md using ReportLab.
"""

import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas


class IEEEArticleCanvas(canvas.Canvas):
    """Custom Canvas with running headers and page footers (Page X of Y)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress running header/footer on title page

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0F172A"))  # Slate 900

        # Running Header
        self.drawString(54, 752, "MedGraphRAG: IEEE-Style Clinical GraphRAG Research Report")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))  # Slate 500
        self.drawRightString(612 - 54, 752, "Academic & Technical Research Paper")

        self.setStrokeColor(colors.HexColor("#CBD5E1"))  # Slate 300
        self.setLineWidth(0.75)
        self.line(54, 744, 612 - 54, 744)

        # Running Footer
        self.line(54, 48, 612 - 54, 48)
        self.drawString(54, 34, "Confidential / Research & Engineering Systems Architecture")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 34, page_str)
        self.restoreState()


def format_inline_markdown(text: str) -> str:
    """Safely escape XML characters and translate markdown formatting into ReportLab HTML tags."""
    # Escape XML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Bold and Italics
    text = re.sub(r"\*\*\*(.*?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)

    # Inline Code
    text = re.sub(r"`(.*?)`", r'<font face="Courier-Bold" color="#0369A1">\1</font>', text)

    # Markdown links [Text](URL) -> Text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"<b>\1</b>", text)

    return text


def generate_pdf_document(md_path: str, pdf_paths: list[str]):
    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"Source markdown file missing: {md_path}")

    raw_text = md_file.read_text(encoding="utf-8")
    lines = raw_text.splitlines()

    # Primary PDF Path
    output_pdf = pdf_paths[0]

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Premium Color Palette
    PRIMARY = colors.HexColor("#0F172A")    # Slate 900 (Primary Headers)
    SECONDARY = colors.HexColor("#0284C7")  # Sky 600 (Subheaders & Accents)
    TEXT_DARK = colors.HexColor("#334155")  # Slate 700 (Body Text)
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Slate 50 (Table Alt Rows)
    CODE_BG = colors.HexColor("#F1F5F9")    # Slate 100 (Code Blocks)
    BORDER = colors.HexColor("#E2E8F0")     # Slate 200

    # Custom Typography Styles
    style_cover_title = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY,
        spaceAfter=12
    )

    style_h1 = ParagraphStyle(
        "H1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        "H2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h3 = ParagraphStyle(
        "H3_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=TEXT_DARK,
        spaceAfter=5
    )

    style_bullet = ParagraphStyle(
        "Bullet_Custom",
        parent=style_body,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    style_code = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.0,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
        backColor=CODE_BG,
        borderColor=BORDER,
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    style_th = ParagraphStyle(
        "TH_Style",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0
    )

    style_td = ParagraphStyle(
        "TD_Style",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=9.5,
        textColor=TEXT_DARK
    )

    story = []
    in_code = False
    code_buffer = []
    in_table = False
    table_buffer = []

    for line in lines:
        # Code Block Boundaries
        if line.strip().startswith("```"):
            if in_code:
                code_raw = "\n".join(code_buffer)
                code_escaped = code_raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(code_escaped.replace("\n", "<br/>").replace(" ", "&nbsp;"), style_code))
                code_buffer = []
                in_code = False
            else:
                in_code = True
                code_buffer = []
            continue

        if in_code:
            code_buffer.append(line)
            continue

        # Table Separator Filter
        if "|" in line and "-|-" in line:
            continue

        # Table Rows Parsing
        if "|" in line:
            cells = [format_inline_markdown(c.strip()) for c in line.split("|")[1:-1]]
            if cells and any(cells):
                if not in_table:
                    in_table = True
                    table_buffer = [cells]
                else:
                    table_buffer.append(cells)
            continue
        elif in_table:
            # Build Table
            if table_buffer:
                table_matrix = []
                for row_idx, r in enumerate(table_buffer):
                    formatted_r = []
                    for cell_text in r:
                        st = style_th if row_idx == 0 else style_td
                        formatted_r.append(Paragraph(cell_text, st))
                    table_matrix.append(formatted_r)

                cols = len(table_buffer[0])
                total_w = 504.0  # 612 - 108 margins
                col_w = total_w / cols

                t = Table(table_matrix, colWidths=[col_w] * cols)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            in_table = False
            table_buffer = []

        line_str = line.strip()
        if not line_str:
            continue

        # Headings Processing
        if line_str.startswith("# "):
            h_text = format_inline_markdown(line_str[2:])
            if not story:  # Cover Title Page
                story.append(Spacer(1, 10))
                story.append(Paragraph(h_text, style_cover_title))
                story.append(HRFlowable(width="100%", thickness=2.5, color=SECONDARY, spaceAfter=12))
            else:
                story.append(Paragraph(h_text, style_h1))
                story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=8))
        elif line_str.startswith("## "):
            story.append(Paragraph(format_inline_markdown(line_str[3:]), style_h2))
        elif line_str.startswith("### "):
            story.append(Paragraph(format_inline_markdown(line_str[4:]), style_h3))
        elif line_str.startswith("#### "):
            story.append(Paragraph(format_inline_markdown(line_str[5:]), style_h3))
        # Bullet Lists
        elif line_str.startswith("- ") or line_str.startswith("* "):
            story.append(Paragraph(f"• {format_inline_markdown(line_str[2:])}", style_bullet))
        elif re.match(r"^\d+\.\s", line_str):
            num, rest = line_str.split(".", 1)
            story.append(Paragraph(f"<b>{num}.</b> {format_inline_markdown(rest.strip())}", style_bullet))
        # Horizontal Separator
        elif line_str == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceBefore=5, spaceAfter=5))
        # Normal Body Text
        else:
            story.append(Paragraph(format_inline_markdown(line_str), style_body))

    # Catch Trailing Table
    if in_table and table_buffer:
        table_matrix = []
        for row_idx, r in enumerate(table_buffer):
            formatted_r = []
            for cell_text in r:
                st = style_th if row_idx == 0 else style_td
                formatted_r.append(Paragraph(cell_text, st))
            table_matrix.append(formatted_r)
        cols = len(table_buffer[0])
        col_w = 504.0 / cols
        t = Table(table_matrix, colWidths=[col_w] * cols)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ]))
        story.append(t)

    # Build Primary PDF
    doc.build(story, canvasmaker=IEEEArticleCanvas)
    print(f"Primary PDF generated successfully: {output_pdf}")

    # Copy to Secondary Paths
    for second_path in pdf_paths[1:]:
        second_dir = os.path.dirname(second_path)
        if second_dir:
            os.makedirs(second_dir, exist_ok=True)
        with open(output_pdf, "rb") as f_in, open(second_path, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"Copied PDF to destination: {second_path}")


if __name__ == "__main__":
    src_md = "/Users/khushaljain/Desktop/MedGraphRAG/PUBLICATION_RESEARCH_REPORT.md"
    dest_pdfs = [
        "/Users/khushaljain/Desktop/MedGraphRAG/PUBLICATION_RESEARCH_REPORT.pdf",
        "/Users/khushaljain/.gemini/antigravity/brain/08d67381-7f1d-4787-afc6-8ffb35978b8a/PUBLICATION_RESEARCH_REPORT.pdf",
        "/Users/khushaljain/Desktop/MedGraphRAG/PROJECT_DOCUMENTATION_REPORT.pdf",
        "/Users/khushaljain/.gemini/antigravity/brain/08d67381-7f1d-4787-afc6-8ffb35978b8a/PROJECT_DOCUMENTATION_REPORT.pdf"
    ]
    generate_pdf_document(src_md, dest_pdfs)
