"""
resume_input/pdf_parser.py

streamlit_app.py의 extract_text_from_pdf()만 이관.
PDF 추출 로직은 한 줄도 변경하지 않는다.
"""
import io
import pdfplumber


def extract_text_from_pdf(f):
    with pdfplumber.open(io.BytesIO(f.read())) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)
