# extract_text.py

import os
from PyPDF2 import PdfReader
from config import PDF_DIR, RAW_TEXT_DIR

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def main():
    os.makedirs(RAW_TEXT_DIR, exist_ok=True)
    for filename in os.listdir(PDF_DIR):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(PDF_DIR, filename)
            text = extract_text_from_pdf(pdf_path)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(RAW_TEXT_DIR, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f'Extracted text from {filename} to {txt_filename}')

if __name__ == '__main__':
    main()
