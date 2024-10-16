# preprocess_text.py

import os
import re
from config import RAW_TEXT_DIR, CLEANED_TEXT_DIR

def clean_text(text):
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text

def main():
    os.makedirs(CLEANED_TEXT_DIR, exist_ok=True)
    for filename in os.listdir(RAW_TEXT_DIR):
        if filename.endswith('.txt'):
            raw_text_path = os.path.join(RAW_TEXT_DIR, filename)
            with open(raw_text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            cleaned_text = clean_text(text)
            cleaned_text_path = os.path.join(CLEANED_TEXT_DIR, filename)
            with open(cleaned_text_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f'Cleaned text saved to {filename}')

if __name__ == '__main__':
    main()
