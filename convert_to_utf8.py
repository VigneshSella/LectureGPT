# convert_to_utf8.py

import os
from config import CLEANED_TEXT_DIR

def convert_to_utf8(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    try:
        # Try decoding with utf-8
        content.decode('utf-8')
        # If successful, do nothing
        return
    except UnicodeDecodeError:
        # Try common encodings
        for encoding in ['latin-1', 'windows-1252', 'iso-8859-1']:
            try:
                decoded_content = content.decode(encoding)
                # Re-encode to utf-8
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(decoded_content)
                print(f'Converted {file_path} from {encoding} to utf-8.')
                return
            except UnicodeDecodeError:
                continue
        print(f'Failed to convert {file_path}.')

def main():
    for filename in os.listdir(CLEANED_TEXT_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(CLEANED_TEXT_DIR, filename)
            convert_to_utf8(file_path)

if __name__ == '__main__':
    main()
