# config.py

# Paths
DATA_DIR = 'data/'
PDF_DIR = DATA_DIR + 'pdfs/'
RAW_TEXT_DIR = DATA_DIR + 'raw_text/'
CLEANED_TEXT_DIR = DATA_DIR + 'cleaned_text/'
MODEL_DIR = 'models/'
VECTOR_DIR = 'vectors/'

# Model parameters
BASE_MODEL_NAME = 'EleutherAI/gpt-neo-125M'  # gpt-6B recommends 16 GB VRAM :(
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
