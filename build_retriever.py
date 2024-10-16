# build_retriever.py

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import CLEANED_TEXT_DIR, VECTOR_DIR, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    texts = []
    for filename in os.listdir(CLEANED_TEXT_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(CLEANED_TEXT_DIR, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())

    # Split texts into chunks
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for text in texts:
        docs.extend(text_splitter.split_text(text))

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Build FAISS index
    vector_store = FAISS.from_texts(docs, embeddings)
    vector_store.save_local(VECTOR_DIR)
    print("FAISS index saved.")

if __name__ == '__main__':
    main()
