# build_retriever.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CLEANED_TEXT_DIR, VECTOR_DIR, EMBEDDING_MODEL_NAME

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # Load documents
    documents = []
    for filename in os.listdir(CLEANED_TEXT_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(CLEANED_TEXT_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Reduced chunk size
        chunk_overlap=50,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    print(f"Total chunks after splitting: {len(docs)}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Build vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save vector store
    vector_store.save_local(VECTOR_DIR)
    print(f"Vector store saved to {VECTOR_DIR}")

if __name__ == '__main__':
    main()
