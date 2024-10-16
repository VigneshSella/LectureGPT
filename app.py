# app.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from config import MODEL_DIR, VECTOR_DIR, EMBEDDING_MODEL_NAME
import gradio as gr

def truncate_text(text, tokenizer, max_length):
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    truncated_text = tokenizer.decode(tokenized_text[:max_length], skip_special_tokens=True)
    return truncated_text

def main():
    # Load fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.model_max_length = 512  # Adjust based on your model's context length

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the language model pipeline with adjusted generation parameters
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=150,
    )

    # Wrap the pipeline with LangChain's LLM
    llm = HuggingFacePipeline(pipeline=generator)

    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever with a limit on the number of documents
    # retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # # Create the QA chain using RetrievalQA.from_chain_type
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="map_reduce"
    # )

    # # Create the QA chain using `map_rerank`
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="map_rerank",
    #     return_source_documents=False,
    # ) 
    # Define a prompt template that truncates the context
    prompt_template = """
    Given the following context, answer the question.
    If you don't know the answer, just say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create the QA chain using RetrievalQAWithSourcesChain
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        max_input_size=1024,  # Set max input size
    )

    # Define Gradio interface
    def answer_question(question):
        # Retrieve documents
        docs = retriever.get_relevant_documents(question)

        # Combine documents into context
        context = " ".join([doc.page_content for doc in docs])

        # Truncate context to fit within model's max length
        max_context_length = tokenizer.model_max_length - len(tokenizer.encode(question)) - 50  # Reserve tokens for the question and response
        truncated_context = truncate_text(context, tokenizer, max_context_length)

        # Create input for the model
        input_text = f"Context:\n{truncated_context}\n\nQuestion:\n{question}\n\nAnswer:"

        # Truncate input_text if necessary
        max_input_length = tokenizer.model_max_length - 50  # Reserve tokens for the response
        input_text = truncate_text(input_text, tokenizer, max_input_length)

        # Generate answer
        response = generator(input_text, max_new_tokens=150, do_sample=False)
        return response[0]['generated_text'][len(input_text):]

    iface = gr.Interface(
        fn=answer_question,
        inputs="text",
        outputs="text",
        title="LectureGPT"
    )
    iface.launch()

if __name__ == '__main__':
    main()
