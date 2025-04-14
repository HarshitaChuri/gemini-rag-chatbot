#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pdfplumber
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ----------------------------------------
# Setup and Configuration
# ----------------------------------------

def setup_api_key(api_key: str) -> None:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    print("API key configured successfully")

# ----------------------------------------
# Section 1: Uploading PDF
# ----------------------------------------

def upload_pdf(pdf_path: str) -> Optional[str]:
    try:
        if os.path.exists(pdf_path):
            print(f"PDF file found at: {pdf_path}")
            return pdf_path
        else:
            print(f"Error: File not found at {pdf_path}")
            return None
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None

# ----------------------------------------
# Section 2: Parsing the PDF
# ----------------------------------------

def parse_pdf(pdf_path: str) -> Optional[str]:
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        print(f"PDF parsed successfully, extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

# ----------------------------------------
# Section 3: Creating Document Chunks
# ----------------------------------------

def create_document_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        print(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error creating document chunks: {e}")
        return []

# ----------------------------------------
# Section 4: Embedding the Documents
# ----------------------------------------

def init_embedding_model(model_name: str = "models/text-embedding-004") -> Optional[GoogleGenerativeAIEmbeddings]:
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
        print("Embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

# ----------------------------------------
# Section 5: Storing in Vector Database (ChromaDB)
# ----------------------------------------

def store_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings,
    text_chunks: List[str],
    metadatas: Optional[List[Dict[str, str]]] = None
) -> Optional[Chroma]:
    try:
        documents = [Document(page_content=txt, metadata=meta)
                     for txt, meta in zip(text_chunks, metadatas)]
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        print(f"Successfully stored {len(documents)} documents in ChromaDB")
        return vectorstore
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None

# ----------------------------------------
# Section 6 & 7: Context Retrieval
# ----------------------------------------

def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n"):
    chunk_contents = []
    for i, chunk in enumerate(relevant_chunks):
        if hasattr(chunk, 'page_content'):
            chunk_text = f"[Chunk {i+1}]: {chunk.page_content}"
            chunk_contents.append(chunk_text)
    return splitter.join(chunk_contents)

# ----------------------------------------
# Section 8: Generating Responses with Gemini
# ----------------------------------------

def query_with_full_context(
    query: str,
    vectorstore: Chroma,
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
    k: int = 3,
    temperature: float = 0.3
) -> Tuple[str, str, List[Any]]:
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        relevant_chunks = retriever.get_relevant_documents(query)
        context = get_context_from_chunks(relevant_chunks)

        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Use ONLY the following context to answer the question. 
If the answer cannot be determined from the context, respond with "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=1024
        )
        response = llm.invoke(prompt)
        return response.content, context, relevant_chunks
    except Exception as e:
        print(f"Error in query_with_full_context: {e}")
        return f"Error generating response: {str(e)}", "", []
