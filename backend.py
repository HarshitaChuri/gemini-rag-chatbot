#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pdfplumber
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
import traceback

# ----------------------------------------
# Setup and Configuration
# ----------------------------------------

def setup_api_key(api_key: str) -> None:
    """Set up the Google API key for Gemini"""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        print("API key configured successfully")
    except Exception as e:
        print(f"Error configuring API key: {e}")
        raise

# ----------------------------------------
# Section 1: Uploading PDF
# ----------------------------------------

def upload_pdf(pdf_path: str) -> Optional[str]:
    """Upload a PDF file and return its path if valid"""
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
    """Parse a PDF file and extract its text content"""
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
    """Split text into overlapping chunks for processing"""
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
    """Initialize the Google Generative AI embeddings model"""
    try:
        # Check if API key is set
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY environment variable not set")
            return None
            
        embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
        
        # Test the embedding model with a simple input
        test_embedding = embedding_model.embed_query("Test query to verify embedding model works")
        print(f"Embedding model initialized successfully. Test embedding size: {len(test_embedding)}")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return None

# ----------------------------------------
# Section 5: Storing in Vector Database
# ----------------------------------------

def store_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings,
    text_chunks: List[str],
    metadatas: Optional[List[Dict[str, str]]] = None
) -> Optional[Any]:
    """Store document embeddings in a compatible vector database"""
    try:
        # Check for empty input
        if not text_chunks or len(text_chunks) == 0:
            print("Error: No text chunks provided to store_embeddings")
            return None
            
        if not metadatas or len(metadatas) == 0:
            print("Error: No metadata provided to store_embeddings")
            return None
            
        # Create Document objects
        documents = []
        for i, (txt, meta) in enumerate(zip(text_chunks, metadatas)):
            if txt and len(txt.strip()) > 0:
                documents.append(Document(page_content=txt, metadata=meta))
            else:
                print(f"Warning: Skipping empty document at index {i}")
                
        if not documents:
            print("Error: No valid documents to embed")
            return None
            
        print(f"Creating embeddings for {len(documents)} documents")
        
        # Try to use FAISS
        try:
            from langchain_community.vectorstores import FAISS
            
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            print(f"Successfully created FAISS vector store with {len(documents)} documents")
            return vectorstore
        except ImportError:
            print("FAISS not available, falling back to simple vector store")
            
            # Simple dictionary-based vector store if FAISS is not available
            # This is a minimal implementation to avoid SQLite dependency issues
            from langchain_community.vectorstores import DocArrayInMemorySearch
            
            vectorstore = DocArrayInMemorySearch.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            print(f"Created in-memory vector store with {len(documents)} documents")
            return vectorstore
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        traceback.print_exc()
        return None

# ----------------------------------------
# Section 6 & 7: Context Retrieval
# ----------------------------------------

def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n"):
    """Format retrieved chunks for context generation"""
    try:
        chunk_contents = []
        for i, chunk in enumerate(relevant_chunks):
            if hasattr(chunk, 'page_content'):
                source = chunk.metadata.get('source', 'Unknown') if hasattr(chunk, 'metadata') else 'Unknown'
                chunk_text = f"[Chunk {i+1} from {source}]: {chunk.page_content}"
                chunk_contents.append(chunk_text)
        return splitter.join(chunk_contents)
    except Exception as e:
        print(f"Error getting context from chunks: {e}")
        return f"Error retrieving context: {str(e)}"

# ----------------------------------------
# Section 8: Generating Responses with Gemini
# ----------------------------------------

def query_with_full_context(
    query: str,
    vectorstore: Any,
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
    k: int = 3,
    temperature: float = 0.3
) -> Tuple[str, str, List[Any]]:
    """Query the Gemini model with context from the vector database"""
    try:
        # Get relevant documents from the vector store
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        relevant_chunks = retriever.get_relevant_documents(query)
        
        print(f"Retrieved {len(relevant_chunks)} relevant chunks for query: {query}")
        
        # Get formatted context
        context = get_context_from_chunks(relevant_chunks)
        
        # Check if we have any context
        if not context or context.strip() == "":
            return "I couldn't find any relevant information in the provided documents.", "", []
            
        # Create prompt with context and query
        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Use ONLY the following context to answer the question. 
If the answer cannot be determined from the context, respond with "I cannot answer this based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

        # Initialize Gemini model with appropriate settings
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=1024
        )
        
        # Generate response
        response = llm.invoke(prompt)
        return response.content, context, relevant_chunks
        
    except Exception as e:
        print(f"Error in query_with_full_context: {e}")
        return f"Error generating response: {str(e)}", "", []
