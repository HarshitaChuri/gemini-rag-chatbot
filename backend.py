import os
import pdfplumber
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma


def setup_api_key(api_key: str) -> None:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)


def upload_pdf(pdf_path: str) -> Optional[str]:
    return pdf_path if os.path.exists(pdf_path) else None


def parse_pdf(pdf_path: str) -> Optional[str]:
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text if text else None
    except:
        return None


def create_document_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    except:
        return []


def init_embedding_model(model_name: str = "models/text-embedding-004") -> Optional[GoogleGenerativeAIEmbeddings]:
    try:
        return GoogleGenerativeAIEmbeddings(model=model_name)
    except:
        return None


def embed_documents(embedding_model: GoogleGenerativeAIEmbeddings, text_chunks: List[str]) -> bool:
    try:
        test_embedding = embedding_model.embed_query(text_chunks[0][:100])
        return bool(test_embedding)
    except:
        return False


def store_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings,
    text_chunks: List[str],
    collection_name: str = "default_collection",
    persist_directory: str = "./chroma_db",
    metadatas: Optional[List[Dict[str, str]]] = None
) -> Optional[Chroma]:
    try:
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            metadatas=metadatas
        )
        vectorstore.persist()
        return vectorstore
    except:
        return None


def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n") -> str:
    chunk_contents = [f"[Chunk {i+1}]: {chunk.page_content}" for i, chunk in enumerate(relevant_chunks) if hasattr(chunk, 'page_content')]
    return splitter.join(chunk_contents)


def query_with_full_context(
    query: str,
    vectorstore: Chroma,
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
    k: int = 3,
    temperature: float = 0.3
) -> Tuple[str, str, List[Any]]:
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        chunks = retriever.get_relevant_documents(query)
        context = get_context_from_chunks(chunks)
        prompt = f"""You are a helpful AI assistant answering questions based on provided context.

Use ONLY the following context to answer the question. 
If the answer cannot be determined from the context, respond with \"I cannot answer this based on the provided context.\"

Context:
{context}

Question: {query}

Answer:"""
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, top_p=0.95, max_output_tokens=1024)
        response = llm.invoke(prompt)
        return response.content, context, chunks
    except Exception as e:
        return f"Error generating response: {str(e)}", "", []
