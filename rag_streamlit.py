import os
import streamlit as st
import tempfile
from backend import (
    setup_api_key,
    upload_pdf,
    parse_pdf,
    create_document_chunks,
    init_embedding_model,
    store_embeddings,
    get_context_from_chunks,
    query_with_full_context,
    check_versions
)

# Set API key from streamlit secrets if available
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="RAG Chatbot with Gemini", page_icon="📚", layout="wide")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

def main():
    # Display versions at startup
    version_info = check_versions()
    
    with st.sidebar:
        st.title("RAG Chatbot")
        st.subheader("Configuration")
        
        # Display version information in an expander
        with st.expander("Library Versions"):
            for lib, version in version_info.items():
                st.write(f"{lib}: {version}")
        
        api_key = st.text_input("Enter Gemini API Key:", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
        if api_key and st.button("Set API Key"):
            os.environ["GOOGLE_API_KEY"] = api_key
            setup_api_key(api_key)
            st.success("API Key set successfully!")

        st.divider()
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents"):
            process_documents(uploaded_files)

        if st.session_state.processed_files:
            st.subheader("Processed Documents")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

        st.divider()
        with st.expander("Advanced Options"):
            st.slider("Number of chunks to retrieve (k)", 1, 10, 3, key="k_value")
            st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature")

    st.title("Retrieval Augmented Generation Chatbot")
    if st.session_state.vectorstore is None:
        st.info("Please upload and process documents to start chatting.")
        with st.expander("How to use this app"):
            st.markdown("""
            1. Enter your Gemini API Key  
            2. Upload PDF documents  
            3. Click "Process Documents"  
            4. Ask questions in the chat!  
            """)
    else:
        display_chat()
        user_query = st.chat_input("Ask a question about your documents...")
        if user_query:
            handle_user_query(user_query)

def process_documents(uploaded_files):
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        debug_info = st.sidebar.expander("Debug Info")

        if st.session_state.embedding_model is None:
            status_text.text("Initializing embedding model...")
            st.session_state.embedding_model = init_embedding_model()
            if st.session_state.embedding_model is None:
                st.sidebar.error("Failed to initialize embedding model.")
                return

        all_chunks = []
        processed_file_names = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress = int((i / len(uploaded_files)) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            pdf_file = upload_pdf(pdf_path)
            if not pdf_file:
                st.sidebar.warning(f"Failed to process {uploaded_file.name}")
                continue

            text = parse_pdf(pdf_file)
            if not text:
                st.sidebar.warning(f"Failed to extract text from {uploaded_file.name}")
                continue
            
            debug_info.write(f"Extracted {len(text)} characters from {uploaded_file.name}")

            chunks = create_document_chunks(text)
            if not chunks:
                st.sidebar.warning(f"Failed to create chunks from {uploaded_file.name}")
                continue
            
            debug_info.write(f"Created {len(chunks)} chunks from {uploaded_file.name}")

            for chunk in chunks:
                if chunk and len(chunk.strip()) > 0:
                    all_chunks.append({
                        "content": chunk,
                        "source": uploaded_file.name
                    })
                else:
                    debug_info.write(f"Skipping empty chunk from {uploaded_file.name}")

            processed_file_names.append(uploaded_file.name)
            os.unlink(pdf_path)

        progress_bar.progress(100)
        status_text.text("Creating vector database...")

        if all_chunks:
            texts = [chunk["content"] for chunk in all_chunks]
            metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

            debug_info.write(f"Preparing to embed {len(texts)} chunks...")
            
            # Validate chunks before embedding
            valid_texts = []
            valid_metadatas = []
            for i, text in enumerate(texts):
                if text and len(text.strip()) > 0:
                    valid_texts.append(text)
                    valid_metadatas.append(metadatas[i])
                else:
                    debug_info.write(f"Skipping empty chunk from {metadatas[i]['source']}")
            
            debug_info.write(f"Embedding {len(valid_texts)} valid chunks...")
            st.sidebar.write(f"Embedding {len(valid_texts)} chunks...")

            vectorstore = store_embeddings(
                st.session_state.embedding_model,
                valid_texts,
                metadatas=valid_metadatas
            )

            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = processed_file_names
                st.sidebar.success("Documents processed!")
            else:
                st.sidebar.error("❌ Failed to create vector database")
                debug_info.write("Check the logs for more details about the failure")
        else:
            st.sidebar.error("No valid chunks extracted.")

        progress_bar.empty()
        status_text.empty()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.sidebar.error(f"Error processing documents: {str(e)}")
        st.sidebar.expander("Error Details").code(error_details)

def handle_user_query(query):
    if st.session_state.vectorstore is None:
        st.error("Please process documents before asking questions")
        return

    st.session_state.conversation.append({"role": "user", "content": query})
    thinking_placeholder = st.empty()
    thinking_placeholder.info("🤔 Thinking...")

    try:
        k = st.session_state.k_value
        temperature = st.session_state.temperature

        response, context, _ = query_with_full_context(
            query,
            st.session_state.vectorstore,
            k=k,
            temperature=temperature
        )

        st.session_state.conversation.append({"role": "assistant", "content": response, "context": context})
        thinking_placeholder.empty()
        display_chat()

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        thinking_placeholder.empty()
        error_msg = f"Error generating response: {str(e)}"
        st.session_state.conversation.append({"role": "assistant", "content": error_msg})
        display_chat()
        st.expander("Error Details").code(error_details)

def display_chat():
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "context" in message and message["context"]:
                with st.expander("View source context"):
                    st.text(message["context"])

if __name__ == "__main__":
    main()
