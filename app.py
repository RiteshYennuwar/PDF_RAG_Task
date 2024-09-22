import streamlit as st
from src.document_processor import read_pdf, chunk_text
from src.embedding_handler import store_embeddings_in_pinecone
from src.pinecone_manager import initialize_pinecone
from src.query_handler import process_query
from src.utils import icon
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, COHERE_API_KEY

# Set page configurations for Streamlit
st.set_page_config(page_icon="📄", layout="wide", page_title="QA Bot with RAG")

def main():
    icon("🤖")
    st.subheader("Ask Questions Based on Your Document")

    # Initialize Pinecone
    index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT)

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        document_text = read_pdf(uploaded_file)

        # Generate and store embeddings in Pinecone
        text_chunks = chunk_text(document_text)
        store_embeddings_in_pinecone(index, text_chunks)
        st.success("Embeddings stored successfully!")

    # Query Input
    query = st.text_input("Ask a question based on the document")
    if query:
        # Process query and generate answer
        generated_answer = process_query(index, query, COHERE_API_KEY)
        
        st.write("Generated Answer")
        st.write(generated_answer)

if __name__ == "__main__":
    main()