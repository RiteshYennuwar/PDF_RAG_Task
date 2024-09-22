import streamlit as st
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import cohere
import numpy as np

st.set_page_config(page_icon="📄", layout="wide", page_title="QA Bot with RAG")

def icon(emoji: str):
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("🤖")
st.subheader("Ask Questions Based on Your Document")

pinecone = Pinecone(api_key='your-apikey', environment='us-west1-gcp')  # please replace it with you Pinecone api key
index_name = "qa-bot"

if not pinecone.list_indexes():  
    pinecone.create_index(
        name=index_name,
        dimension=768,  
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )
index = pinecone.Index(index_name)

cohere_client = cohere.Client(api_key='your-apikey')# please replace it with you cohere api key

# Use a more powerful embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def create_embeddings(text):
    embeddings = embedding_model.encode(text)
    normalized_embeddings = [normalize_embedding(emb) for emb in embeddings]
    return [embedding.tolist() for embedding in normalized_embeddings]  

def store_embeddings_in_pinecone(text_chunks):
    embeddings = create_embeddings(text_chunks)
    vectors = [{"id": str(i), "values": embedding, "metadata": {"text": chunk}} for i, (embedding, chunk) in enumerate(zip(embeddings, text_chunks))]
    index.upsert(vectors=vectors, namespace="document-namespace")  

def chunk_text(text, max_chunk_size=512):
    sentences = text.split('. ')  
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def query_pinecone(query_text):
    query_embedding = normalize_embedding(embedding_model.encode([query_text])[0])  
    query_results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True, namespace="document-namespace")
    return [match['metadata']['text'] for match in query_results['matches']]

def generate_answer_from_chunks(query, retrieved_chunks):
    combined_text = " ".join(retrieved_chunks) 
    prompt = f"{combined_text}\n\nAnswer the following question: {query}"
    response = cohere_client.generate(
        model='c4ai-aya-23-35b',
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    document_text = ''
    for page in reader.pages:
        document_text += page.extract_text()

    st.write("Extracted Document Text")

    text_chunks = chunk_text(document_text)
    store_embeddings_in_pinecone(text_chunks)
    st.success("Embeddings stored successfully!")

query = st.text_input("Ask a question based on the document")
if query:
    retrieved_chunks = query_pinecone(query)
    
    generated_answer = generate_answer_from_chunks(query, retrieved_chunks)
    
    st.write("Generated Answer")
    st.write(generated_answer)
