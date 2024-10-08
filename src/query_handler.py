import cohere
from pinecone import Index
from sentence_transformers import SentenceTransformer
from typing import List
from config import EMBEDDING_MODEL_NAME, TOP_K_RESULTS, GENERATION_MODEL, MAX_TOKENS
from .utils import normalize_embedding, format_query

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def query_pinecone(index: Index, query_text: str) -> List[str]:
    query_embedding = normalize_embedding(embedding_model.encode([query_text])[0])
    query_results = index.query(
        vector=query_embedding.tolist(),
        top_k=TOP_K_RESULTS,
        include_metadata=True,
        namespace="document-namespace"
    )
    return [match['metadata']['text'] for match in query_results['matches']]

def generate_answer(query: str, retrieved_chunks: List[str], cohere_api_key: str) -> str:
    cohere_client = cohere.Client(api_key=cohere_api_key)
    
    context = " ".join(retrieved_chunks)
    prompt = format_query(query, context)
    
    response = cohere_client.generate(
        model=GENERATION_MODEL,
        prompt=prompt,
        max_tokens=MAX_TOKENS
    )
    
    return response.generations[0].text

def process_query(index: Index, query: str, cohere_api_key: str) -> str:
    retrieved_chunks = query_pinecone(index, query)
    answer = generate_answer(query, retrieved_chunks, cohere_api_key)
    return answer