import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key_here")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = "qa-bot"

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your_cohere_api_key_here")

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

MAX_CHUNK_SIZE = 512

TOP_K_RESULTS = 5

MAX_TOKENS = 300
GENERATION_MODEL = "c4ai-aya-23-35b"

PAGE_ICON = "📄"
PAGE_TITLE = "QA Bot with RAG"