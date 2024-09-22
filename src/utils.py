import streamlit as st
import numpy as np
from typing import List

def icon(emoji: str) -> None:
    """
    Displays an emoji as a Notion-style page icon.
    
    Args:
    emoji (str): The emoji to display.
    """
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize the embedding vector.
    
    Args:
    embedding (np.ndarray): The embedding vector to normalize.
    
    Returns:
    np.ndarray: The normalized embedding vector.
    """
    return embedding / np.linalg.norm(embedding)

def normalize_embeddings(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize a list of embedding vectors.
    
    Args:
    embeddings (List[np.ndarray]): A list of embedding vectors to normalize.
    
    Returns:
    List[np.ndarray]: A list of normalized embedding vectors.
    """
    return [normalize_embedding(emb) for emb in embeddings]

def format_query(query: str, context: str) -> str:
    """
    Format the query with context for the language model.
    
    Args:
    query (str): The user's query.
    context (str): The context retrieved from the document.
    
    Returns:
    str: The formatted query string.
    """
    return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
    text (str): The text to truncate.
    max_length (int): The maximum length of the truncated text.
    
    Returns:
    str: The truncated text.
    """
    return text[:max_length] + "..." if len(text) > max_length else text