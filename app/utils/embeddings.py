from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import os
import logging
from typing import List, Union, Optional
from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Initialize the embeddings model with API key from environment
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(text: str) -> List[float]:
    """Convert text to embedding vector

    Args:
        text: Text to convert to embedding

    Returns:
        Embedding vector as a list of floats
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        # Return zero vector with the correct dimension
        return [0.0] * 1536  # Default OpenAI embedding size

    try:
        # Preprocess the text
        processed_text = preprocess_text(text)

        # Get the embedding from the model
        embedding = embeddings_model.embed_query(processed_text)

        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return zero vector in case of error
        return [0.0] * 1536

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """Convert multiple texts to embeddings

    Args:
        texts: List of texts to convert

    Returns:
        List of embedding vectors
    """
    if not texts:
        logger.warning("Empty list provided for batch embedding")
        return []

    try:
        # Preprocess all texts
        processed_texts = [preprocess_text(text) for text in texts]

        # Get embeddings for all texts
        embeddings = embeddings_model.embed_documents(processed_texts)

        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}")
        # Return empty list in case of error
        return []

def preprocess_text(text: str) -> str:
    """Preprocess text for embedding

    Args:
        text: Text to preprocess

    Returns:
        Preprocessed text
    """
    if not text:
        return ""

    # Basic preprocessing
    processed_text = text.strip()

    # Truncate if necessary (OpenAI models have token limits)
    max_tokens = 8000  # Conservative estimate
    if len(processed_text.split()) > max_tokens:
        processed_text = " ".join(processed_text.split()[:max_tokens])
        logger.warning(f"Text truncated to {max_tokens} tokens")

    return processed_text

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors

    Args:
        vec1, vec2: Vectors to compare

    Returns:
        Similarity score between 0 and 1
    """
    if not vec1 or not vec2:
        return 0.0

    try:
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate dot product
        dot_product = np.dot(v1, v2)

        # Calculate magnitudes
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Calculate cosine similarity
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors

    Args:
        vec1, vec2: Vectors to compare

    Returns:
        Distance value (lower means more similar)
    """
    if not vec1 or not vec2:
        return float('inf')

    try:
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate Euclidean distance
        return np.linalg.norm(v1 - v2)
    except Exception as e:
        logger.error(f"Error calculating Euclidean distance: {str(e)}")
        return float('inf')
