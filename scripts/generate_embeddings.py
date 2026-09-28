#!/usr/bin/env python
"""
Generate Embeddings Script

This script generates embeddings for texts in JSON files and saves them
for later use, reducing API calls during runtime.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Union
from dotenv import load_dotenv

# Add the project root to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.embeddings import batch_get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(input_file: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Load data from a JSON file

    Args:
        input_file: Path to JSON file

    Returns:
        Loaded data (list or dictionary)
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {input_file}: {str(e)}")
        raise

def extract_texts(data: Union[List[Dict[str, Any]], Dict[str, Any]],
                 text_field: str = "text") -> List[str]:
    """Extract texts from data structure

    Args:
        data: Data structure (list or dictionary)
        text_field: Field name containing the text

    Returns:
        List of texts
    """
    texts = []

    if isinstance(data, list):
        for item in data:
            if text_field in item:
                texts.append(item[text_field])
            else:
                logger.warning(f"Item missing '{text_field}' field: {item}")
    elif isinstance(data, dict):
        if text_field in data:
            texts.append(data[text_field])
        else:
            logger.warning(f"Data missing '{text_field}' field: {data}")
    else:
        logger.error(f"Unsupported data type: {type(data)}")
        raise ValueError("Input data must be a list or dictionary")

    return texts

def add_embeddings_to_data(data: Union[List[Dict[str, Any]], Dict[str, Any]],
                          embeddings: List[List[float]],
                          text_field: str = "text",
                          embedding_field: str = "embedding") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Add embeddings to data structure

    Args:
        data: Data structure (list or dictionary)
        embeddings: List of embedding vectors
        text_field: Field name containing the text
        embedding_field: Field name to store the embedding

    Returns:
        Data structure with embeddings added
    """
    if isinstance(data, list):
        embedding_index = 0
        for item in data:
            if text_field in item:
                item[embedding_field] = embeddings[embedding_index]
                embedding_index += 1
    elif isinstance(data, dict):
        if text_field in data:
            data[embedding_field] = embeddings[0]

    return data

def generate_and_save_embeddings(input_file: str,
                                output_file: str,
                                text_field: str = "text",
                                embedding_field: str = "embedding"):
    """Generate embeddings for texts in a JSON file and save them

    Args:
        input_file: JSON file with texts to embed
        output_file: Output file to save embeddings
        text_field: Field name containing the text
        embedding_field: Field name to store the embedding
    """
    try:
        # Load data
        data = load_data(input_file)
        logger.info(f"Loaded data from {input_file}")

        # Extract texts
        texts = extract_texts(data, text_field)
        logger.info(f"Extracted {len(texts)} texts for embedding")

        if not texts:
            logger.warning("No texts found to embed")
            return

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = batch_get_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Add embeddings to data
        data_with_embeddings = add_embeddings_to_data(data, embeddings, text_field, embedding_field)

        # Save data with embeddings
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_with_embeddings, f, ensure_ascii=False, indent=2)

        logger.info(f"Embeddings saved to {output_file}")

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        sys.exit(1)

def main():
    """Main function to generate embeddings"""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for texts")
    parser.add_argument("--input", required=True, help="Input JSON file with texts")
    parser.add_argument("--output", required=True, help="Output file to save embeddings")
    parser.add_argument("--text-field", default="text", help="Field name containing the text")
    parser.add_argument("--embedding-field", default="embedding", help="Field name to store the embedding")

    args = parser.parse_args()

    generate_and_save_embeddings(
        args.input,
        args.output,
        args.text_field,
        args.embedding_field
    )

if __name__ == "__main__":
    main()
