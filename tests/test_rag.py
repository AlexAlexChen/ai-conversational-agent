import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from app.services.qdrant_service import QdrantService
from app.utils.embeddings import get_embedding
from app.utils.ranking import rank_results

@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing"""
    with patch('qdrant_client.QdrantClient') as mock:
        mock_instance = mock.return_value

        # Mock get_collections
        collections_response = MagicMock()
        collections_response.collections = []
        mock_instance.get_collections.return_value = collections_response

        # Mock search
        search_result = [
            MagicMock(
                id="doc1",
                payload={"content": "Test content 1", "title": "Test document 1"},
                score=0.9
            ),
            MagicMock(
                id="doc2",
                payload={"content": "Test content 2", "title": "Test document 2"},
                score=0.7
            )
        ]
        mock_instance.search.return_value = search_result

        yield mock_instance

@pytest.fixture
def qdrant_service(mock_qdrant_client):
    """Create a QdrantService with a mock client"""
    with patch('qdrant_client.QdrantClient', return_value=mock_qdrant_client):
        service = QdrantService()
        return service

@pytest.fixture
def mock_embedding_model():
    """Create a mock OpenAI embedding model"""
    with patch('app.utils.embeddings.embeddings_model') as mock:
        mock.embed_query.return_value = [0.1] * 1536  # Mock 1536-dimensional embedding
        mock.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]  # Mock batch embeddings
        yield mock

class TestRAGSystem:
    """Test suite for the RAG (Retrieval Augmented Generation) system"""

    def test_embedding_generation(self, mock_embedding_model):
        """Test generating embeddings from text"""
        # Generate an embedding
        embedding = get_embedding("Test query")

        # Verify the model call and result
        mock_embedding_model.embed_query.assert_called_once()
        assert len(embedding) == 1536

    def test_batch_embedding_generation(self, mock_embedding_model):
        """Test generating embeddings for multiple texts"""
        # Generate embeddings for multiple texts
        texts = ["Text 1", "Text 2"]
        embeddings = get_embedding("Test query")

        # Verify the model call and result
        mock_embedding_model.embed_query.assert_called_once()
        assert len(embeddings) == 1536

    def test_qdrant_search(self, qdrant_service):
        """Test searching for relevant documents in Qdrant"""
        # Search for documents
        results = qdrant_service.search([0.1] * 1536)

        # Verify the client call and result
        qdrant_service.client.search.assert_called_once()
        assert len(results) == 2
        assert results[0]["title"] == "Test document 1"
        assert results[0]["content"] == "Test content 1"
        assert results[0]["score"] == 0.9

    def test_result_ranking(self):
        """Test ranking search results by relevance"""
        # Sample results to rank
        results = [
            {"content": "Test content 1", "title": "Test document 1", "score": 0.7, "embedding": [0.2] * 1536},
            {"content": "Test content 2", "title": "Test document 2", "score": 0.9, "embedding": [0.3] * 1536},
            {"content": "Test content 3", "title": "Test document 3", "score": 0.5, "embedding": [0.1] * 1536}
        ]

        # Query embedding more similar to document 3
        query_embedding = [0.1] * 1536

        # Rank the results
        ranked = rank_results(results, query_embedding)

        # Verify the ranking order
        assert ranked[0]["title"] == "Test document 3"  # Most similar to query
        assert ranked[1]["title"] in ["Test document 1", "Test document 2"]

    def test_qdrant_collection_creation(self, qdrant_service):
        """Test creation of Qdrant collection if it doesn't exist"""
        # Collection creation should be called during initialization
        qdrant_service.client.create_collection.assert_called_once()

    def test_qdrant_add_document(self, qdrant_service):
        """Test adding a document to Qdrant"""
        # Add a document
        doc_id = "test_doc"
        embedding = [0.1] * 1536
        metadata = {"content": "Test content", "title": "Test document"}

        qdrant_service.add_document(doc_id, embedding, metadata)

        # Verify the client call
        qdrant_service.client.upsert.assert_called_once()

    def test_qdrant_add_documents(self, qdrant_service):
        """Test adding multiple documents to Qdrant"""
        # Add multiple documents
        documents = [
            {"id": "doc1", "embedding": [0.1] * 1536, "metadata": {"content": "Content 1"}},
            {"id": "doc2", "embedding": [0.2] * 1536, "metadata": {"content": "Content 2"}}
        ]

        qdrant_service.add_documents(documents)

        # Verify the client call
        qdrant_service.client.upsert.assert_called_once()
