from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import Dict, List, Any, Optional, Union
import uuid
import logging

from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class QdrantService:
    """Service for Qdrant vector store operations"""

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 collection_name: Optional[str] = None):
        """Initialize the Qdrant service

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Vector collection name
        """
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.collection_name = collection_name or QDRANT_COLLECTION

        # Initialize the Qdrant client
        self._init_client()

        # Initialize collection if it doesn't exist
        self._init_collection()

    def _init_client(self) -> None:
        """Initialize the Qdrant client"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def _init_collection(self) -> None:
        """Initialize the vector collection if it doesn't exist"""
        try:
            # Get existing collections
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            # Create the collection if it doesn't exist
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )

                logger.info(f"Collection {self.collection_name} created")
            else:
                logger.info(f"Collection {self.collection_name} exists")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {str(e)}")
            raise

    def add_document(self,
                     document_id: Optional[str] = None,
                     embedding: List[float] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """Add a document to the vector store

        Args:
            document_id: Document ID (generated if not provided)
            embedding: Vector embedding
            metadata: Document metadata

        Returns:
            Document ID
        """
        if not embedding:
            raise ValueError("Embedding vector is required")

        if not metadata:
            metadata = {}

        # Generate a document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())

        try:
            # Create point for Qdrant
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload=metadata
            )

            # Upsert the point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Document {document_id} added to Qdrant")

            return document_id
        except Exception as e:
            logger.error(f"Failed to add document to Qdrant: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents to the vector store

        Args:
            documents: List of document dictionaries with 'id', 'embedding', and 'metadata'

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        try:
            # Prepare points for Qdrant
            points = []
            document_ids = []

            for doc in documents:
                # Check for required fields
                if 'embedding' not in doc:
                    raise ValueError(f"Document missing 'embedding' field: {doc}")

                # Generate an ID if not provided
                doc_id = doc.get('id', str(uuid.uuid4()))
                document_ids.append(doc_id)

                # Create point
                points.append(PointStruct(
                    id=doc_id,
                    vector=doc['embedding'],
                    payload=doc.get('metadata', {})
                ))

            # Upload in batches (Qdrant recommends max 100 points per batch)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            logger.info(f"Added {len(documents)} documents to Qdrant")

            return document_ids
        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {str(e)}")
            raise

    def search(self,
               embedding: List[float],
               limit: int = 5,
               filter_condition: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents

        Args:
            embedding: Query vector embedding
            limit: Maximum number of results to return
            filter_condition: Optional filter for the search

        Returns:
            List of dictionaries with content and metadata
        """
        if not embedding:
            raise ValueError("Embedding vector is required")

        try:
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                query_filter=filter_condition
            )

            # Format results
            formatted_results = []

            for result in search_result:
                # Extract payload fields
                payload = result.payload
                formatted_result = {
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "url": payload.get("url", ""),
                    "source": payload.get("source", "vector_store"),
                    "score": result.score
                }

                # Include all other payload fields
                for key, value in payload.items():
                    if key not in formatted_result:
                        formatted_result[key] = value

                formatted_results.append(formatted_result)

            logger.debug(f"Found {len(formatted_results)} similar documents")

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {str(e)}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document_id]
            )

            logger.debug(f"Document {document_id} deleted from Qdrant")

            return True
        except Exception as e:
            logger.error(f"Failed to delete document from Qdrant: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection

        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Format the information
            info = {
                "name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count
            }

            return info
        except Exception as e:
            logger.error(f"Failed to get Qdrant collection info: {str(e)}")
            return {}

# Dependency for FastAPI
def get_qdrant_service():
    """Get a Qdrant service instance for dependency injection"""
    return QdrantService()
