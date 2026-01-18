"""
Vector Store Wrapper

ChromaDB wrapper with Voyage AI embeddings integration.
Handles document indexing, retrieval, and metadata filtering.
"""

import os
from typing import List, Dict, Optional
import voyageai
import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Vector store using ChromaDB and Voyage AI embeddings.

    Features:
    - Document indexing with Voyage AI embeddings
    - Vector similarity search
    - Metadata filtering
    - Collection management
    """

    def __init__(
        self,
        collection_name: str = "note2agent_docs",
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        voyage_api_key: Optional[str] = None,
        voyage_model: str = "voyage-2"
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of ChromaDB collection
            chromadb_host: ChromaDB server host
            chromadb_port: ChromaDB server port
            voyage_api_key: Voyage AI API key (or set VOYAGE_API_KEY env var)
            voyage_model: Voyage AI model name
        """
        self.collection_name = collection_name
        self.voyage_model = voyage_model

        # Initialize Voyage AI client
        api_key = voyage_api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Voyage AI API key not provided")

        self.voyage_client = voyageai.Client(api_key=api_key)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=chromadb_host,
            port=chromadb_port
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Note2Agent document embeddings"}
        )

    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
                   Format: [{"text": str, "metadata": dict}, ...]
        """
        if not chunks:
            return

        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Generate embeddings using Voyage AI
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self._generate_embeddings(texts)

        # Generate unique IDs for each chunk
        ids = [
            f"{metadata['source']}_{metadata.get('page', 0)}_{metadata['chunk_index']}"
            for metadata in metadatas
        ]

        # Add to ChromaDB
        print(f"Adding {len(texts)} chunks to ChromaDB...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Successfully indexed {len(texts)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (ChromaDB where clause)

        Returns:
            List of results with text, metadata, and similarity score
        """
        # Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        # Format results
        formatted_results = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "id": results['ids'][0][i]
                })

        return formatted_results

    def delete_by_source(self, source: str) -> None:
        """
        Delete all chunks from a specific source file.

        Args:
            source: Source filename to delete
        """
        self.collection.delete(
            where={"source": source}
        )
        print(f"Deleted all chunks from: {source}")

    def delete_by_hash(self, file_hash: str) -> None:
        """
        Delete all chunks with a specific file hash.

        Args:
            file_hash: File hash to delete
        """
        self.collection.delete(
            where={"file_hash": file_hash}
        )
        print(f"Deleted all chunks with hash: {file_hash}")

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "model": self.voyage_model
        }

    def clear_collection(self) -> None:
        """
        Delete all documents from the collection.
        """
        # Delete the collection
        self.chroma_client.delete_collection(name=self.collection_name)

        # Recreate empty collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Note2Agent document embeddings"}
        )

        print(f"Cleared collection: {self.collection_name}")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Voyage AI.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Voyage AI has batch limits, so we process in batches
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Call Voyage AI API
            result = self.voyage_client.embed(
                batch,
                model=self.voyage_model,
                input_type="document"
            )

            all_embeddings.extend(result.embeddings)

        return all_embeddings

    def health_check(self) -> bool:
        """
        Check if ChromaDB is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.chroma_client.heartbeat()
            return True
        except Exception as e:
            print(f"ChromaDB health check failed: {str(e)}")
            return False


# Utility functions

def create_vector_store(
    collection_name: str = "note2agent_docs",
    chromadb_host: str = "localhost",
    chromadb_port: int = 8000
) -> VectorStore:
    """
    Factory function to create a vector store instance.

    Args:
        collection_name: Name of ChromaDB collection
        chromadb_host: ChromaDB server host
        chromadb_port: ChromaDB server port

    Returns:
        Initialized VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        chromadb_host=chromadb_host,
        chromadb_port=chromadb_port
    )


# TODO: Add re-ranking functionality
# TODO: Add hybrid search (vector + BM25)
# TODO: Add batch update functionality
# TODO: Add incremental indexing with change detection
