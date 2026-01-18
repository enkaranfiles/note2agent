"""
Vector Store Wrapper

ChromaDB wrapper with local Sentence Transformers embeddings.
Handles document indexing, retrieval, and metadata filtering.
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Vector store using ChromaDB and local Sentence Transformers embeddings.

    Features:
    - Document indexing with local embeddings (completely free)
    - Vector similarity search
    - Metadata filtering
    - Collection management
    """

    def __init__(
        self,
        collection_name: str = "note2agent_docs",
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize vector store with local embeddings.

        Args:
            collection_name: Name of ChromaDB collection
            chromadb_host: ChromaDB server host
            chromadb_port: ChromaDB server port
            embedding_model: Sentence Transformers model name
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize Sentence Transformers model (downloads on first use)
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded")

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

        # Generate embeddings using local model
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
            "model": self.embedding_model_name
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
        Generate embeddings using local Sentence Transformers model.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Generate embeddings (handles batching automatically)
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Convert numpy arrays to lists
        return embeddings.tolist()

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
    chromadb_port: int = 8000,
    embedding_model: str = "BAAI/bge-small-en-v1.5"
) -> VectorStore:
    """
    Factory function to create a vector store instance.

    Args:
        collection_name: Name of ChromaDB collection
        chromadb_host: ChromaDB server host
        chromadb_port: ChromaDB server port
        embedding_model: Sentence Transformers model name

    Returns:
        Initialized VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        chromadb_host=chromadb_host,
        chromadb_port=chromadb_port,
        embedding_model=embedding_model
    )


# TODO: Add re-ranking functionality
# TODO: Add hybrid search (vector + BM25)
# TODO: Add batch update functionality
# TODO: Add support for other embedding models
