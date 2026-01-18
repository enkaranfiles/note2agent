"""
Knowledge Base Manager

Manages document indexing, refresh, and change detection.
Handles incremental updates when documents are added/modified.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from core.document_processor import DocumentProcessor, process_directory
from core.vector_store import VectorStore


class KnowledgeBase:
    """
    Manages the document knowledge base with change detection.

    Features:
    - Tracks indexed files and their hashes
    - Incremental indexing (only updates changed files)
    - Full refresh capability
    - Statistics and status reporting
    """

    def __init__(
        self,
        vector_store: VectorStore,
        document_processor: Optional[DocumentProcessor] = None,
        index_file: str = "./data/index_metadata.json"
    ):
        """
        Initialize knowledge base manager.

        Args:
            vector_store: VectorStore instance
            document_processor: DocumentProcessor instance (creates default if None)
            index_file: Path to file tracking indexed documents
        """
        self.vector_store = vector_store
        self.document_processor = document_processor or DocumentProcessor()
        self.index_file = Path(index_file)

        # Load or create index metadata
        self.index_metadata = self._load_index_metadata()

    def refresh(
        self,
        documents_path: str,
        incremental: bool = True,
        force: bool = False
    ) -> Dict:
        """
        Refresh the knowledge base from a documents directory.

        Args:
            documents_path: Path to documents directory
            incremental: If True, only process changed files
            force: If True, re-index all files regardless of changes

        Returns:
            Dictionary with refresh statistics
        """
        print(f"\n{'='*60}")
        print(f"Knowledge Base Refresh: {documents_path}")
        print(f"Mode: {'Force All' if force else 'Incremental' if incremental else 'Full'}")
        print(f"{'='*60}\n")

        docs_path = Path(documents_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents path not found: {documents_path}")

        stats = {
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": []
        }

        # Find all supported files
        supported_extensions = {".pdf", ".md", ".markdown", ".txt"}
        all_files = [
            f for f in docs_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        stats["total_files"] = len(all_files)
        print(f"Found {len(all_files)} document(s)\n")

        # Track current files
        current_file_paths = {str(f.absolute()) for f in all_files}

        # Handle deleted files (files in index but not in directory)
        if incremental and not force:
            deleted_files = self._handle_deleted_files(current_file_paths)
            stats["deleted_files"] = len(deleted_files)

        # Process each file
        for file_path in all_files:
            try:
                should_process = self._should_process_file(
                    file_path,
                    incremental=incremental,
                    force=force
                )

                if not should_process:
                    print(f"⏭️  Skipped (unchanged): {file_path.name}")
                    stats["skipped_files"] += 1
                    continue

                # Process the file
                print(f"📄 Processing: {file_path.name}")
                chunks = self.document_processor.process_file(str(file_path))

                if not chunks:
                    print(f"⚠️  No content extracted from: {file_path.name}")
                    continue

                # Get file hash from first chunk's metadata
                file_hash = chunks[0]["metadata"]["file_hash"]

                # Delete old version if exists
                if incremental:
                    self.vector_store.delete_by_hash(file_hash)

                # Index new chunks
                self.vector_store.add_documents(chunks)

                # Update index metadata
                self._update_index_metadata(file_path, file_hash, len(chunks))

                stats["processed_files"] += 1
                stats["total_chunks"] += len(chunks)

                print(f"✅ Indexed: {file_path.name} ({len(chunks)} chunks)\n")

            except Exception as e:
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                print(f"❌ {error_msg}\n")
                stats["errors"].append(error_msg)

        # Save updated index metadata
        self._save_index_metadata()

        # Print summary
        print(f"\n{'='*60}")
        print("Refresh Complete")
        print(f"{'='*60}")
        print(f"Total files found: {stats['total_files']}")
        print(f"Processed: {stats['processed_files']}")
        print(f"Skipped (unchanged): {stats['skipped_files']}")
        print(f"Deleted: {stats['deleted_files']}")
        print(f"Total chunks indexed: {stats['total_chunks']}")
        if stats["errors"]:
            print(f"Errors: {len(stats['errors'])}")
        print(f"{'='*60}\n")

        return stats

    def clear(self) -> None:
        """
        Clear the entire knowledge base.
        """
        print("Clearing knowledge base...")
        self.vector_store.clear_collection()
        self.index_metadata = {"files": {}, "last_updated": None}
        self._save_index_metadata()
        print("Knowledge base cleared successfully")

    def get_status(self) -> Dict:
        """
        Get knowledge base status and statistics.

        Returns:
            Dictionary with status information
        """
        vector_stats = self.vector_store.get_collection_stats()

        status = {
            "vector_store": vector_stats,
            "indexed_files": len(self.index_metadata.get("files", {})),
            "last_updated": self.index_metadata.get("last_updated"),
            "chromadb_healthy": self.vector_store.health_check()
        }

        return status

    def list_indexed_files(self) -> List[Dict]:
        """
        List all indexed files with their metadata.

        Returns:
            List of file information dictionaries
        """
        files = []
        for file_path, metadata in self.index_metadata.get("files", {}).items():
            files.append({
                "path": file_path,
                "hash": metadata.get("hash"),
                "chunks": metadata.get("chunks"),
                "indexed_at": metadata.get("indexed_at")
            })

        return files

    def _should_process_file(
        self,
        file_path: Path,
        incremental: bool,
        force: bool
    ) -> bool:
        """
        Determine if a file should be processed.

        Args:
            file_path: Path to file
            incremental: Incremental mode flag
            force: Force re-index flag

        Returns:
            True if file should be processed
        """
        if force:
            return True

        if not incremental:
            return True

        # Check if file is in index
        file_key = str(file_path.absolute())
        if file_key not in self.index_metadata.get("files", {}):
            return True  # New file

        # Check if file hash changed
        current_hash = self.document_processor._compute_file_hash(file_path)
        indexed_hash = self.index_metadata["files"][file_key].get("hash")

        return current_hash != indexed_hash

    def _handle_deleted_files(self, current_file_paths: set) -> List[str]:
        """
        Remove deleted files from vector store and index.

        Args:
            current_file_paths: Set of current file paths

        Returns:
            List of deleted file paths
        """
        deleted_files = []

        indexed_files = list(self.index_metadata.get("files", {}).keys())

        for file_path in indexed_files:
            if file_path not in current_file_paths:
                # File was deleted
                file_hash = self.index_metadata["files"][file_path].get("hash")
                if file_hash:
                    self.vector_store.delete_by_hash(file_hash)

                del self.index_metadata["files"][file_path]
                deleted_files.append(file_path)
                print(f"🗑️  Removed deleted file: {Path(file_path).name}")

        return deleted_files

    def _update_index_metadata(
        self,
        file_path: Path,
        file_hash: str,
        num_chunks: int
    ) -> None:
        """
        Update index metadata for a file.

        Args:
            file_path: Path to file
            file_hash: File hash
            num_chunks: Number of chunks indexed
        """
        file_key = str(file_path.absolute())

        self.index_metadata["files"][file_key] = {
            "hash": file_hash,
            "chunks": num_chunks,
            "indexed_at": datetime.now().isoformat()
        }

        self.index_metadata["last_updated"] = datetime.now().isoformat()

    def _load_index_metadata(self) -> Dict:
        """
        Load index metadata from file.

        Returns:
            Index metadata dictionary
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load index metadata: {e}")

        return {"files": {}, "last_updated": None}

    def _save_index_metadata(self) -> None:
        """
        Save index metadata to file.
        """
        # Ensure directory exists
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_file, 'w') as f:
            json.dump(self.index_metadata, f, indent=2)


# Utility function

def create_knowledge_base(
    vector_store: VectorStore,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> KnowledgeBase:
    """
    Factory function to create a knowledge base instance.

    Args:
        vector_store: VectorStore instance
        chunk_size: Chunk size for document processor
        chunk_overlap: Chunk overlap for document processor

    Returns:
        Initialized KnowledgeBase instance
    """
    document_processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return KnowledgeBase(
        vector_store=vector_store,
        document_processor=document_processor
    )


# TODO: Add progress bars for large document sets
# TODO: Add parallel processing for multiple files
# TODO: Add support for custom metadata fields
# TODO: Add rollback functionality for failed refreshes
