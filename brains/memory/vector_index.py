"""
vector_index.py
~~~~~~~~~~~~~~~

Pluggable vector index for semantic retrieval.

This module provides a local vector index that can:
- Store embeddings for memories/episodes
- Retrieve top-K semantically similar items for a query
- Be optional and pluggable (different backends)

The default implementation uses a simple in-memory index with optional
SQLite persistence. More sophisticated backends (FAISS, ChromaDB) can
be swapped in via the EmbeddingProvider abstraction.

Usage:
    from brains.memory.vector_index import (
        VectorIndex,
        get_vector_index,
        VectorHit,
    )

    # Get the singleton index
    index = get_vector_index()

    # Store a document
    index.upsert("doc1", "What is Python?", {"source": "qa"})

    # Search for similar documents
    hits = index.search("Tell me about Python programming", top_k=5)
    for hit in hits:
        print(f"{hit.id}: {hit.score:.3f}")
"""

from __future__ import annotations

import json
import math
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import sqlite3

from brains.maven_paths import get_reports_path


# Storage paths
VECTOR_DB_PATH = get_reports_path("vector_index.sqlite")
VECTOR_INDEX_PATH = get_reports_path("vector_index.json")


@dataclass
class VectorHit:
    """A single hit from vector search."""
    id: str
    score: float
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class EmbeddingProvider:
    """
    Abstract base for embedding providers.

    Override embed_texts() to use a different embedding model.
    """

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        raise NotImplementedError

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        results = self.embed_texts([text])
        return results[0] if results else []


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    Simple embedding provider using TF-IDF-like token vectors.

    This is a fallback when no LLM embedding is available. It creates
    sparse vectors based on word frequencies and IDF weighting.

    For production use, replace with LLMEmbeddingProvider.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        # Remove very short tokens
        return [t for t in tokens if len(t) > 2]

    def _hash_token(self, token: str, dimension: int) -> int:
        """Hash a token to a dimension index."""
        h = hashlib.md5(token.encode()).hexdigest()
        return int(h, 16) % dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate simple hash-based embeddings."""
        embeddings = []

        for text in texts:
            # Create a sparse vector
            vector = [0.0] * self.dimension
            tokens = self._tokenize(text)

            if not tokens:
                embeddings.append(vector)
                continue

            # Count token frequencies
            freq: Dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

            # Create embedding
            for token, count in freq.items():
                idx = self._hash_token(token, self.dimension)
                # TF-IDF-like weighting
                tf = math.log(1 + count)
                idf = self._idf.get(token, 1.0)
                vector[idx] += tf * idf

            # Normalize
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]

            embeddings.append(vector)

        return embeddings

    def update_idf(self, texts: List[str]) -> None:
        """Update IDF statistics from a corpus."""
        doc_freq: Dict[str, int] = {}

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._doc_count += len(texts)

        for token, df in doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (df + 1)) + 1


class LLMEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using the LLM service.

    This uses the existing LLM infrastructure to generate embeddings.
    Falls back to SimpleEmbeddingProvider if LLM is not available.
    """

    def __init__(self):
        self._fallback = SimpleEmbeddingProvider()
        self._llm_available = self._check_llm()

    def _check_llm(self) -> bool:
        """Check if LLM service is available for embeddings."""
        try:
            from brains.tools.llm_service import llm_service
            return llm_service is not None and hasattr(llm_service, "enabled")
        except ImportError:
            return False

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LLM or fallback."""
        if not self._llm_available:
            return self._fallback.embed_texts(texts)

        try:
            from brains.tools.llm_service import llm_service

            # Try to use LLM for embeddings if available
            # Most LLM APIs don't directly expose embeddings, so we use fallback
            # This is a placeholder for when embedding API is available
            return self._fallback.embed_texts(texts)

        except Exception:
            return self._fallback.embed_texts(texts)


class VectorIndex:
    """
    Local vector index for semantic search.

    Supports multiple backends:
    - "memory": In-memory only (fast, no persistence)
    - "sqlite": SQLite-backed persistence
    - "json": JSON file persistence

    The index stores text documents with their embeddings and supports
    top-K nearest neighbor search using cosine similarity.
    """

    def __init__(
        self,
        backend: str = "sqlite",
        embedding_provider: Optional[EmbeddingProvider] = None,
        dimension: int = 384,
    ):
        self.backend = backend
        self.dimension = dimension
        self._provider = embedding_provider or SimpleEmbeddingProvider(dimension)

        # In-memory storage
        self._vectors: Dict[str, List[float]] = {}
        self._texts: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Initialize backend
        if backend == "sqlite":
            self._init_sqlite()
            self._load_from_sqlite()
        elif backend == "json":
            self._load_from_json()

        print(f"[VECTOR_INDEX] Initialized with backend={backend}, dimension={dimension}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(VECTOR_DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                text TEXT,
                vector BLOB,
                metadata TEXT,
                created_at REAL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vectors_created
            ON vectors(created_at)
        """)

        conn.commit()
        conn.close()

    def _load_from_sqlite(self) -> None:
        """Load vectors from SQLite."""
        try:
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()

            cursor.execute("SELECT id, text, vector, metadata FROM vectors")
            for row in cursor.fetchall():
                doc_id, text, vector_blob, metadata_json = row
                self._texts[doc_id] = text
                self._vectors[doc_id] = json.loads(vector_blob)
                self._metadata[doc_id] = json.loads(metadata_json) if metadata_json else {}

            conn.close()
            print(f"[VECTOR_INDEX] Loaded {len(self._vectors)} vectors from SQLite")
        except Exception as e:
            print(f"[VECTOR_INDEX] Failed to load from SQLite: {e}")

    def _save_to_sqlite(self, doc_id: str) -> None:
        """Save a single vector to SQLite."""
        try:
            conn = sqlite3.connect(str(VECTOR_DB_PATH))
            cursor = conn.cursor()

            vector_json = json.dumps(self._vectors.get(doc_id, []))
            metadata_json = json.dumps(self._metadata.get(doc_id, {}))

            cursor.execute("""
                INSERT OR REPLACE INTO vectors (id, text, vector, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                doc_id,
                self._texts.get(doc_id, ""),
                vector_json,
                metadata_json,
                time.time()
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[VECTOR_INDEX] Failed to save to SQLite: {e}")

    def _load_from_json(self) -> None:
        """Load vectors from JSON file."""
        try:
            if VECTOR_INDEX_PATH.exists():
                data = json.loads(VECTOR_INDEX_PATH.read_text())
                self._vectors = data.get("vectors", {})
                self._texts = data.get("texts", {})
                self._metadata = data.get("metadata", {})
                print(f"[VECTOR_INDEX] Loaded {len(self._vectors)} vectors from JSON")
        except Exception as e:
            print(f"[VECTOR_INDEX] Failed to load from JSON: {e}")

    def _save_to_json(self) -> None:
        """Save all vectors to JSON file."""
        try:
            VECTOR_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "vectors": self._vectors,
                "texts": self._texts,
                "metadata": self._metadata,
            }
            VECTOR_INDEX_PATH.write_text(json.dumps(data))
        except Exception as e:
            print(f"[VECTOR_INDEX] Failed to save to JSON: {e}")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(v1) != len(v2):
            return 0.0

        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert or update a document in the index.

        Args:
            doc_id: Unique document ID
            text: Text content to embed
            metadata: Optional metadata dict

        Returns:
            True if successful
        """
        try:
            # Generate embedding
            embedding = self._provider.embed_text(text)

            # Store
            self._vectors[doc_id] = embedding
            self._texts[doc_id] = text
            self._metadata[doc_id] = metadata or {}

            # Persist
            if self.backend == "sqlite":
                self._save_to_sqlite(doc_id)
            elif self.backend == "json":
                self._save_to_json()

            return True
        except Exception as e:
            print(f"[VECTOR_INDEX] Upsert failed: {e}")
            return False

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the index.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if document was deleted
        """
        if doc_id not in self._vectors:
            return False

        del self._vectors[doc_id]
        self._texts.pop(doc_id, None)
        self._metadata.pop(doc_id, None)

        # Persist deletion
        if self.backend == "sqlite":
            try:
                conn = sqlite3.connect(str(VECTOR_DB_PATH))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vectors WHERE id = ?", (doc_id,))
                conn.commit()
                conn.close()
            except Exception:
                pass
        elif self.backend == "json":
            self._save_to_json()

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorHit]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Maximum number of results
            filters: Optional metadata filters (key=value matches)
            min_score: Minimum similarity score

        Returns:
            List of VectorHit objects, sorted by score descending
        """
        if not self._vectors:
            return []

        # Embed query
        query_vector = self._provider.embed_text(query)
        if not query_vector:
            return []

        # Compute similarities
        scores: List[Tuple[str, float]] = []

        for doc_id, doc_vector in self._vectors.items():
            # Apply filters
            if filters:
                doc_meta = self._metadata.get(doc_id, {})
                if not all(doc_meta.get(k) == v for k, v in filters.items()):
                    continue

            score = self._cosine_similarity(query_vector, doc_vector)
            if score >= min_score:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        hits = []
        for doc_id, score in scores[:top_k]:
            hit = VectorHit(
                id=doc_id,
                score=score,
                text=self._texts.get(doc_id, ""),
                metadata=self._metadata.get(doc_id, {}),
            )
            hits.append(hit)

        return hits

    def count(self) -> int:
        """Get the number of documents in the index."""
        return len(self._vectors)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "backend": self.backend,
            "dimension": self.dimension,
            "document_count": self.count(),
            "has_vectors": self.count() > 0,
        }

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._vectors.clear()
        self._texts.clear()
        self._metadata.clear()

        if self.backend == "sqlite":
            try:
                conn = sqlite3.connect(str(VECTOR_DB_PATH))
                cursor = conn.cursor()
                cursor.execute("DELETE FROM vectors")
                conn.commit()
                conn.close()
            except Exception:
                pass
        elif self.backend == "json":
            self._save_to_json()


# Module-level singleton
_vector_index: Optional[VectorIndex] = None
_vector_index_enabled: bool = True


def is_vector_index_enabled() -> bool:
    """Check if vector index is enabled."""
    global _vector_index_enabled
    return _vector_index_enabled


def set_vector_index_enabled(enabled: bool) -> None:
    """Enable or disable vector index."""
    global _vector_index_enabled
    _vector_index_enabled = enabled


def get_vector_index(backend: str = "sqlite") -> VectorIndex:
    """Get the singleton vector index."""
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex(backend=backend)
    return _vector_index


def search_similar(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    min_score: float = 0.0,
) -> List[VectorHit]:
    """Convenience function to search the vector index."""
    if not is_vector_index_enabled():
        return []
    return get_vector_index().search(query, top_k=top_k, filters=filters, min_score=min_score)


def index_memory(
    memory_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function to index a memory."""
    if not is_vector_index_enabled():
        return False
    return get_vector_index().upsert(memory_id, text, metadata)


# Public API
__all__ = [
    "VectorIndex",
    "VectorHit",
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
    "LLMEmbeddingProvider",
    "get_vector_index",
    "search_similar",
    "index_memory",
    "is_vector_index_enabled",
    "set_vector_index_enabled",
]
