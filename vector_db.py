"""
Vector Database Module for AI-Rabbi

This module provides a PostgreSQL/pgvector-based vector storage solution
as a replacement for Pinecone. It supports both pgvector (production)
and Pinecone (legacy) backends.

Usage:
    from vector_db import VectorDB

    # For pgvector (recommended)
    db = VectorDB(backend='pgvector')

    # For Pinecone (legacy)
    db = VectorDB(backend='pinecone')
"""

import os
from abc import ABC, abstractmethod


class VectorDBBase(ABC):
    """Abstract base class for vector database implementations."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """Query for similar vectors."""
        pass

    @abstractmethod
    def upsert(self, vectors: list[dict]) -> None:
        """Insert or update vectors."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


class PgVectorDB(VectorDBBase):
    """PostgreSQL with pgvector extension implementation."""

    def __init__(self, connection_string: str = None):
        """
        Initialize pgvector connection.

        Args:
            connection_string: PostgreSQL connection string.
                              Defaults to DATABASE_URL env var.
        """
        try:
            import psycopg2
            from psycopg2.extras import execute_values
        except ImportError:
            raise ImportError("psycopg2 is required for pgvector. Install with: pip install psycopg2-binary")

        self.psycopg2 = psycopg2
        self.execute_values = execute_values
        self.conn_string = connection_string or os.environ.get("DATABASE_URL")

        if not self.conn_string:
            raise ValueError("DATABASE_URL environment variable is required for pgvector")

        self.conn = None
        self._ensure_schema()

    def _get_connection(self):
        """Get or create database connection."""
        if self.conn is None or self.conn.closed:
            self.conn = self.psycopg2.connect(self.conn_string)
        return self.conn

    def _ensure_schema(self):
        """Ensure the required schema exists."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create embeddings table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS midrash_embeddings (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    segment_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create index for fast similarity search (if not exists)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS midrash_embeddings_embedding_idx
                ON midrash_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)

            conn.commit()

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Query for the most similar vectors using cosine similarity.

        Args:
            embedding: Query embedding vector (1536 dimensions)
            top_k: Number of results to return

        Returns:
            List of dicts with id, filename, segment_number, text, score
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Convert embedding to pgvector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            cur.execute("""
                SELECT id, filename, segment_number, text,
                       1 - (embedding <=> %s::vector) as score
                FROM midrash_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, top_k))

            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'segment_number': row[2],
                    'text': row[3],
                    'score': row[4]
                })

            return results

    def upsert(self, vectors: list[dict]) -> None:
        """
        Insert or update vectors in the database.

        Args:
            vectors: List of dicts with keys: id, filename, segment_number, text, embedding
        """
        if not vectors:
            return

        conn = self._get_connection()
        with conn.cursor() as cur:
            # Prepare values for upsert
            values = []
            for v in vectors:
                embedding_str = '[' + ','.join(map(str, v['embedding'])) + ']'
                values.append((
                    v['id'],
                    v['filename'],
                    v['segment_number'],
                    v['text'],
                    embedding_str
                ))

            # Upsert using ON CONFLICT
            self.execute_values(
                cur,
                """
                INSERT INTO midrash_embeddings (id, filename, segment_number, text, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    segment_number = EXCLUDED.segment_number,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                values,
                template="(%s, %s, %s, %s, %s::vector)"
            )

            conn.commit()

    def count(self) -> int:
        """Return the number of vectors in the database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM midrash_embeddings")
            return cur.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None


class PineconeDB(VectorDBBase):
    """Pinecone vector database implementation (legacy)."""

    def __init__(self, api_key: str = None, environment: str = None, index_name: str = 'midrash'):
        """
        Initialize Pinecone connection.

        Args:
            api_key: Pinecone API key. Defaults to PINECONE_API_KEY env var.
            environment: Pinecone environment. Defaults to PINECONE_ENV env var.
            index_name: Name of the Pinecone index.
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError("pinecone-client is required. Install with: pip install pinecone-client==2.2.1")

        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.environment = environment or os.environ.get("PINECONE_ENV")
        self.index_name = index_name

        if not self.api_key or not self.environment:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENV environment variables are required")

        pinecone.init(api_key=self.api_key, environment=self.environment)

        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=1536,
                metric='cosine',
                metadata_config={'indexed': ['channel_id', 'published']}
            )

        self.index = pinecone.Index(self.index_name)

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Query Pinecone for similar vectors.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of dicts with id and score (metadata fetched separately)
        """
        res = self.index.query(embedding, top_k=top_k, include_metadata=True)

        results = []
        for match in res['matches']:
            results.append({
                'id': int(match['id']),
                'score': match['score'],
                'metadata': match.get('metadata', {})
            })

        return results

    def upsert(self, vectors: list[dict]) -> None:
        """
        Upsert vectors to Pinecone.

        Args:
            vectors: List of dicts with keys: id, embedding, and optional metadata
        """
        if not vectors:
            return

        # Format for Pinecone
        pinecone_vectors = []
        for v in vectors:
            metadata = v.get('metadata', {})
            if 'segment_number' in v:
                metadata['segment_number'] = v['segment_number']
            if 'filename' in v:
                metadata['filename'] = v['filename']

            pinecone_vectors.append((
                str(v['id']),
                v['embedding'],
                metadata
            ))

        self.index.upsert(vectors=pinecone_vectors)

    def close(self) -> None:
        """No-op for Pinecone (no persistent connection to close)."""
        pass


class VectorDB:
    """
    Factory class for vector database backends.

    Supports 'pgvector' (recommended) and 'pinecone' (legacy) backends.
    """

    def __init__(self, backend: str = None, **kwargs):
        """
        Initialize the vector database.

        Args:
            backend: 'pgvector' or 'pinecone'. Defaults to VECTOR_DB_BACKEND env var,
                    or 'pgvector' if not set.
            **kwargs: Additional arguments passed to the backend constructor.
        """
        self.backend_name = backend or os.environ.get("VECTOR_DB_BACKEND", "pgvector")

        if self.backend_name == "pgvector":
            self._db = PgVectorDB(**kwargs)
        elif self.backend_name == "pinecone":
            self._db = PineconeDB(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}. Use 'pgvector' or 'pinecone'.")

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """Query for similar vectors."""
        return self._db.query(embedding, top_k)

    def upsert(self, vectors: list[dict]) -> None:
        """Insert or update vectors."""
        return self._db.upsert(vectors)

    def close(self) -> None:
        """Close database connection."""
        return self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
