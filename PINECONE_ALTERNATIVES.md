# Pinecone Alternatives for AI-Rabbi

A comprehensive analysis of cost-effective vector database alternatives to Pinecone that can be self-hosted on Railway.

## Executive Summary

**Recommended Solution: PostgreSQL with pgvector on Railway**

For the AI-Rabbi application, **pgvector** is the optimal choice because:
- Lowest cost ($5-10/month on Railway)
- One-click deployment on Railway with official templates
- Consolidates vector storage with existing SQLite logging (can migrate both to one Postgres instance)
- Sufficient for the ~30,000-50,000 vectors in the Midrash dataset
- Native SQL interface simplifies debugging and maintenance

## Current Pinecone Usage Analysis

Based on `app.py` and `generate-midrash-embeddings.py`:

| Aspect | Current Implementation |
|--------|----------------------|
| **Vector Dimensions** | 1536 (OpenAI ada-002) |
| **Similarity Metric** | Cosine |
| **Query Pattern** | `query(vector, top_k=3)` |
| **Upsert Pattern** | Batch upsert of 100 vectors |
| **Index Name** | `midrash` |
| **Estimated Vectors** | ~30,000-50,000 segments |
| **Metadata** | `length`, `filename`, `segment_number` |

## Alternatives Comparison

### 1. PostgreSQL + pgvector (Recommended)

| Criteria | Details |
|----------|---------|
| **Monthly Cost** | $5-15 on Railway |
| **Railway Support** | [Official one-click template](https://railway.com/deploy/pgvector-latest) |
| **Setup Complexity** | Low - SQL-based, familiar interface |
| **Scale Limit** | Up to 10M vectors comfortably |
| **Latency** | ~10-50ms for small datasets |

**Pros:**
- Can consolidate SQLite logging into the same Postgres instance
- Full SQL query capabilities for debugging
- Railway auto-scales up to 32 vCPU / 32 GB RAM
- Supports HNSW and IVFFlat indexing
- No vendor lock-in

**Cons:**
- Slightly higher latency than specialized vector DBs at scale
- Manual index tuning for optimal performance

### 2. Qdrant

| Criteria | Details |
|----------|---------|
| **Monthly Cost** | $5-10 on Railway (self-hosted) |
| **Railway Support** | [Official template](https://railway.com/deploy/qdrant) |
| **Setup Complexity** | Medium - REST/gRPC API |
| **Scale Limit** | Billions of vectors |
| **Latency** | ~5-20ms |

**Pros:**
- Written in Rust - very fast
- Purpose-built for vector search
- Rich filtering capabilities
- Active development community

**Cons:**
- Separate service to manage
- Learning curve for Qdrant-specific API
- Requires persistent storage configuration

### 3. ChromaDB

| Criteria | Details |
|----------|---------|
| **Monthly Cost** | Free (embedded) or $5-10 (server mode) |
| **Railway Support** | Requires custom Docker setup |
| **Setup Complexity** | Low - Python-native API |
| **Scale Limit** | Up to 1M vectors (single node) |
| **Latency** | ~50-100ms |

**Pros:**
- Simplest Python API
- Can run embedded (no separate server)
- Great for prototyping

**Cons:**
- Not officially supported on Railway
- Performance limitations at scale
- Single-node architecture

### 4. Milvus Lite

| Criteria | Details |
|----------|---------|
| **Monthly Cost** | Free (embedded) |
| **Railway Support** | Limited - heavy resource requirements |
| **Setup Complexity** | High |
| **Scale Limit** | Billions of vectors |
| **Latency** | ~5-15ms |

**Pros:**
- Highly scalable
- Rich feature set

**Cons:**
- Overkill for this use case
- Complex deployment
- High memory requirements

## Cost Comparison

| Solution | Monthly Cost | Notes |
|----------|-------------|-------|
| **Pinecone (current)** | $0-70+ | Free tier has dormancy issues, paid starts at $70/mo |
| **pgvector on Railway** | $5-15 | Best value, can consolidate with app DB |
| **Qdrant on Railway** | $5-10 | Good performance/cost ratio |
| **ChromaDB** | $5-10 | If custom Docker works |

## Detailed Migration Roadmap

### Phase 1: Setup & Testing (Local Development)

#### Step 1.1: Install pgvector locally

```bash
# Install PostgreSQL with pgvector extension
# On macOS:
brew install postgresql@16
brew install pgvector

# On Ubuntu:
sudo apt install postgresql-16 postgresql-16-pgvector
```

#### Step 1.2: Create the database schema

```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the midrash embeddings table
CREATE TABLE midrash_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    segment_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX ON midrash_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

#### Step 1.3: Create the new database module

Create `vector_db.py`:

```python
import os
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

class VectorDB:
    def __init__(self):
        self.conn_string = os.environ.get("DATABASE_URL")
        self.conn = None

    def connect(self):
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.conn_string)
        return self.conn

    def upsert_vectors(self, vectors: list[dict]):
        """
        Upsert vectors to the database.
        vectors: list of dicts with keys: id, filename, segment_number, text, embedding
        """
        conn = self.connect()
        with conn.cursor() as cur:
            # Using ON CONFLICT for upsert behavior
            query = """
                INSERT INTO midrash_embeddings (id, filename, segment_number, text, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    segment_number = EXCLUDED.segment_number,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
            """
            values = [
                (v['id'], v['filename'], v['segment_number'], v['text'], v['embedding'])
                for v in vectors
            ]
            execute_values(cur, query, values)
            conn.commit()

    def query(self, embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Query for the most similar vectors using cosine similarity.
        Returns list of dicts with id, filename, segment_number, text, score.
        """
        conn = self.connect()
        with conn.cursor() as cur:
            # Convert embedding to string format for pgvector
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

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
```

### Phase 2: Data Migration Script

Create `migrate_to_pgvector.py`:

```python
import os
import pandas as pd
import openai
from vector_db import VectorDB

# Load existing data
df = pd.read_pickle('output-5sentence_without_embeddings.pkl')

# Initialize connections
openai.api_key = os.environ.get("OPENAI_API_KEY")
vector_db = VectorDB()

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Process in batches
batch_size = 100
total_rows = len(df)

print(f"Migrating {total_rows} vectors to pgvector...")

for i in range(0, total_rows, batch_size):
    batch = df.iloc[i:i+batch_size]
    vectors = []

    for idx, row in batch.iterrows():
        try:
            embedding = get_embedding(row['text'])
            vectors.append({
                'id': idx,
                'filename': row['filename'],
                'segment_number': row['segment_number'],
                'text': row['text'],
                'embedding': embedding
            })
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Retry with truncated text
            embedding = get_embedding(row['text'][:6000])
            vectors.append({
                'id': idx,
                'filename': row['filename'],
                'segment_number': row['segment_number'],
                'text': row['text'][:6000],
                'embedding': embedding
            })

    vector_db.upsert_vectors(vectors)
    print(f"Migrated {min(i + batch_size, total_rows)}/{total_rows} vectors")

print("Migration complete!")
vector_db.close()
```

### Phase 3: Update Application Code

Modify `app.py` to use pgvector instead of Pinecone:

```python
# Remove these imports:
# import pinecone

# Add this import:
from vector_db import VectorDB

# Replace Pinecone initialization with:
vector_db = VectorDB()

# Replace get_relevant_sources function:
def get_relevant_sources(topic):
    # Get embedding from OpenAI
    xq = get_embedding(text=topic)

    # Query pgvector
    results = vector_db.query(xq, top_k=3)

    texts = []
    for r in results:
        texts.append({
            'filename': r['filename'],
            'segment_number': r['segment_number'],
            'text': r['text']
        })
    return texts

# Remove scheduled_task for Pinecone keepalive (no longer needed!)
# pgvector doesn't have dormancy issues
```

### Phase 4: Railway Deployment

#### Step 4.1: Deploy pgvector on Railway

1. Go to [Railway pgvector template](https://railway.com/deploy/pgvector-latest)
2. Click "Deploy Now"
3. Railway will provision a PostgreSQL instance with pgvector enabled
4. Copy the `DATABASE_URL` from the service variables

#### Step 4.2: Update environment variables

Add to Railway service:
```
DATABASE_URL=postgresql://...  # From pgvector service
OPENAI_API_KEY=your-key
LOG_PASSWORD=your-password
```

#### Step 4.3: Update requirements.txt

```
flask==2.3.2
openai==0.27.2
psycopg2-binary==2.9.9
pandas==2.0.1
tiktoken==0.3.3
gunicorn==20.1.0
APScheduler==3.10.4
flask-httpauth==4.8.0
flask-login==0.6.2
werkzeug==2.3.4
```

Remove `pinecone-client` from requirements.

#### Step 4.4: Initialize database on Railway

```bash
# Connect to Railway PostgreSQL
railway run psql $DATABASE_URL

# Run the schema creation SQL from Step 1.2
```

#### Step 4.5: Run migration

```bash
# Set environment variables
export DATABASE_URL="your-railway-postgres-url"
export OPENAI_API_KEY="your-key"

# Run migration script
python migrate_to_pgvector.py
```

### Phase 5: Cleanup

1. Remove Pinecone-related code from `app.py`
2. Remove `pinecone-client` from `requirements.txt`
3. Delete Pinecone environment variables
4. Remove the keepalive scheduler (pgvector doesn't go dormant)
5. Consider migrating SQLite logs to the same Postgres instance

## Optional: Consolidate SQLite to Postgres

Since you'll have Postgres running anyway, you can migrate the logging tables:

```sql
-- Add to your pgvector database
CREATE TABLE request_logs (
    id SERIAL PRIMARY KEY,
    ip_address TEXT,
    real_ip_address TEXT,
    query TEXT,
    collected_messages TEXT,
    user_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE response_logs (
    id SERIAL PRIMARY KEY,
    ip_address TEXT,
    real_ip_address TEXT,
    query TEXT,
    collected_messages TEXT,
    user_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

This eliminates the SQLite dependency entirely and gives you a single database to manage.

## Rollback Plan

If issues arise during migration:

1. Keep Pinecone credentials in a backup `.env` file
2. Maintain the original `app.py` in a git branch
3. The Pinecone index remains intact unless explicitly deleted
4. Can switch back by reverting code changes and restoring env vars

## Timeline Checklist

- [ ] Set up local PostgreSQL with pgvector for testing
- [ ] Create and test `vector_db.py` module locally
- [ ] Deploy pgvector template on Railway
- [ ] Run migration script to populate pgvector
- [ ] Update `app.py` to use new vector_db module
- [ ] Test search functionality on staging
- [ ] Deploy updated app to Railway
- [ ] Verify production functionality
- [ ] Remove Pinecone dependencies
- [ ] (Optional) Migrate SQLite logs to Postgres
- [ ] Delete Pinecone index after successful migration

## Sources

- [Railway pgvector Template](https://railway.com/deploy/pgvector-latest)
- [Railway pgvector Hosting Guide](https://blog.railway.com/p/hosting-postgres-with-pgvector)
- [Railway Qdrant Template](https://railway.com/deploy/qdrant)
- [Qdrant Pricing](https://qdrant.tech/pricing/)
- [Pinecone vs Chroma Comparison](https://aloa.co/ai/comparisons/vector-database-comparison/pinecone-vs-chroma)
- [Vector Database Comparison 2025](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [ChromaDB vs Pinecone Trade-offs](https://www.sourceboxai.com/blog/chromadb-vs-pinecone-the-real-trade-offs-between-self-hosted-and-managed-vector-databases)
