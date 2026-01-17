# AI-Rabbi

An AI-powered chatbot that answers life's questions using Jewish Midrash source texts. The application uses semantic search to find relevant passages from Midrash literature and GPT-4 to provide thoughtful, rabbi-style analysis and responses.

## Architecture Overview

The application supports two vector database backends: **pgvector** (recommended) and **Pinecone** (legacy).

### pgvector Architecture (Recommended)
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flask Web     │────▶│   OpenAI API    │     │   PostgreSQL    │
│   Application   │     │   (GPT-4 +      │     │   + pgvector    │
│   (app.py)      │◀────│   Embeddings)   │     │   (Railway)     │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         └───────────────────────────────────────────────┘
```

### Pinecone Architecture (Legacy)
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flask Web     │────▶│   OpenAI API    │     │    Pinecone     │
│   Application   │     │   (GPT-4 +      │     │  Vector Store   │
│   (app.py)      │◀────│   Embeddings)   │     │   (midrash)     │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         │              ┌─────────────────┐              │
         └─────────────▶│    SQLite DB    │◀─────────────┘
                        │ (search_logs.db)│
                        └─────────────────┘
```

### Core Components

- **Flask Web Server** (`app.py`): Main application handling HTTP requests, user authentication, and streaming responses
- **Vector Database**: pgvector (PostgreSQL) or Pinecone for semantic search over Midrash embeddings
- **OpenAI GPT-4**: Generates rabbi-style responses based on retrieved source texts
- **SQLite**: Logs user requests and responses for analytics

## Project Structure

```
AI-Rabbi/
├── app.py                              # Main Flask application
├── vector_db.py                        # Vector database abstraction (pgvector/Pinecone)
├── migrate_to_pgvector.py              # Migration script from Pinecone to pgvector
├── generate-midrash-embeddings.py      # Script to generate and upload embeddings (legacy)
├── schema.sql                          # SQLite database schema
├── requirements.txt                    # Python dependencies
├── Procfile                            # Heroku/Railway deployment configuration
│
├── CLAUDE.md                           # This file - AI assistant documentation
├── HUMAN_TASKS.md                      # Manual tasks for deployment
├── PINECONE_ALTERNATIVES.md            # Vector DB comparison and migration guide
├── PRODUCT_ROADMAP.md                  # Product strategy and roadmap
│
├── output-5sentence_without_embeddings.pkl  # Preprocessed Midrash data
├── output_without_embeddings.pkl       # Alternative Midrash data
├── MJ prompt.txt                       # Midjourney prompt for logo generation
├── minimize_output_file.py             # Utility (deprecated)
│
├── static/
│   ├── css/
│   │   └── main.css                    # Application styles (comprehensive)
│   └── img/
│       └── airabbi.png                 # Logo image
│
└── templates/
    ├── index.html                      # Main search interface (with markdown rendering)
    ├── request_logs.html               # Admin view for request logs
    └── response_logs.html              # Admin view for response logs
```

## Key Files

### `app.py` - Main Application

The core Flask application with the following key functions:

| Function | Description |
|----------|-------------|
| `get_embedding()` | Creates text embeddings using OpenAI's ada-002 model |
| `get_relevant_sources()` | Queries vector DB for top 3 relevant Midrash passages |
| `search_function()` | Streams GPT-4 responses with Midrash context |
| `is_request_allowed()` | Rate limiting check (5 requests/day per user) |
| `get_remaining_requests()` | Returns remaining daily requests for a user |

**Routes:**
- `GET /` - Main search interface
- `POST /search` - Submit a question and receive streaming response
- `GET /api/remaining-requests` - Check remaining daily requests (JSON)
- `GET /health` - Health check endpoint (JSON)
- `GET /logs` - View request logs (admin, HTTP Basic Auth)
- `GET /response_logs` - View response logs (admin, HTTP Basic Auth)
- `POST /login` - User authentication
- `GET /logout` - User logout

### `vector_db.py` - Vector Database Abstraction

Provides a unified interface for both pgvector and Pinecone backends:

```python
from vector_db import VectorDB

# Auto-selects based on VECTOR_DB_BACKEND env var
db = VectorDB()

# Or explicitly choose backend
db = VectorDB(backend='pgvector')
db = VectorDB(backend='pinecone')

# Query for similar vectors
results = db.query(embedding, top_k=3)

# Upsert vectors
db.upsert(vectors)
```

### `migrate_to_pgvector.py` - Migration Script

Migrates embeddings from pickle files to pgvector:

```bash
# Dry run
python migrate_to_pgvector.py --dry-run

# Full migration
python migrate_to_pgvector.py --batch-size 50

# Resume interrupted migration
python migrate_to_pgvector.py --resume

# Verify migration
python migrate_to_pgvector.py --verify

# Test query
python migrate_to_pgvector.py --test
```

### `templates/index.html` - Frontend

Modern UI with:
- Markdown rendering (via marked.js)
- Example question chips
- Loading states with spinner
- Copy response button
- Rate limit warnings with countdown
- Error handling with retry
- Mobile-responsive design
- Plausible analytics integration

## Environment Variables

### Required (All Deployments)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 and embeddings |
| `SECRET_KEY` | Flask session secret (generate random 32+ chars) |
| `LOG_PASSWORD` | Password for admin log access |

### For pgvector Backend (Recommended)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `VECTOR_DB_BACKEND` | Set to `pgvector` |

### For Pinecone Backend (Legacy)

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_ENV` | Pinecone environment (e.g., `us-west1-gcp`) |
| `VECTOR_DB_BACKEND` | Set to `pinecone` (or omit, it's the default) |

## Development Workflow

### Local Development with pgvector

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://..."
export VECTOR_DB_BACKEND="pgvector"
export OPENAI_API_KEY="your-key"
export SECRET_KEY="your-secret"
export LOG_PASSWORD="your-password"

# Run the application
python app.py
# Server starts at http://localhost:3000
```

### Local Development with Pinecone (Legacy)

```bash
# Set environment variables
export PINECONE_API_KEY="your-key"
export PINECONE_ENV="your-env"
export OPENAI_API_KEY="your-key"
export LOG_PASSWORD="your-password"

# Run the application
python app.py
```

### Deployment (Railway - Recommended)

1. Deploy pgvector template from Railway marketplace
2. Deploy app, linking to pgvector service
3. Set environment variables in Railway dashboard
4. Run migration script locally against Railway DB

See `HUMAN_TASKS.md` for detailed deployment steps.

## Key Technical Details

### Embeddings & Search

- **Model**: `text-embedding-ada-002` (1536 dimensions)
- **Similarity**: Cosine similarity
- **Retrieval**: Top 3 most relevant passages per query
- **Index**: HNSW index for fast approximate nearest neighbor search

### Rate Limiting

- Users are limited to 5 requests per 24-hour period
- Users identified by cookie-based UUID (`user_id`)
- Returns HTTP 429 when limit exceeded
- Frontend shows countdown to reset

### GPT-4 System Prompt

The AI responds as a "Rabbi chatbot analyzing Midrash" with instructions to:
1. Analyze how each source text answers the question
2. Provide step-by-step reasoning
3. Cite sources explicitly
4. Identify conflicting advice between texts
5. Provide a modern rabbinic sermon story
6. Offer theological/philosophical explanation
7. Say "I don't know" if sources don't answer the question

### Frontend Features

- **Markdown Rendering**: Responses rendered with headers, lists, bold, etc.
- **Example Questions**: Clickable chips for common questions
- **Loading State**: Animated spinner while searching
- **Copy Button**: One-click copy of response
- **Error Handling**: Network errors, timeouts, rate limits
- **Mobile Responsive**: Works on all screen sizes

## Conventions

### Code Style

- Python code follows standard conventions with type hints
- Debug printing controlled by `print_all` flag (default: `False`)
- Jinja2 templates with Bootstrap 5 for styling
- CSS organized by component with clear section headers

### Database

- **SQLite** (`search_logs.db`): Request/response logging
- **PostgreSQL/pgvector**: Vector embeddings (production)
- Schema initialized on app startup via `init_db()`
- Flask `g` object used for request-scoped connections

### Security

- Admin routes protected by HTTP Basic Auth
- SQL injection **fixed** - all queries use parameterized statements
- Secret key from environment variable
- Rate limiting prevents abuse

## Common Tasks

### Switching Vector Backends

```bash
# Switch to pgvector
export VECTOR_DB_BACKEND=pgvector

# Switch to Pinecone
export VECTOR_DB_BACKEND=pinecone
```

### Adding New Midrash Sources

1. Place text files in `Sefaria-Export/txt/Midrash/[source]/English/`
2. Run `python generate-midrash-embeddings.py` (for Pinecone)
3. Or run `python migrate_to_pgvector.py` (for pgvector)

### Viewing Logs

Access admin log views at:
- `/logs` - Request logs
- `/response_logs` - Response logs with AI answers

Requires HTTP Basic Auth with username `admin` and `LOG_PASSWORD` env var.

### Modifying the AI Personality

Edit the `system_prompt` variable in `app.py` within the `search_function()` to change how the AI Rabbi responds.

## Dependencies

Key packages from `requirements.txt`:

**Core:**
- `flask==2.3.2` - Web framework
- `gunicorn==20.1.0` - Production WSGI server

**AI:**
- `openai==0.27.2` - OpenAI API client
- `tiktoken==0.3.3` - Token counting

**Vector Databases:**
- `psycopg2-binary==2.9.9` - PostgreSQL/pgvector driver
- `pinecone-client==2.2.1` - Pinecone client (legacy)

**Data:**
- `pandas==2.0.1` - Data manipulation

**Note**: Uses legacy OpenAI v0.x API. The pgvector integration uses modern PostgreSQL best practices.

## Related Documentation

- **HUMAN_TASKS.md** - Manual deployment and configuration steps
- **PINECONE_ALTERNATIVES.md** - Vector database comparison and migration guide
- **PRODUCT_ROADMAP.md** - Product strategy, user archetypes, and feature roadmap
