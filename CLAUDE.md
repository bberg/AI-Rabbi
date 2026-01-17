# AI-Rabbi

An AI-powered chatbot that answers life's questions using Jewish Midrash source texts. The application uses semantic search to find relevant passages from Midrash literature and GPT-4 to provide thoughtful, rabbi-style analysis and responses.

## Architecture Overview

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
- **Pinecone Vector Database**: Stores embeddings of Midrash text segments for semantic search
- **OpenAI GPT-4**: Generates rabbi-style responses based on retrieved source texts
- **SQLite**: Logs user requests and responses for analytics

## Project Structure

```
AI-Rabbi/
├── app.py                              # Main Flask application
├── generate-midrash-embeddings.py      # Script to generate and upload embeddings
├── minimize_output_file.py             # Utility to reduce pickle file size (deprecated)
├── schema.sql                          # SQLite database schema
├── requirements.txt                    # Python dependencies
├── Procfile                            # Heroku deployment configuration
├── output-5sentence_without_embeddings.pkl  # Preprocessed Midrash data (5 sentences per segment)
├── output_without_embeddings.pkl       # Alternative Midrash data
├── MJ prompt.txt                       # Midjourney prompt for logo generation
├── static/
│   ├── css/
│   │   └── main.css                    # Application styles
│   └── img/
│       └── airabbi.png                 # Logo image
└── templates/
    ├── index.html                      # Main search interface
    ├── request_logs.html               # Admin view for request logs
    └── response_logs.html              # Admin view for response logs
```

## Key Files

### `app.py` - Main Application

The core Flask application with the following key functions:

| Function | Line | Description |
|----------|------|-------------|
| `get_embedding()` | 124 | Creates text embeddings using OpenAI's ada-002 model |
| `get_relevant_sources()` | 195 | Queries Pinecone for top 3 relevant Midrash passages |
| `search_function()` | 128 | Streams GPT-4 responses with Midrash context |
| `is_request_allowed()` | 108 | Rate limiting check (5 requests/day per user) |
| `scheduled_task()` | 80 | Keepalive task to prevent Pinecone database dormancy |

**Routes:**
- `GET /` - Main search interface
- `POST /search` - Submit a question and receive streaming response
- `GET /logs` - View request logs (admin, HTTP Basic Auth)
- `GET /response_logs` - View response logs (admin, HTTP Basic Auth)
- `POST /login` - User authentication
- `GET /logout` - User logout

### `generate-midrash-embeddings.py` - Data Pipeline

Processes Midrash text files from Sefaria export and uploads embeddings to Pinecone:

1. Finds all `.txt` files in `Sefaria-Export/txt/Midrash/*/English/`
2. Splits texts into segments (default: 5 sentences each)
3. Generates embeddings using OpenAI ada-002
4. Uploads embeddings to Pinecone in batches of 100
5. Saves progress to pickle files to resume interrupted uploads

### `schema.sql` - Database Schema

Two tables for logging:
- `request_logs`: Records incoming search queries
- `response_logs`: Records queries with their AI-generated responses

Both tables track: `id`, `ip_address`, `real_ip_address`, `query`, `collected_messages`, `user_id`, `timestamp`

## Environment Variables

Required environment variables for deployment:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 and embeddings |
| `PINECONE_API_KEY` | Pinecone API key for vector storage |
| `PINECONE_ENV` | Pinecone environment (e.g., `us-west1-gcp`) |
| `LOG_PASSWORD` | Password for admin log access |

## Development Workflow

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export PINECONE_API_KEY="your-key"
export PINECONE_ENV="your-env"
export LOG_PASSWORD="your-password"

# Run the application
python app.py
# Server starts at http://localhost:3000
```

### Deployment (Heroku)

The application is configured for Heroku deployment via `Procfile`:
```
web: gunicorn app:app -t 300 --bind 0.0.0.0:$PORT --log-file -
```

The 300-second timeout accommodates long-running GPT-4 streaming responses.

## Key Technical Details

### Embeddings & Search

- **Model**: `text-embedding-ada-002` (1536 dimensions)
- **Index**: Pinecone index named `midrash` with cosine similarity
- **Retrieval**: Top 3 most relevant passages per query
- **Token counting**: Uses `tiktoken` with `cl100k_base` encoding

### Rate Limiting

- Users are limited to 5 requests per 24-hour period
- Users identified by cookie-based UUID (`user_id`)
- Check implemented in `is_request_allowed()` function

### GPT-4 System Prompt

The AI responds as a "Rabbi chatbot analyzing Midrash" with instructions to:
- Analyze how each source text answers the question
- Provide step-by-step reasoning
- Cite sources explicitly
- Identify conflicting advice between texts
- Provide a modern rabbinic sermon story
- Offer theological/philosophical explanation
- Say "I don't know" if sources don't answer the question

### Keepalive Scheduler

A background scheduler (`APScheduler`) runs every 12 hours to query Pinecone and prevent database dormancy on free tier.

## Conventions

### Code Style

- Python code follows standard conventions
- No type hints currently used
- Debug printing controlled by `print_all` flag (default: `False`)
- Jinja2 templates with Bootstrap 5 for styling

### Database

- SQLite database file: `search_logs.db`
- Schema initialized on app startup via `init_db()`
- Flask `g` object used for request-scoped database connections

### Security Notes

- Admin routes protected by HTTP Basic Auth (`flask_httpauth`)
- User sessions managed via `flask_login`
- SQL injection vulnerability exists in `/search` route (line 236) - uses string concatenation instead of parameterized queries
- Secret key is hardcoded (`'super secret key'`) - should use environment variable in production

## Common Tasks

### Adding New Midrash Sources

1. Place text files in `Sefaria-Export/txt/Midrash/[source]/English/`
2. Run `python generate-midrash-embeddings.py`
3. Monitor progress - saves checkpoints to pickle files
4. Restart application to pick up new data

### Viewing Logs

Access admin log views at:
- `/logs` - Request logs
- `/response_logs` - Response logs with AI answers

Requires HTTP Basic Auth with username `admin` and `LOG_PASSWORD` env var.

### Modifying the AI Personality

Edit the system prompt in `app.py` line 151 within the `search_function()` to change how the AI Rabbi responds.

## Dependencies

Key packages from `requirements.txt`:
- `flask==2.3.2` - Web framework
- `openai==0.27.2` - OpenAI API client (legacy v0.x API)
- `pinecone-client==2.2.1` - Vector database client (legacy v2.x API)
- `pandas==2.0.1` - Data manipulation
- `tiktoken==0.3.3` - Token counting
- `gunicorn==20.1.0` - Production WSGI server
- `APScheduler==3.10.4` - Background job scheduler

**Note**: Uses legacy OpenAI and Pinecone client versions. Modern projects should use `openai>=1.0.0` and `pinecone-client>=3.0.0` with updated API syntax.

## Known Issues & TODOs

Based on code comments:
- SQL injection vulnerability in search route (should use parameterized queries)
- Hardcoded secret key should be environment variable
- `minimize_output_file.py` marked as deprecated
- Consider second-pass retrieval within top articles for better relevance
- Exception handling for development mode via cookies (incomplete)
