#!/usr/bin/env python3
"""
Migration Script: Pinecone to pgvector

This script migrates the Midrash embeddings from the existing pickle files
to a PostgreSQL database with pgvector extension.

Prerequisites:
    1. PostgreSQL with pgvector extension deployed (e.g., on Railway)
    2. DATABASE_URL environment variable set
    3. OPENAI_API_KEY environment variable set
    4. Required packages: psycopg2-binary, openai, pandas, tiktoken

Usage:
    export DATABASE_URL="postgresql://..."
    export OPENAI_API_KEY="sk-..."
    python migrate_to_pgvector.py

Options:
    --dry-run       Show what would be migrated without making changes
    --batch-size    Number of vectors to process at once (default: 50)
    --resume        Resume from last checkpoint
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Check required environment variables
def check_environment():
    """Verify required environment variables are set."""
    required = ['DATABASE_URL', 'OPENAI_API_KEY']
    missing = [var for var in required if not os.environ.get(var)]

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nSet them with:")
        for var in missing:
            print(f"  export {var}=\"your-value\"")
        sys.exit(1)


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Get embedding from OpenAI API.

    Args:
        text: Text to embed
        model: OpenAI embedding model

    Returns:
        List of floats (1536 dimensions)
    """
    import openai

    text = text.replace("\n", " ")

    # Handle long texts by truncating
    max_chars = 8000  # Safe limit for ada-002
    if len(text) > max_chars:
        text = text[:max_chars]

    try:
        result = openai.Embedding.create(input=[text], model=model)
        return result['data'][0]['embedding']
    except Exception as e:
        print(f"  Warning: Embedding failed, retrying with truncated text: {e}")
        # Retry with more aggressive truncation
        text = text[:6000]
        result = openai.Embedding.create(input=[text], model=model)
        return result['data'][0]['embedding']


def load_checkpoint(checkpoint_file: str) -> dict:
    """Load migration checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'last_index': -1, 'migrated_count': 0}


def save_checkpoint(checkpoint_file: str, last_index: int, migrated_count: int):
    """Save migration progress checkpoint."""
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'last_index': last_index,
            'migrated_count': migrated_count,
            'timestamp': datetime.now().isoformat()
        }, f)


def migrate(dry_run: bool = False, batch_size: int = 50, resume: bool = False):
    """
    Main migration function.

    Args:
        dry_run: If True, don't actually migrate, just show what would happen
        batch_size: Number of records to process in each batch
        resume: If True, resume from last checkpoint
    """
    import pandas as pd
    import openai

    from vector_db import PgVectorDB

    # Set OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Load the source data
    pickle_file = 'output-5sentence_without_embeddings.pkl'
    if not os.path.exists(pickle_file):
        print(f"Error: Source file '{pickle_file}' not found")
        sys.exit(1)

    print(f"Loading source data from {pickle_file}...")
    df = pd.read_pickle(pickle_file)
    total_records = len(df)
    print(f"Found {total_records} records to migrate")

    # Checkpoint handling
    checkpoint_file = 'migration_checkpoint.json'
    checkpoint = load_checkpoint(checkpoint_file) if resume else {'last_index': -1, 'migrated_count': 0}
    start_index = checkpoint['last_index'] + 1

    if start_index > 0:
        print(f"Resuming from index {start_index} ({checkpoint['migrated_count']} already migrated)")

    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Would migrate {total_records - start_index} records")
        print(f"Batch size: {batch_size}")
        print("\nSample records:")
        for idx, row in df.head(3).iterrows():
            print(f"  [{idx}] {row['filename'][:50]}... (segment {row['segment_number']})")
        return

    # Initialize pgvector database
    print("\nConnecting to PostgreSQL...")
    db = PgVectorDB()

    # Check existing count
    existing_count = db.count()
    print(f"Existing vectors in database: {existing_count}")

    # Process in batches
    batch = []
    migrated_count = checkpoint['migrated_count']
    errors = []

    print(f"\nStarting migration from index {start_index}...")
    print(f"Processing in batches of {batch_size}")
    print("-" * 50)

    for idx, row in df.iloc[start_index:].iterrows():
        try:
            # Rate limiting for OpenAI API
            if len(batch) > 0 and len(batch) % 10 == 0:
                time.sleep(0.5)  # Avoid rate limits

            # Get embedding
            print(f"  [{idx}/{total_records}] Processing: {row['filename'][-40:]}...", end="", flush=True)
            embedding = get_embedding(row['text'])

            batch.append({
                'id': idx,
                'filename': row['filename'],
                'segment_number': int(row['segment_number']),
                'text': row['text'],
                'embedding': embedding
            })
            print(" OK")

            # Upsert batch
            if len(batch) >= batch_size:
                print(f"\n  Upserting batch of {len(batch)} vectors...")
                db.upsert(batch)
                migrated_count += len(batch)
                save_checkpoint(checkpoint_file, idx, migrated_count)
                print(f"  Progress: {migrated_count}/{total_records} ({100*migrated_count/total_records:.1f}%)\n")
                batch = []

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            if batch:
                db.upsert(batch)
                migrated_count += len(batch)
            save_checkpoint(checkpoint_file, idx, migrated_count)
            print(f"Checkpoint saved. Resume with: python migrate_to_pgvector.py --resume")
            db.close()
            sys.exit(0)

        except Exception as e:
            print(f" ERROR: {e}")
            errors.append({'index': idx, 'error': str(e)})
            continue

    # Final batch
    if batch:
        print(f"\nUpserting final batch of {len(batch)} vectors...")
        db.upsert(batch)
        migrated_count += len(batch)

    # Cleanup
    db.close()

    # Remove checkpoint file on success
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # Summary
    print("\n" + "=" * 50)
    print("MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Total records processed: {total_records}")
    print(f"Successfully migrated: {migrated_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nError details:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  Index {err['index']}: {err['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

        # Save errors to file
        with open('migration_errors.json', 'w') as f:
            json.dump(errors, f, indent=2)
        print("\nFull error log saved to: migration_errors.json")


def verify_migration():
    """Verify the migration was successful."""
    import pandas as pd
    from vector_db import PgVectorDB

    print("Verifying migration...")

    # Load source data
    df = pd.read_pickle('output-5sentence_without_embeddings.pkl')
    source_count = len(df)

    # Check database
    db = PgVectorDB()
    db_count = db.count()
    db.close()

    print(f"Source records: {source_count}")
    print(f"Database records: {db_count}")

    if db_count >= source_count:
        print("Migration verified successfully!")
        return True
    else:
        print(f"Warning: Missing {source_count - db_count} records")
        return False


def test_query():
    """Test a sample query against the migrated database."""
    import openai
    from vector_db import PgVectorDB

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    print("\nTesting query...")
    test_question = "What is the meaning of life?"

    # Get embedding for test question
    embedding = get_embedding(test_question)

    # Query database
    db = PgVectorDB()
    results = db.query(embedding, top_k=3)
    db.close()

    print(f"\nQuery: '{test_question}'")
    print(f"Results: {len(results)} matches\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r['score']:.4f}")
        print(f"   Source: {r['filename']}")
        print(f"   Text: {r['text'][:150]}...")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate Midrash embeddings to pgvector')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--verify', action='store_true', help='Verify migration')
    parser.add_argument('--test', action='store_true', help='Test query after migration')

    args = parser.parse_args()

    check_environment()

    if args.verify:
        verify_migration()
    elif args.test:
        test_query()
    else:
        migrate(dry_run=args.dry_run, batch_size=args.batch_size, resume=args.resume)

        if not args.dry_run:
            print("\nRunning verification...")
            if verify_migration():
                print("\nRunning test query...")
                test_query()
