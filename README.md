# Local Agentic RAG Setup

Add a `data` directory and drop in any PDFs that you want available for RAG.

Run `python populate_database.py` to seed the chroma database with embeddings.

You can then run `python query_data` to start the chat

### Cleaning the database

Anytime you run the database population command it will update the database with new PDFs. If you need to clean it and start fresh because embeddings seem off, append the `--reset` flag.