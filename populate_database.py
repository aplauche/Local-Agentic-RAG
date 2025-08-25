import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import PyPDFLoader


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    # documents = load_documents()
    documents = load_documents_from_directory(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)


# def load_documents():
#     document_loader = PyPDFDirectoryLoader(DATA_PATH)
#     return document_loader.load()

# def load_blog_posts():
#     directory = "blog_posts"
#     documents = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".txt"):
#             loader = TextLoader(os.path.join(directory, filename), encoding='utf-8')
#             documents.extend(loader.load())
#     return documents

def load_documents_from_directory(directory, recursive=False):
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": lambda path: TextLoader(path, encoding='utf-8'),
        ".html": UnstructuredHTMLLoader,
    }

    documents = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()

            loader_cls = supported_extensions.get(ext)
            if loader_cls:
                try:
                    print(f"[+] Loading: {filepath}")
                    loader = loader_cls(filepath)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"[!] Failed to load {filepath}: {e}")
            else:
                print(f"[~] Skipping unsupported file: {filename}")

        if not recursive:
            break

    return documents



def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = generate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def generate_chunk_ids(chunks):

    # This will create IDs like "data/example.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
