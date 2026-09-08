"""
cli.py
Command-line inspection tool for the ChromaDB store used by the RAG
pipeline. Lets you list collections, count chunks, peek at sample
records, and delete/reset the database without writing throwaway
scripts every time.

Usage:
    python rag/cli.py list
    python rag/cli.py count
    python rag/cli.py inspect chunks_fixed_size
    python rag/cli.py inspect chunks_fixed_size --n 10
    python rag/cli.py delete chunks_fixed_size
    python rag/cli.py delete chunks_fixed_size --yes
    python rag/cli.py reset
    python rag/cli.py reset --yes
    python rag/cli.py --help
"""

# imports
import argparse
import shutil
from chromadb import PersistentClient

# database location
DB_PATH = "data/chroma_db"

# create client
client = PersistentClient(path=DB_PATH)


def list_collections():
    """Print collection names and how many items are in each."""
    collections = client.list_collections()

    if not collections:
        print(f"No collections found in '{DB_PATH}'.")
        return

    print(f"Collections in '{DB_PATH}':")
    for col in collections:
        # col from list_collections() doesn't always carry a live count,
        # so re-fetch it to get an accurate .count()
        live_col = client.get_collection(name=col.name)
        print(f"  - {col.name:30s} {live_col.count():5d} items")


def count_collections():
    """Print just the per-collection counts, no other detail."""
    collections = client.list_collections()

    if not collections:
        print(f"No collections found in '{DB_PATH}'.")
        return

    total = 0
    for col in collections:
        live_col = client.get_collection(name=col.name)
        n = live_col.count()
        total += n
        print(f"{col.name}: {n}")
    print(f"---\nTotal chunks across {len(collections)} collection(s): {total}")


def inspect_collection(name: str, n: int = 5):
    """
    Retrieve a small number of records from `name` and print:
    chunk ID, document ID, strategy, text (truncated).
    """
    try:
        collection = client.get_collection(name=name)
    except Exception:
        print(f"Collection '{name}' not found. Run 'list' to see available collections.")
        return

    total = collection.count()
    if total == 0:
        print(f"Collection '{name}' exists but is empty.")
        return

    n = min(n, total)
    results = collection.get(
        limit=n,
        include=["documents", "metadatas"],
    )

    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    print(f"Inspecting '{name}' ({total} total items, showing {n}):\n")
    for chunk_id, text, meta in zip(ids, docs, metas):
        doc_id = meta.get("doc_id", "?")
        strategy = meta.get("strategy", "?")
        preview = text[:120].replace("\n", " ")
        print(f"chunk_id : {chunk_id}")
        print(f"doc_id   : {doc_id}")
        print(f"strategy : {strategy}")
        print(f"text     : {preview}{'...' if len(text) > 120 else ''}")
        print("-" * 60)


def delete_collection(name: str, skip_confirm: bool = False):
    """Verify the collection exists, confirm, then delete it."""
    existing_names = {col.name for col in client.list_collections()}
    if name not in existing_names:
        print(f"Collection '{name}' not found. Available: {sorted(existing_names) or 'none'}")
        return

    if not skip_confirm:
        answer = input(f"Delete collection '{name}'? This cannot be undone. [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    client.delete_collection(name=name)
    print(f"Deleted collection '{name}'.")


def reset_database(skip_confirm: bool = False):
    """
    Deletes every collection AND removes the persistent DB directory on
    disk, so the next run starts from a completely clean state. This is
    more destructive than delete_collection, hence the stronger prompt.
    """
    collections = client.list_collections()
    names = [col.name for col in collections]

    if not skip_confirm:
        print(f"This will PERMANENTLY delete ALL {len(names)} collection(s) "
              f"in '{DB_PATH}': {names}")
        answer = input("Type 'RESET' to confirm: ").strip()
        if answer != "RESET":
            print("Aborted.")
            return

    for name in names:
        client.delete_collection(name=name)

    # remove the on-disk directory entirely so nothing stale is left behind
    shutil.rmtree(DB_PATH, ignore_errors=True)
    print(f"Database at '{DB_PATH}' has been reset.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and manage the ChromaDB store used by the RAG pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all collections with item counts.")
    subparsers.add_parser("count", help="Print item counts per collection and total.")

    inspect_parser = subparsers.add_parser("inspect", help="Show sample records from a collection.")
    inspect_parser.add_argument("name", help="Collection name, e.g. chunks_fixed_size")
    inspect_parser.add_argument("--n", type=int, default=5, help="Number of records to show (default: 5)")

    delete_parser = subparsers.add_parser("delete", help="Delete a single collection.")
    delete_parser.add_argument("name", help="Collection name to delete")
    delete_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    reset_parser = subparsers.add_parser("reset", help="Delete ALL collections and reset the database.")
    reset_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if args.command == "list":
        list_collections()
    elif args.command == "count":
        count_collections()
    elif args.command == "inspect":
        inspect_collection(args.name, n=args.n)
    elif args.command == "delete":
        delete_collection(args.name, skip_confirm=args.yes)
    elif args.command == "reset":
        reset_database(skip_confirm=args.yes)


if __name__ == "__main__":
    main()