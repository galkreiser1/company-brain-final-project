"""Load the NovaOps documents into Maya's OpenSearch index.

    .venv/bin/python -m scripts.ingest_documents

Run order:
  1. read  - split the documents into chunk records (maya/ingest.py). No AWS.
  2. index - create the index, if it doesn't exist yet.
  3. embed - add a vector to each record (one Bedrock call per chunk).
  4. load  - upload all records in one bulk request.
  5. check - wait until the chunks are searchable, then count them per collection.

Safe to re-run: each chunk is stored under its chunk_id, so loading it again replaces
it instead of adding a copy.
"""

import time
from collections import Counter

from opensearchpy import helpers

from maya.config import DATA_DIR, OPENSEARCH_INDEX
from maya.ingest import read_records
from maya.opensearch import create_index, embed_text, opensearch_client


# 1. READ ---------------------------------------------------------------------

def read_documents() -> list[dict]:
    """The chunk records, after printing any documents that had to be skipped."""

    records, skipped = read_records(DATA_DIR)
    for reason in skipped:
        print(f"  SKIPPED {reason}")
    print(f"Read {len(records)} chunks ({len(skipped)} documents skipped).")
    return records


# 3. EMBED --------------------------------------------------------------------

def embed_records(records: list[dict]) -> None:
    """Add a vector to each record, in place."""

    for number, record in enumerate(records, start=1):
        record["vector"] = embed_text(record["text"])
        if number % 50 == 0 or number == len(records):
            print(f"  embedded {number}/{len(records)}")


# 4. LOAD ---------------------------------------------------------------------

def bulk_actions(records: list[dict]) -> list[dict]:
    """One bulk "index" action per record.

    `_id` is the chunk_id. Indexing a document with an _id that already exists
    replaces it, which is why re-running creates no duplicates.
    """

    return [
        {"_index": OPENSEARCH_INDEX, "_id": record["chunk_id"], "_source": record}
        for record in records
    ]


# 5. CHECK --------------------------------------------------------------------

def wait_until_searchable(client, expected: int, timeout: int = 60) -> bool:
    """Wait until a search sees at least `expected` chunks. False if it times out.

    Serverless has no manual "refresh": new documents show up in searches a few
    seconds after loading, so we ask for the count every 2 seconds.
    """

    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.count(index=OPENSEARCH_INDEX)["count"] >= expected:
            return True
        time.sleep(2)
    return False


def indexed_per_collection(client) -> dict[str, int]:
    """How many chunks the index holds for each collection (a terms aggregation)."""

    response = client.search(index=OPENSEARCH_INDEX, body={
        "size": 0,  # no documents, only the counts
        "aggs": {"per_collection": {"terms": {"field": "collection"}}},
    })
    buckets = response["aggregations"]["per_collection"]["buckets"]
    return {bucket["key"]: bucket["doc_count"] for bucket in buckets}


def print_counts(records: list[dict], indexed: dict[str, int]) -> None:
    """Chunks read vs chunks in the index, per collection. They should match."""

    read = Counter(record["collection"] for record in records)
    print(f"  {'collection':<16}{'read':>6}{'indexed':>9}")
    for collection, count in read.items():
        print(f"  {collection:<16}{count:>6}{indexed.get(collection, 0):>9}")
    print(f"  {'total':<16}{len(records):>6}{sum(indexed.values()):>9}")


def main() -> None:
    records = read_documents()

    client = opensearch_client()
    if create_index(client):
        print(f"Created index '{OPENSEARCH_INDEX}'.")
    else:
        print(f"Index '{OPENSEARCH_INDEX}' exists; its chunks will be replaced.")

    print("Embedding...")
    embed_records(records)

    loaded, _ = helpers.bulk(client, bulk_actions(records))
    print(f"Loaded {loaded} chunks into '{OPENSEARCH_INDEX}'.")

    if not wait_until_searchable(client, len(records)):
        print("  (not all chunks are searchable yet; the counts below may be low)")
    print_counts(records, indexed_per_collection(client))


if __name__ == "__main__":
    main()
