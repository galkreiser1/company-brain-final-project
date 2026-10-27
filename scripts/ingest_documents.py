"""Load the NovaOps documents into Maya's OpenSearch index.

    .venv/bin/python -m scripts.ingest_documents

Run order:
  1. read  - split the documents into chunk records (maya/ingest.py). No AWS.
  2. index - create the index, if it doesn't exist yet.
  3. embed - add a vector to each record (one Bedrock call per chunk).
  4. load  - upload all records in one bulk request.
  5. clean - delete chunks this run didn't load (removed sections or files).
  6. check - wait until the chunks are searchable, then count them per collection.

Safe to re-run: each chunk is stored under its chunk_id, so loading it again replaces
it instead of adding a copy, and step 5 removes chunks that no longer exist.
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


# 5. CLEAN --------------------------------------------------------------------

def indexed_chunk_ids(client) -> set[str]:
    """The _id of every chunk in the index.

    One search returns at most 10,000 hits, plenty for our ~270 chunks.
    """

    response = client.search(index=OPENSEARCH_INDEX, body={
        "size": 10_000,
        "_source": False,  # only the ids, not the text and vectors
        "query": {"match_all": {}},
    })
    return {hit["_id"] for hit in response["hits"]["hits"]}


def delete_stale_chunks(client, records: list[dict]) -> list[str]:
    """Delete every chunk in the index that this run didn't load. Returns their ids.

    A chunk goes stale when its section or file was removed, or its document was
    skipped this run. Only call this after the load succeeded: after a failed load,
    good chunks would look stale too.
    """

    loaded_ids = {record["chunk_id"] for record in records}
    stale_ids = sorted(indexed_chunk_ids(client) - loaded_ids)
    actions = [
        {"_op_type": "delete", "_index": OPENSEARCH_INDEX, "_id": chunk_id}
        for chunk_id in stale_ids
    ]
    helpers.bulk(client, actions)
    return stale_ids


# 6. CHECK --------------------------------------------------------------------

def wait_until_searchable(client, expected: int, timeout: int = 60) -> bool:
    """Wait until a search sees exactly `expected` chunks. False if it times out.

    Serverless has no manual "refresh": new documents show up in searches a few
    seconds after loading, so we ask for the count every 2 seconds.
    """

    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.count(index=OPENSEARCH_INDEX)["count"] == expected:
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

    # bulk() raises an error if any chunk fails to load, so a failed load stops here,
    # before the cleanup below.
    loaded, _ = helpers.bulk(client, bulk_actions(records))
    print(f"Loaded {loaded} chunks into '{OPENSEARCH_INDEX}'.")

    stale_ids = delete_stale_chunks(client, records)
    print(f"Deleted {len(stale_ids)} stale chunks.")
    for chunk_id in stale_ids:
        print(f"  deleted {chunk_id}")

    if not wait_until_searchable(client, len(records)):
        print("  (the index isn't up to date yet; the counts below may be off)")
    print_counts(records, indexed_per_collection(client))


if __name__ == "__main__":
    main()
