"""OpenSearch and embeddings: connect, turn text into vectors, and create the index.

Based on Lesson 7's homework (client.py). One client of each kind per process, so
these are plain functions, not a class.
"""

import json

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from maya.config import (
    EMBEDDING_MODEL_ID, OPENSEARCH_COLLECTION, OPENSEARCH_INDEX, REGION,
)


EMBED_DIM = 1024  # Titan V2 vector width

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


# EMBEDDINGS ------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Turn text into a 1024-number Titan vector, scaled to length 1.

    Documents and questions must both go through this same function, or their
    vectors can't be compared.
    """

    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


# THE CLIENT ------------------------------------------------------------------

def resolve_endpoint() -> str:
    """Look up the collection's URL by its name.

    Recreating a collection gives it a new URL. Looking it up by name means .env
    never needs editing after a recreate.
    """

    if not OPENSEARCH_COLLECTION:
        raise SystemExit("Set OPENSEARCH_COLLECTION in .env.")

    serverless = boto3.client("opensearchserverless", region_name=REGION)
    found = serverless.batch_get_collection(names=[OPENSEARCH_COLLECTION])
    details = found.get("collectionDetails", [])
    if not details:
        raise SystemExit(f"Collection '{OPENSEARCH_COLLECTION}' not found.")

    endpoint = details[0].get("collectionEndpoint")
    if not endpoint:
        status = details[0].get("status")
        raise SystemExit(f"Collection '{OPENSEARCH_COLLECTION}' is {status}; wait for ACTIVE.")
    return endpoint


def opensearch_client() -> OpenSearch:
    """A client for the Serverless collection, signed with SigV4.

    There's no username or password: each request is signed with your AWS
    credentials and checked by IAM. "aoss" is the service name for Serverless.
    """

    host = resolve_endpoint().removeprefix("https://").rstrip("/")
    credentials = boto3.Session(region_name=REGION).get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        # An idle collection scales down, so the first request can take longer than
        # the 10-second default while it warms up.
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )


# THE INDEX -------------------------------------------------------------------
#
# The mapping says what type each field has. "keyword" fields match exactly, which
# makes them cheap to filter on (e.g. audience = "all"). "text" fields are split into
# words for keyword search. "date" fields allow ranges (e.g. updated after June).

INDEX_BODY = {
    "settings": {"index": {"knn": True}},  # knn: allow vector search on this index
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                # Titan vectors have length 1, so inner product ranks the same as
                # cosine similarity, and is cheaper to compute.
                "method": {"name": "hnsw", "space_type": "innerproduct"},
            },
            "text": {"type": "text"},
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "collection": {"type": "keyword"},
            "section": {"type": "keyword"},
            "subject_employee_id": {"type": "keyword"},
            "audience": {"type": "keyword"},
            "updated": {"type": "date", "format": "yyyy-MM-dd"},
        }
    },
}


def create_index(client: OpenSearch) -> bool:
    """Create the Maya index if it doesn't exist yet. Returns True if it was created.

    An existing index is kept as it is: re-running the ingest overwrites its chunks
    instead of starting from an empty index.
    """

    if client.indices.exists(index=OPENSEARCH_INDEX):
        return False
    client.indices.create(index=OPENSEARCH_INDEX, body=INDEX_BODY)
    return True
