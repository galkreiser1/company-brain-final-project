"""OpenSearch and embeddings: connect to the collection, and turn text into vectors.

Based on Lesson 7's homework (client.py). One client of each kind per process, so
these are plain functions, not a class.
"""

import json

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from maya.config import EMBEDDING_MODEL_ID, OPENSEARCH_COLLECTION, REGION


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
