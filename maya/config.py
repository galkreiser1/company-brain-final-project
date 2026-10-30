"""Settings for the Maya agent, read once from the environment.

Every other module imports its settings from here, so there is one place to look.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


# Read settings (AWS keys, model ID, ...) from the nearest .env file.
load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parent.parent

# The NovaOps dataset (database, documents, workflows). A relative path in .env is
# read from the repo root, so scripts work from any folder. (Joining an absolute path
# onto REPO_ROOT just gives back the absolute path.)
DATA_DIR = REPO_ROOT / os.environ.get("NOVAOPS_DATA_DIR", "data/novaops")

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:9877/mcp")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
# The respond step's model. Nova 2 Lite wrote broken JSON for the nested MayaReply
# (0 of 8 valid, stopReason "malformed_tool_use"); Nova Pro wrote it 4 of 4 times.
RESPOND_MODEL_ID = os.environ.get("BEDROCK_RESPOND_MODEL_ID", "us.amazon.nova-pro-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Document search: the OpenSearch Serverless collection (found by name) and the index
# inside it, plus the model that turns text into vectors.
OPENSEARCH_COLLECTION = os.environ.get("OPENSEARCH_COLLECTION")
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "novaops-maya")
EMBEDDING_MODEL_ID = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
