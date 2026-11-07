"""
The retrieval tools' engine (server side).

Lesson 9's rag.py, widened from one corpus to two. Retrieval over local FAISS
indexes — the same idea as Lessons 6–7, deliberately shrunk so this lab stays
standalone (no OpenSearch collection to stand up). The MCP server calls search();
the agent on the other side never sees any of this.

Two corpora, because this lesson's workflow needs two different kinds of document:
  - it_kb   : IT how-to / troubleshooting articles      -> search_knowledge_base
  - hr_docs : employment documents + internal memos     -> search_hr_documents

The second one is new here, and it is where the session's *early constraints* live
(the Q3 SaaS freeze memo, an offer letter's equipment entitlement). Retrieval is
one context source among several in this lesson — not the subject of it.

Each document is short, so we embed it whole as a single chunk. Titan returns
normalized vectors, so an inner-product index is cosine similarity.
"""

import functools
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "novaops" / "documents"
REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = 1024   # Titan Text Embeddings V2 default width
TOP_K = 3

# A corpus is just a name and the folders whose .md files go into its index.
CORPORA = {
    "it_kb": (DATA_DIR / "it_kb",),
    "hr_docs": (DATA_DIR / "employment", DATA_DIR / "internal_memos"),
}

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed_text(text: str) -> list[float]:
    """One Titan embedding call. normalize=True so inner product == cosine
    similarity. The index and the query must use the SAME model, or the vector
    geometry is meaningless."""
    resp = _bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
    )
    return json.loads(resp["body"].read())["embedding"]


@functools.lru_cache(maxsize=len(CORPORA))
def _index(corpus: str):
    """Embed every document in one corpus and build its FAISS index. lru_cache
    makes this run once per corpus — on first search, or at server startup."""
    paths = sorted(p for folder in CORPORA[corpus] for p in folder.glob("*.md"))
    docs = [(p.stem, p.read_text(encoding="utf-8")) for p in paths]
    vectors = np.array([embed_text(text) for _, text in docs], dtype="float32")
    # Flat inner-product index = exact cosine search. Fine for a few dozen docs;
    # Lesson 7 is where this becomes an approximate HNSW index in a managed store.
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index, docs


def warm_index() -> dict[str, int]:
    """Force every index to build now (called at server startup). Returns doc counts."""
    return {corpus: len(_index(corpus)[1]) for corpus in CORPORA}


def search(corpus: str, query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k documents in `corpus` most similar to `query`, best first."""
    index, docs = _index(corpus)
    qvec = np.array([embed_text(query)], dtype="float32")
    scores, ids = index.search(qvec, k)
    return [
        {"document": docs[i][0], "score": round(float(s), 3), "text": docs[i][1]}
        for s, i in zip(scores[0], ids[0])
    ]
