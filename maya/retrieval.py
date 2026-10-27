"""Retrieval over the NovaOps documents, limited to what the caller may see.

Based on Lesson 7's homework (retrieval.py and reranker.py).

Access rule: a document about one employee (subject_employee_id is set) is visible
only to HR and to that employee. Every other document is visible to everyone.
"""

from typing import Protocol

from opensearchpy import OpenSearch

from maya.config import MODEL_ID, OPENSEARCH_INDEX
from maya.opensearch import bedrock, embed_text, opensearch_client
from maya.schemas import CallerContext, Evidence


HR_GROUP = "UG_HR"

# How many chunks a search returns, after reranking.
TOP_K = 4
# The vector search fetches this many candidates; the reranker then keeps the best few.
CANDIDATES = 20
# The fields each hit returns. The vector itself isn't needed after the search.
SOURCE_FIELDS = ["doc_id", "chunk_id", "collection", "section", "subject_employee_id",
                 "updated", "text"]


# 1. WHO MAY SEE WHAT ---------------------------------------------------------

def authorize(caller: CallerContext, subject_employee_id: str | None) -> bool:
    """May this caller see a document about this employee?

    Gets: the caller, and the document's subject_employee_id (None if it's about
    nobody in particular).
    Returns: True if the document has no subject, the caller is in HR, or the
    caller is the subject.
    """

    if subject_employee_id is None:
        return True
    if caller.user_group == HR_GROUP:
        return True
    return caller.caller_id == subject_employee_id


def access_filter(caller: CallerContext) -> list[dict]:
    """The same rule as authorize(), written as OpenSearch filter clauses.

    Gets: the caller.
    Returns: [] for HR (no limit). For anyone else, one clause: audience is "all"
    OR the chunk is about the caller.
    """

    if caller.user_group == HR_GROUP:
        return []
    return [{
        "bool": {
            "should": [
                {"term": {"audience": "all"}},
                {"term": {"subject_employee_id": caller.caller_id}},
            ],
            "minimum_should_match": 1,  # at least one of the two must hold
        }
    }]


# 2. FILTERED VECTOR SEARCH ---------------------------------------------------

def knn_body(vector: list[float], caller: CallerContext, n: int,
             collections: list[str] | None = None) -> dict:
    """The k-NN search request, with the filters inside the k-NN block.

    Gets: the query's vector, the caller, how many hits to return, and optionally
    the only collections to search.
    Returns: the request body.

    The filter is inside the k-NN block, so OpenSearch only considers chunks the
    caller may see, and still returns a full `n` of them. Filtering the results
    afterwards could leave fewer than `n`, or none.
    """

    knn = {"vector": vector, "k": n}
    clauses = access_filter(caller)
    if collections:
        clauses = clauses + [{"terms": {"collection": collections}}]  # any of these
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}
    return {"size": n, "query": {"knn": {"vector": knn}}, "_source": SOURCE_FIELDS}


def knn_candidates(client: OpenSearch, query: str, caller: CallerContext,
                   n: int = CANDIDATES, collections: list[str] | None = None) -> list[dict]:
    """The `n` chunks closest in meaning to the query, among those the caller may see.

    Gets: the OpenSearch client, the query text, the caller, `n`, and optionally the
    only collections to search.
    Returns: the chunks' fields (doc_id, text, ...), closest first.
    """

    body = knn_body(embed_text(query), caller, n, collections)
    hits = client.search(index=OPENSEARCH_INDEX, body=body)["hits"]["hits"]
    return [hit["_source"] for hit in hits]


# 3. RERANK -------------------------------------------------------------------
#
# Vector search is cheap but rough: each chunk's vector was made before any question
# existed. The reranker reads the question and every candidate together, and scores
# how well each one answers it. Too slow for the whole index; fine for 20 candidates.
#
# The model must answer by calling the "rank" tool ("toolChoice" forces the call),
# so its reply is always a list of scores we can read. The tool description and the
# prompt are kept word for word from Lesson 7, since the model reads them.

RERANK_TOOL = {
    "toolSpec": {
        "name": "rank",
        "description": "Score every candidate passage for relevance to the query.",
        "inputSchema": {"json": {
            "type": "object", "additionalProperties": False, "required": ["scores"],
            "properties": {"scores": {
                "type": "array",
                "description": "One entry per candidate, referenced by the [index] shown.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["index", "score"],
                    "properties": {
                        "index": {"type": "integer", "description": "The candidate's [index]."},
                        "score": {"type": "number", "description": "Relevance from 0.0 to 1.0."},
                    },
                },
            }},
        }},
    }
}


def rerank_prompt(query: str, candidates: list[dict]) -> str:
    """The reranker's prompt: the query, then every candidate labelled [0], [1], ...

    Gets: the query and the candidates.
    Returns: the prompt text. Each candidate is sent whole, so it's scored on the
    same text the answer will later use.
    """

    listing = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(candidates))
    return (
        f"Query: {query}\n\nScore each candidate passage from 0.0 to 1.0 for how "
        f"well it helps answer the query. Use the exact [index] shown for each."
        f"\n\n{listing}"
    )


def read_scores(response: dict, count: int) -> dict[int, float]:
    """The scores from the model's "rank" tool call.

    Gets: the Bedrock response, and how many candidates there were.
    Returns: {candidate index: score}. Indexes outside 0..count-1 are ignored and
    scores are clamped to 0.0-1.0, since the model could echo back anything.
    """

    scores = {}
    for block in response["output"]["message"]["content"]:
        if "toolUse" not in block:
            continue
        for entry in block["toolUse"]["input"].get("scores", []):
            index = entry.get("index")
            if isinstance(index, int) and 0 <= index < count:
                scores[index] = max(0.0, min(1.0, float(entry.get("score", 0.0))))
    return scores


def rerank(query: str, candidates: list[dict]) -> list[tuple[dict, float]]:
    """Score all candidates in one Nova call, and sort them best first.

    Gets: the query and the candidates.
    Returns: (candidate, score) pairs, best first. A candidate the model skipped
    scores 0.0, so it sinks to the bottom instead of disappearing.
    """

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": rerank_prompt(query, candidates)}]}],
        toolConfig={"tools": [RERANK_TOOL], "toolChoice": {"tool": {"name": "rank"}}},
    )
    scores = read_scores(response, len(candidates))
    order = sorted(range(len(candidates)), key=lambda i: scores.get(i, 0.0), reverse=True)
    return [(candidates[i], scores.get(i, 0.0)) for i in order]


# 4. THE RETRIEVER ------------------------------------------------------------

class EvidenceRetriever(Protocol):
    """Anything that can search the documents for a caller.

    A Protocol only describes the method a retriever must have. The agent depends
    on this, not on OpenSearch, so a test can pass in a fake retriever instead.
    """

    def search(self, query: str, caller: CallerContext, k: int = TOP_K,
               collections: list[str] | None = None) -> list[Evidence]:
        ...


def to_evidence(chunk: dict, score: float) -> Evidence:
    """One reranked chunk as Evidence.

    Gets: the chunk's fields from OpenSearch, and its reranker score.
    Returns: the Evidence (subject_employee_id is left out: the agent doesn't need it).
    """

    return Evidence(
        doc_id=chunk["doc_id"],
        chunk_id=chunk["chunk_id"],
        collection=chunk["collection"],
        section=chunk["section"],
        updated=chunk.get("updated"),
        text=chunk["text"],
        score=score,
    )


class OpenSearchRetriever:
    """The real retriever: filtered k-NN search on OpenSearch, then the Nova reranker."""

    def __init__(self, client: OpenSearch | None = None):
        self.client = client or opensearch_client()

    def search(self, query: str, caller: CallerContext, k: int = TOP_K,
               collections: list[str] | None = None) -> list[Evidence]:
        """The `k` chunks that best answer the query, among those the caller may see.

        Gets: the query, the caller (set by code, never by the model), `k`, and
        optionally the only collections to search (None: all of them).
        Returns: up to `k` Evidence, best first ([] if the caller may see nothing
        relevant).
        """

        candidates = knn_candidates(self.client, query, caller, collections=collections)
        # A second check, in Python, in case the filter is ever wrong: a chunk the
        # caller may not see never reaches the reranker or the agent.
        allowed = [c for c in candidates if authorize(caller, c["subject_employee_id"])]
        if not allowed:
            return []

        ranked = rerank(query, allowed)
        return [to_evidence(chunk, score) for chunk, score in ranked[:k]]


# 5. SEARCHING ABOUT ONE EMPLOYEE ---------------------------------------------

def search_about(retriever: EvidenceRetriever, query: str, caller: CallerContext,
                 subject_employee_id: str | None, k: int = TOP_K) -> list[Evidence]:
    """Search, but only if the caller may see documents about this employee.

    Gets: a retriever, the query, the caller, the employee the question is about
    (None if nobody in particular), and `k`.
    Returns: the retriever's evidence, or [] without searching at all when
    authorize() denies the caller. Not searching means nothing about that employee
    can leak, not even through the reranker's model call.
    """

    if not authorize(caller, subject_employee_id):
        return []
    return retriever.search(query, caller, k=k)
