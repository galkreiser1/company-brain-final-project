"""Retrieval over the NovaOps documents, limited to what the caller may see.

Based on Lesson 7's homework (retrieval.py and reranker.py).

Access rule: a document about one employee (subject_employee_id is set) is visible
only to HR and to that employee. Every other document is visible to everyone.
"""

from opensearchpy import OpenSearch

from maya.config import MODEL_ID, OPENSEARCH_INDEX
from maya.opensearch import bedrock, embed_text
from maya.schemas import CallerContext


HR_GROUP = "UG_HR"

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

def knn_body(vector: list[float], caller: CallerContext, n: int) -> dict:
    """The k-NN search request, with the access filter inside the k-NN block.

    Gets: the query's vector, the caller, and how many hits to return.
    Returns: the request body.

    The filter is inside the k-NN block, so OpenSearch only considers chunks the
    caller may see, and still returns a full `n` of them. Filtering the results
    afterwards could leave fewer than `n`, or none.
    """

    knn = {"vector": vector, "k": n}
    clauses = access_filter(caller)
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}
    return {"size": n, "query": {"knn": {"vector": knn}}, "_source": SOURCE_FIELDS}


def knn_candidates(client: OpenSearch, query: str, caller: CallerContext,
                   n: int = CANDIDATES) -> list[dict]:
    """The `n` chunks closest in meaning to the query, among those the caller may see.

    Gets: the OpenSearch client, the query text, the caller, and `n`.
    Returns: the chunks' fields (doc_id, text, ...), closest first.
    """

    body = knn_body(embed_text(query), caller, n)
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
