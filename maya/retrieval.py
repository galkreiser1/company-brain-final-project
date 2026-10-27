"""Retrieval over the NovaOps documents, limited to what the caller may see.

Based on Lesson 7's homework (retrieval.py and reranker.py).

Access rule: a document about one employee (subject_employee_id is set) is visible
only to HR and to that employee. Every other document is visible to everyone.
"""

from opensearchpy import OpenSearch

from maya.config import OPENSEARCH_INDEX
from maya.opensearch import embed_text
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
