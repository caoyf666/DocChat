"""Embedding helpers."""

from typing import List

from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def generate_bge_m3_hybrid_vectors(model: BGEM3EmbeddingFunction, embedding_documents: List[str]):
    """Generate BGE-M3 dense and sparse vectors for text documents."""
    if not embedding_documents:
        raise ValueError("embedding_documents cannot be empty")

    if not all(isinstance(doc, str) and doc.strip() for doc in embedding_documents):
        raise ValueError("embedding_documents contains empty or non-string values")

    try:
        embedding_result = model.encode_documents(embedding_documents)
    except Exception as e:
        raise RuntimeError(f"BGE-M3 embedding failed: {e}") from e

    if "dense" not in embedding_result or "sparse" not in embedding_result:
        raise RuntimeError(f"embedding result missing fields: {list(embedding_result.keys())}")

    try:
        processed_sparse = []
        csr_array = embedding_result["sparse"]

        for index in range(len(embedding_documents)):
            start = csr_array.indptr[index]
            end = csr_array.indptr[index + 1]
            token_ids = csr_array.indices[start:end].tolist()
            weights = csr_array.data[start:end].tolist()
            processed_sparse.append(dict(zip(token_ids, weights)))
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"sparse vector parsing failed: {e}") from e

    return {
        "dense": [dense.tolist() for dense in embedding_result["dense"]],
        "sparse": processed_sparse,
    }
