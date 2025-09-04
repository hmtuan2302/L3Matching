from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

import faiss
import numpy as np
from thefuzz import process


def exact_match(titles: List[str], input_title: str) -> List[str]:
    """
    Perform exact case-insensitive match of input_title in titles list.

    Args:
        titles (list): List of titles (strings)
        input_title (str): Title to search for

    Returns:
        list: Titles that exactly match (case-insensitive)
    """
    if not titles or not input_title:
        return []
    input_lower = input_title.strip().lower()
    return [title for title in titles if title.strip().lower() == input_lower]


def fuzzy_search(
    input_title: str,
    title_list: List[str],
    limit: int = 5,
    threshold: int = 70,
) -> List[Tuple[str, int]]:
    """
    Perform fuzzy matching on a list of titles.

    Args:
        input_title (str): The title to search for.
        title_list (list): A list of titles to compare against.
        limit (int): Maximum number of results to return.
        threshold (int): Minimum similarity score to consider a match.

    Returns:
        list: List of tuples (matched_title, score) sorted by score descending.
    """
    if not input_title or not title_list:
        return []
    matches = process.extract(input_title, title_list, limit=limit)
    return [match for match in matches if match[1] >= threshold]


def build_hnsw_index(
    embeddings: np.ndarray,
    M: int = 32,
    efConstruction: int = 200,
    efSearch: int = 64,
) -> faiss.IndexHNSWFlat:
    """
    Build an HNSW index for given embeddings.

    Args:
        embeddings (np.ndarray): Array of shape (N, d).
        M (int): Number of neighbors in HNSW graph.
        efConstruction (int): Construction accuracy.
        efSearch (int): Search accuracy.

    Returns:
        FAISS index object.
    """
    if embeddings.size == 0:
        raise ValueError('Embeddings array is empty.')
    N, d = embeddings.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def find_similar_embeddings(
    input_embedding: np.ndarray,
    historical_embeddings: np.ndarray,
    limit: int = 10,
    threshold: float = 0.7,
    index: Optional[faiss.IndexHNSWFlat] = None,
    include_self: bool = False,
) -> List[dict]:
    """
    Find similar embeddings for the input embedding from historical embeddings
    using FAISS HNSW.

    Args:
        input_embedding (array-like): Query embedding, shape (d,).
        historical_embeddings (np.ndarray): Historical embeddings,
            shape (N, d).
        limit (int): Max number of results.
        threshold (float): Minimum similarity score.
        index (faiss.IndexHNSWFlat): Optional pre-built FAISS index.
        include_self (bool): Whether to include the input embedding itself
        in results.

    Returns:
        List[dict]: [{'index': int, 'similarity': float}]
    """
    if input_embedding is None or len(input_embedding) == 0:
        raise ValueError('Input embedding is empty.')
    if threshold < 0 or threshold > 1:
        raise ValueError('Threshold must be between 0 and 1.')

    query_vec = np.array(input_embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)

    if index is None:
        index = build_hnsw_index(historical_embeddings)

    # +1 to account for potential self
    distances, indices = index.search(query_vec, limit + 1)
    similarities = 1 - distances[0]

    results = []
    for idx, sim in zip(indices[0], similarities):
        if idx == -1:
            continue
        if not include_self and sim >= 0.9999:  # likely self-match
            continue
        if sim >= threshold:
            results.append({'index': int(idx), 'similarity': float(sim)})

    return results[:limit]
