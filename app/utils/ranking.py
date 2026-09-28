from typing import List, Dict, Any, Optional, Callable
import logging
from app.utils.embeddings import cosine_similarity, euclidean_distance

logger = logging.getLogger(__name__)

def rank_results(results: List[Dict[str, Any]],
                query_embedding: List[float],
                alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Rank search results by relevance to the query

    Args:
        results: List of result dictionaries with content and metadata
        query_embedding: Embedding vector of the query
        alpha: Weight factor for semantic similarity vs. other factors

    Returns:
        List of results sorted by relevance
    """
    if not results:
        return []

    if not query_embedding:
        logger.warning("No query embedding provided for ranking")
        return results

    try:
        # Calculate relevance scores for each result
        for result in results:
            # Get semantic similarity if the result has an embedding
            if "embedding" in result:
                semantic_score = cosine_similarity(query_embedding, result["embedding"])
            else:
                # Default score for results without embeddings
                semantic_score = 0.5

            # Get the result's own relevance score if available
            result_score = result.get("score", 0.5)

            # Calculate final relevance score as weighted combination
            relevance_score = alpha * semantic_score + (1 - alpha) * result_score

            # Store the score in the result
            result["relevance_score"] = relevance_score

        # Sort results by relevance score
        sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

        return sorted_results
    except Exception as e:
        logger.error(f"Error ranking results: {str(e)}")
        return results

def filter_results(results: List[Dict[str, Any]],
                  min_score: float = 0.6,
                  filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
    """Filter results by relevance score and custom filter function

    Args:
        results: List of result dictionaries with relevance scores
        min_score: Minimum relevance score to include
        filter_fn: Optional custom filter function

    Returns:
        Filtered list of results
    """
    if not results:
        return []

    try:
        # Apply minimum score filter
        score_filtered = [r for r in results if r.get("relevance_score", 0) >= min_score]

        # Apply custom filter if provided
        if filter_fn:
            filtered_results = [r for r in score_filtered if filter_fn(r)]
        else:
            filtered_results = score_filtered

        return filtered_results
    except Exception as e:
        logger.error(f"Error filtering results: {str(e)}")
        return results

def deduplicate_results(results: List[Dict[str, Any]],
                       similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """Remove duplicate or near-duplicate results

    Args:
        results: List of result dictionaries
        similarity_threshold: Threshold for considering results as duplicates

    Returns:
        Deduplicated list of results
    """
    if not results:
        return []

    try:
        deduplicated = []

        for result in results:
            # Check if this result is a duplicate of any already included result
            is_duplicate = False

            for included in deduplicated:
                # Check for exact URL match
                if result.get("url") and included.get("url") and result["url"] == included["url"]:
                    is_duplicate = True
                    break

                # Check for content similarity if embeddings are available
                if ("embedding" in result and "embedding" in included and
                    cosine_similarity(result["embedding"], included["embedding"]) > similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated
    except Exception as e:
        logger.error(f"Error deduplicating results: {str(e)}")
        return results

def diversify_results(results: List[Dict[str, Any]],
                    diversity_threshold: float = 0.7,
                    max_similar: int = 2) -> List[Dict[str, Any]]:
    """Diversify results to avoid too many similar items

    Args:
        results: List of result dictionaries
        diversity_threshold: Threshold for considering results similar
        max_similar: Maximum number of similar results to include

    Returns:
        Diversified list of results
    """
    if not results:
        return []

    try:
        diversified = []
        clusters = []

        for result in results:
            # Find a cluster for this result
            added_to_cluster = False

            for cluster in clusters:
                # Check if result is similar to cluster head
                if ("embedding" in result and "embedding" in cluster[0] and
                    cosine_similarity(result["embedding"], cluster[0]["embedding"]) > diversity_threshold):
                    # Add to cluster if not full
                    if len(cluster) < max_similar:
                        cluster.append(result)
                        diversified.append(result)
                    added_to_cluster = True
                    break

            # Create new cluster if not added to existing one
            if not added_to_cluster:
                clusters.append([result])
                diversified.append(result)

        return diversified
    except Exception as e:
        logger.error(f"Error diversifying results: {str(e)}")
        return results
