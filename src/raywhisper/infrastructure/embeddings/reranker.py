"""Cross-encoder reranker for improving search results."""

from loguru import logger
from sentence_transformers import CrossEncoder

from ...domain.value_objects.context import SearchResult


class CrossEncoderReranker:
    """Reranker using cross-encoder models for improved relevance scoring."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        """Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model to use.
                Default is BAAI/bge-reranker-v2-m3 (MIT license).
                Alternative: jinaai/jina-reranker-v2-base-multilingual (non-commercial license).
        """
        logger.info(f"Loading reranker model: {model_name}")
        self._model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 3,
    ) -> list[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: The search query.
            results: List of search results to rerank.
            top_n: Number of top results to return after reranking.

        Returns:
            List of reranked search results.
        """
        if not results:
            logger.debug("No results to rerank")
            return []

        if len(results) <= top_n:
            logger.debug(f"Results ({len(results)}) <= top_n ({top_n}), reranking all")

        logger.debug(f"Reranking {len(results)} results to top {top_n}")

        # Prepare pairs for cross-encoder
        pairs = [(query, result.content) for result in results]

        # Get scores from cross-encoder
        scores = self._model.predict(pairs)

        # Combine with original results and create new SearchResult objects with updated scores
        reranked = [
            SearchResult(
                content=result.content,
                score=float(score),
                metadata=result.metadata,
            )
            for result, score in zip(results, scores)
        ]

        # Sort by score (descending) and take top_n
        reranked.sort(key=lambda x: x.score, reverse=True)
        top_results = reranked[:top_n]

        logger.debug(
            f"Reranking complete. Top score: {top_results[0].score:.3f}, "
            f"Bottom score: {top_results[-1].score:.3f}"
        )

        return top_results

