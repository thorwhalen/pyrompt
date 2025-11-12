"""
Semantic search for prompts and templates using embeddings.

Provides similarity-based search over collections.
"""

from typing import List, Tuple, Optional, Dict
from collections.abc import Mapping

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class SemanticIndex:
    """
    Semantic search index for prompts/templates.

    Uses oa.embeddings to create vector representations,
    then supports similarity search.

    Examples:
        >>> from pyrompt import PromptCollection, SemanticIndex
        >>> prompts = PromptCollection('my_project')
        >>> prompts['python'] = "You are a Python expert."
        >>> prompts['data'] = "You specialize in data analysis."
        >>> index = SemanticIndex(prompts)
        >>> results = index.search("help with pandas", top_k=2)
    """

    def __init__(
        self,
        collection: Mapping[str, str],
        *,
        auto_update: bool = False,
        embedding_model: str = 'text-embedding-3-small',
        batch_size: int = 100
    ):
        """
        Create semantic index for a collection.

        Args:
            collection: Mapping of keys to text (prompts/templates)
            auto_update: Whether to auto-update on collection changes
            embedding_model: Model to use for embeddings
            batch_size: Batch size for bulk embedding

        Raises:
            ImportError: If numpy or oa not available
        """
        if not HAVE_NUMPY:
            raise ImportError(
                "numpy required for semantic search. Install with: pip install numpy"
            )

        try:
            from oa import embeddings
            self._embeddings_func = embeddings
        except ImportError:
            raise ImportError(
                "oa required for semantic search. Install with: pip install oa"
            )

        self.collection = collection
        self.auto_update = auto_update
        self.embedding_model = embedding_model
        self.batch_size = batch_size

        # Storage for embeddings
        self._embeddings: Dict[str, np.ndarray] = {}

        # Build initial index
        self.rebuild()

    def rebuild(self):
        """Rebuild entire index from current collection."""
        keys = list(self.collection.keys())
        texts = [self.collection[k] for k in keys]

        if not texts:
            self._embeddings = {}
            return

        # Use batch computation for efficiency
        try:
            from oa.batch_embeddings import compute_embeddings

            # Use batch API for large collections
            _, embedding_vectors = compute_embeddings(
                segments=texts,
                batch_size=self.batch_size,
                model=self.embedding_model,
                verbosity=0
            )
        except (ImportError, Exception):
            # Fall back to regular embeddings
            embedding_vectors = self._embeddings_func(
                texts,
                model=self.embedding_model
            )

        # Store embeddings
        self._embeddings = {
            key: np.array(emb)
            for key, emb in zip(keys, embedding_vectors)
        }

    def add(self, key: str):
        """
        Add single item to index.

        Args:
            key: Key of item to add
        """
        text = self.collection[key]
        # Single embedding computation
        emb = self._embeddings_func(text, model=self.embedding_model)
        self._embeddings[key] = np.array(emb)

    def remove(self, key: str):
        """
        Remove item from index.

        Args:
            key: Key of item to remove
        """
        self._embeddings.pop(key, None)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        diversity_threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar prompts.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (if collection has metadata)
            diversity_threshold: If set, exclude results too similar to each other

        Returns:
            List of (key, similarity_score) tuples, sorted by score descending

        Example:
            >>> results = index.search("help with data analysis", top_k=3)
            >>> for key, score in results:
            ...     print(f"{key}: {score:.3f}")
        """
        if not self._embeddings:
            return []

        # Get query embedding
        query_emb = np.array(self._embeddings_func(query, model=self.embedding_model))

        # Compute similarities
        similarities = []
        for key, emb in self._embeddings.items():
            # Cosine similarity
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            similarities.append((key, float(sim)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Apply filters if provided
        if filters and hasattr(self.collection, 'meta'):
            similarities = [
                (k, s) for k, s in similarities
                if self._matches_filters(k, filters)
            ]

        # Apply diversity if requested
        if diversity_threshold:
            similarities = self._diversify_results(
                similarities, diversity_threshold
            )

        return similarities[:top_k]

    def _matches_filters(self, key: str, filters: dict) -> bool:
        """
        Check if item matches metadata filters.

        Args:
            key: Item key
            filters: Dict of metadata filters

        Returns:
            True if item matches all filters
        """
        if not hasattr(self.collection, 'meta') or self.collection.meta is None:
            return True

        if key not in self.collection.meta:
            return False

        meta = self.collection.meta[key]
        for filter_key, filter_val in filters.items():
            if filter_key not in meta:
                return False
            if isinstance(filter_val, (list, set)):
                # Check if any filter value in meta value
                meta_val = meta[filter_key]
                if isinstance(meta_val, (list, set)):
                    if not set(filter_val) & set(meta_val):
                        return False
                elif meta_val not in filter_val:
                    return False
            elif meta[filter_key] != filter_val:
                return False

        return True

    def _diversify_results(
        self,
        results: List[Tuple[str, float]],
        threshold: float
    ) -> List[Tuple[str, float]]:
        """
        Remove results too similar to each other.

        Keep results that are less than threshold similar to already-selected results.
        Implements Maximal Marginal Relevance (MMR) approach.

        Args:
            results: List of (key, score) tuples
            threshold: Similarity threshold (0-1)

        Returns:
            Filtered list of results
        """
        if not results:
            return results

        diverse = [results[0]]  # Always keep top result

        for key, score in results[1:]:
            emb = self._embeddings[key]

            # Check similarity to already-selected results
            too_similar = False
            for selected_key, _ in diverse:
                selected_emb = self._embeddings[selected_key]
                sim = np.dot(emb, selected_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(selected_emb)
                )
                if sim >= threshold:
                    too_similar = True
                    break

            if not too_similar:
                diverse.append((key, score))

        return diverse

    def cluster(self, n_clusters: int = 5):
        """
        Cluster prompts/templates into groups.

        Args:
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster_id -> list of keys

        Raises:
            ImportError: If scikit-learn not available
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn required for clustering. "
                "Install with: pip install scikit-learn"
            )

        if not self._embeddings:
            return {}

        # Prepare data
        keys = list(self._embeddings.keys())
        vectors = np.array([self._embeddings[k] for k in keys])

        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(keys)), random_state=42)
        labels = kmeans.fit_predict(vectors)

        # Group by cluster
        clusters = {}
        for key, label in zip(keys, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(key)

        return clusters
