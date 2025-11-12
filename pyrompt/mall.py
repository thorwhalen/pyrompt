"""
Collection of collections - a "mall" of prompt/template stores.

Provides nested access: mall['collection_name']['prompt_key']
"""

from typing import Dict, Optional, List
from collections.abc import Mapping

from pyrompt.base import PromptCollection, TemplateCollection


class PromptMall(Mapping):
    """
    Collection of collections - a "mall" of stores.

    Provides nested access: mall['collection_name']['prompt_key']

    Examples:
        >>> mall = PromptMall('my_workspace')
        >>> mall['system']['python_expert'] = "You are a Python expert."
        >>> mall['templates']['greeting'] = "Hello {name}!"
        >>> list(mall.keys())
        ['system', 'templates']
        >>> results = mall.search('python')
    """

    def __init__(
        self,
        workspace_name: str,
        *,
        base_path: Optional[str] = None,
        collection_names: Optional[List[str]] = None,
        with_metadata: bool = False
    ):
        """
        Create a mall (collection of collections).

        Args:
            workspace_name: Name for this workspace
            base_path: Root directory
            collection_names: Initial collection names to create
            with_metadata: Whether collections should have metadata

        Example:
            >>> mall = PromptMall('my_workspace', collection_names=['system', 'templates'])
        """
        self.workspace_name = workspace_name
        self.base_path = base_path
        self.with_metadata = with_metadata

        # Storage for collections
        self._collections: Dict[str, PromptCollection] = {}
        self._collection_types: Dict[str, str] = {}  # Track type: 'prompt' or 'template'

        # Create initial collections
        if collection_names:
            for name in collection_names:
                self._create_collection(name, 'prompt')

    def _create_collection(
        self,
        name: str,
        collection_type: str = 'prompt'
    ):
        """
        Create a new collection.

        Args:
            name: Collection name
            collection_type: 'prompt' or 'template'
        """
        if collection_type == 'prompt':
            self._collections[name] = PromptCollection(
                f"{self.workspace_name}_{name}",
                base_path=self.base_path,
                with_metadata=self.with_metadata
            )
        elif collection_type == 'template':
            self._collections[name] = TemplateCollection(
                f"{self.workspace_name}_{name}",
                base_path=self.base_path,
                with_metadata=self.with_metadata
            )
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")

        self._collection_types[name] = collection_type

    def add_collection(
        self,
        name: str,
        collection_type: str = 'prompt'
    ):
        """
        Add a new collection to the mall.

        Args:
            name: Collection name
            collection_type: 'prompt' or 'template'

        Example:
            >>> mall.add_collection('personas', 'prompt')
            >>> mall['personas']['analyst'] = "You are a data analyst."
        """
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")

        self._create_collection(name, collection_type)

    def remove_collection(self, name: str):
        """
        Remove a collection from the mall.

        Note: This only removes it from the mall, not from disk.

        Args:
            name: Collection name
        """
        if name in self._collections:
            del self._collections[name]
            del self._collection_types[name]

    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        **search_kwargs
    ) -> Dict[str, List[tuple]]:
        """
        Search across multiple collections.

        Args:
            query: Search query
            collections: Collection names to search (None = all)
            **search_kwargs: Additional args for SemanticIndex.search

        Returns:
            Dict mapping collection_name -> search results

        Example:
            >>> results = mall.search('python expert', top_k=3)
            >>> for coll_name, matches in results.items():
            ...     print(f"\n{coll_name}:")
            ...     for key, score in matches:
            ...         print(f"  {key}: {score:.3f}")
        """
        try:
            from pyrompt.search import SemanticIndex
        except ImportError:
            raise ImportError(
                "Semantic search requires oa and numpy. "
                "Install with: pip install pyrompt[search]"
            )

        collections = collections or list(self._collections.keys())

        results = {}
        for coll_name in collections:
            if coll_name not in self._collections:
                continue

            coll = self._collections[coll_name]
            try:
                index = SemanticIndex(coll)
                results[coll_name] = index.search(query, **search_kwargs)
            except Exception:
                # Skip collections that fail to index
                continue

        return results

    def get_collection_type(self, name: str) -> Optional[str]:
        """
        Get the type of a collection.

        Args:
            name: Collection name

        Returns:
            'prompt', 'template', or None if not found
        """
        return self._collection_types.get(name)

    # Mapping interface
    def __getitem__(self, key: str):
        if key not in self._collections:
            # Auto-create collection on first access (as prompt collection)
            self._create_collection(key, 'prompt')
        return self._collections[key]

    def __iter__(self):
        return iter(self._collections)

    def __len__(self):
        return len(self._collections)

    def __contains__(self, key):
        return key in self._collections

    def __repr__(self):
        return (
            f"PromptMall('{self.workspace_name}', "
            f"{len(self._collections)} collections)"
        )

    def keys(self):
        """Get collection names."""
        return self._collections.keys()

    def values(self):
        """Get collections."""
        return self._collections.values()

    def items(self):
        """Get (name, collection) pairs."""
        return self._collections.items()
