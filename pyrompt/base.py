"""
Core collection classes for prompts and templates.

Provides PromptCollection and TemplateCollection with MutableMapping interfaces.
"""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Optional, List, Callable

from pyrompt.stores import (
    mk_prompt_store,
    mk_template_store,
    mk_metadata_store,
    mk_collection_path,
    get_default_base_path,
)
from pyrompt.engines import detect_engine, get_engine, get_engine_by_extension, TemplateEngine


class PromptCollection(MutableMapping):
    """
    Collection of prompts with MutableMapping interface.

    Backed by file storage via dol, with optional metadata.

    Examples:
        >>> prompts = PromptCollection('my_project')
        >>> prompts['system'] = "You are a helpful assistant."
        >>> print(prompts['system'])
        You are a helpful assistant.
        >>> len(prompts)
        1
    """

    def __init__(
        self,
        collection_name: str,
        *,
        base_path: Optional[str] = None,
        with_metadata: bool = False,
        store_factory: Optional[Callable] = None
    ):
        """
        Create or open a prompt collection.

        Args:
            collection_name: Name of the collection
            base_path: Root directory (defaults to user data dir)
            with_metadata: Whether to enable metadata storage
            store_factory: Custom store factory (for testing/extension)
        """
        self.collection_name = collection_name
        self.base_path = base_path or get_default_base_path()
        self.collection_dir = mk_collection_path(collection_name, base_path)

        # Ensure directory exists
        self.collection_dir.mkdir(parents=True, exist_ok=True)

        # Create stores
        if store_factory:
            self._store = store_factory()
        else:
            prompts_dir = self.collection_dir / 'prompts'
            prompts_dir.mkdir(exist_ok=True)
            self._store = mk_prompt_store(str(prompts_dir))

        # Optional metadata store
        self.meta = None
        if with_metadata:
            meta_dir = self.collection_dir / '_prompt_meta'
            meta_dir.mkdir(exist_ok=True)
            self.meta = mk_metadata_store(str(meta_dir))

    # MutableMapping interface
    def __getitem__(self, key: str) -> str:
        return self._store[key]

    def __setitem__(self, key: str, value: str):
        self._store[key] = value

    def __delitem__(self, key: str):
        del self._store[key]
        # Also delete metadata if it exists
        if self.meta is not None and key in self.meta:
            del self.meta[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"PromptCollection('{self.collection_name}', {len(self)} prompts)"

    def __contains__(self, key):
        return key in self._store


class TemplateCollection(MutableMapping):
    """
    Collection of templates with multi-engine support.

    Automatically detects template engine based on file extension
    or content. Provides rendering capabilities.

    Examples:
        >>> templates = TemplateCollection('my_project')
        >>> templates['greeting.txt'] = "Hello {name}!"
        >>> templates.render('greeting.txt', name='Alice')
        'Hello Alice!'
        >>> # With Jinja2 (if installed)
        >>> templates['greeting.jinja2'] = "Hello {{ name }}!"
        >>> templates.render('greeting.jinja2', name='Bob')
        'Hello Bob!'
    """

    def __init__(
        self,
        collection_name: str,
        *,
        base_path: Optional[str] = None,
        with_metadata: bool = False,
        default_engine: str = 'format'
    ):
        """
        Create or open a template collection.

        Args:
            collection_name: Name of the collection
            base_path: Root directory (defaults to user data dir)
            with_metadata: Whether to enable metadata storage
            default_engine: Default engine name ('format', 'jinja2', etc.)
        """
        self.collection_name = collection_name
        self.base_path = base_path or get_default_base_path()
        self.collection_dir = mk_collection_path(collection_name, base_path)
        self.default_engine = default_engine

        # Create template directory
        templates_dir = self.collection_dir / 'templates'
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Store accepts any extension - we handle detection
        self._store = mk_template_store(str(templates_dir))

        # Metadata
        self.meta = None
        if with_metadata:
            meta_dir = self.collection_dir / '_template_meta'
            meta_dir.mkdir(exist_ok=True)
            self.meta = mk_metadata_store(str(meta_dir))

    def _detect_engine_for_key(self, key: str) -> TemplateEngine:
        """
        Detect which engine to use for a template.

        Args:
            key: Template key

        Returns:
            TemplateEngine instance
        """
        # Extract extension(s) from key
        parts = key.split('.')
        if len(parts) > 1:
            # Try longest extension first (e.g., .jinja2.txt -> .jinja2)
            for i in range(len(parts) - 1):
                ext = '.' + '.'.join(parts[i+1:])
                engine = get_engine_by_extension(ext)
                if engine:
                    return engine

                # Also try individual extensions
                single_ext = '.' + parts[i+1]
                engine = get_engine_by_extension(single_ext)
                if engine:
                    return engine

        # Fall back to content detection if template exists
        if key in self._store:
            content = self._store[key]
            return detect_engine(content)

        # Use default
        engine = get_engine(self.default_engine)
        if engine is None:
            raise ValueError(f"Default engine '{self.default_engine}' not found")
        return engine

    def render(self, key: str, **kwargs) -> str:
        """
        Render a template with provided values.

        Args:
            key: Template key
            **kwargs: Values to inject

        Returns:
            Rendered template string

        Example:
            >>> templates['greeting'] = "Hello {name}!"
            >>> templates.render('greeting', name='Alice')
            'Hello Alice!'
        """
        template_str = self._store[key]
        engine = self._detect_engine_for_key(key)
        return engine.render(template_str, **kwargs)

    def parse(self, key: str) -> dict:
        """
        Parse a template to extract structure.

        Returns dict with placeholders, defaults, metadata.

        Args:
            key: Template key

        Returns:
            Dict with 'placeholders', 'defaults', 'metadata'

        Example:
            >>> templates['greeting'] = "Hello {name}!"
            >>> info = templates.parse('greeting')
            >>> info['placeholders']
            ['name']
        """
        template_str = self._store[key]
        engine = self._detect_engine_for_key(key)
        return engine.parse_template(template_str)

    def to_prompt_function(self, key: str, **prompt_func_kwargs):
        """
        Convert template to an AI-enabled function using oa.prompt_function.

        Args:
            key: Template key
            **prompt_func_kwargs: Additional kwargs for prompt_function
                (e.g., 'system', 'model', 'temperature')

        Returns:
            Callable function that invokes LLM with rendered template

        Example:
            >>> from oa import prompt_function
            >>> templates['explain'] = "Explain {concept} in simple terms."
            >>> explain = templates.to_prompt_function('explain')
            >>> result = explain(concept="quantum computing")
        """
        try:
            from oa import prompt_function
        except ImportError:
            raise ImportError(
                "oa not installed. Install with: pip install oa"
            )

        template_str = self._store[key]
        parse_info = self.parse(key)

        return prompt_function(
            template_str,
            defaults=parse_info.get('defaults', {}),
            **prompt_func_kwargs
        )

    def to_prompt_json_function(
        self,
        key: str,
        json_schema: dict,
        **prompt_func_kwargs
    ):
        """
        Convert template to JSON-returning AI function.

        Uses oa.prompt_json_function to ensure structured output.

        Args:
            key: Template key
            json_schema: JSON schema for output validation
            **prompt_func_kwargs: Additional kwargs

        Returns:
            Function that returns parsed JSON dict

        Example:
            >>> schema = {
            ...     "name": "entities",
            ...     "schema": {
            ...         "type": "object",
            ...         "properties": {
            ...             "people": {"type": "array", "items": {"type": "string"}}
            ...         }
            ...     }
            ... }
            >>> templates['extract'] = "Extract people from: {text}"
            >>> extract = templates.to_prompt_json_function('extract', schema)
            >>> result = extract(text="Alice met Bob")
        """
        try:
            from oa import prompt_json_function
        except ImportError:
            raise ImportError(
                "oa not installed. Install with: pip install oa"
            )

        template_str = self._store[key]
        parse_info = self.parse(key)

        return prompt_json_function(
            template_str,
            json_schema=json_schema,
            defaults=parse_info.get('defaults', {}),
            **prompt_func_kwargs
        )

    def create_prompt_functions(
        self,
        keys: Optional[List[str]] = None,
        **common_kwargs
    ):
        """
        Create PromptFuncs collection from templates.

        Args:
            keys: Template keys to include (None = all)
            **common_kwargs: Common kwargs for all prompt_functions

        Returns:
            PromptFuncs object with functions for each template

        Example:
            >>> funcs = templates.create_prompt_functions()
            >>> result = funcs.greeting(name='Alice')
        """
        try:
            from oa import PromptFuncs
        except ImportError:
            raise ImportError(
                "oa not installed. Install with: pip install oa"
            )

        keys = keys or list(self._store.keys())

        # Create dict of template_name -> template_string
        template_dict = {k: self._store[k] for k in keys}

        return PromptFuncs(
            template_store=template_dict,
            **common_kwargs
        )

    # MutableMapping interface delegates to _store
    def __getitem__(self, key: str) -> str:
        return self._store[key]

    def __setitem__(self, key: str, value: str):
        self._store[key] = value

    def __delitem__(self, key: str):
        del self._store[key]
        # Also delete metadata if it exists
        if self.meta is not None and key in self.meta:
            del self.meta[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __repr__(self):
        return f"TemplateCollection('{self.collection_name}', {len(self)} templates)"

    def __contains__(self, key):
        return key in self._store
