# pyrompt Implementation Plan

## Overview

This document provides a comprehensive implementation plan for **pyrompt**, a flexible prompt and template management system with multi-engine template support, semantic search, and GitHub integration.

**Core Philosophy:** Follow i2mint principles - create simple facades with clean interfaces, make everything composable, use `Mapping`/`MutableMapping` abstractions via `dol`, and enable extensibility through plugins.

## Architecture Overview

```
pyrompt/
├── pyrompt/
│   ├── __init__.py              # Public API exports
│   ├── base.py                  # Core abstractions (PromptCollection, TemplateCollection)
│   ├── stores.py                # Storage implementations using dol
│   ├── engines/                 # Template engine plugins
│   │   ├── __init__.py         # Engine registry and base class
│   │   ├── format_engine.py    # Python format strings (default)
│   │   ├── jinja2_engine.py    # Jinja2 support (optional)
│   │   └── mustache_engine.py  # Mustache support (optional)
│   ├── search.py                # Semantic search using oa.embeddings
│   ├── github_integration.py   # GitHub-based collections
│   ├── mall.py                  # Collection of collections
│   └── util.py                  # Helper functions
├── tests/
│   ├── test_stores.py
│   ├── test_engines.py
│   ├── test_collections.py
│   └── test_search.py
└── pyproject.toml
```

## Implementation Phases

### Phase 1: Core Storage Layer (Foundation)

**Goal:** Implement storage abstractions using `dol` for prompts and templates.

#### 1.1 Basic File Storage (stores.py)

Use `dol` tools to create file-based stores:

```python
# Key dol tools to use:
from dol import Files, wrap_kvs, filt_iter, KeyCodecs, add_ipython_key_completions
from dol import Pipe

# Implementation pattern:
def mk_prompt_store(rootdir: str, extension: str = 'txt'):
    """
    Create a file store for prompts.
    
    Use Files as base, then wrap_kvs to add key transformations.
    """
    # Start with Files - gives us MutableMapping over filesystem
    base_store = Files(rootdir, max_levels=0)
    
    # Filter by extension using filt_iter
    filtered = filt_iter(base_store, filt=filt_iter.suffixes(f'.{extension}'))
    
    # Add key transformations to hide extensions from users
    # KeyCodecs can transform keys on read/write
    store = wrap_kvs(
        filtered,
        key_of_id=lambda k: k.replace(f'.{extension}', ''),
        id_of_key=lambda k: f'{k}.{extension}'
    )
    
    # Add IPython autocomplete support
    return add_ipython_key_completions(store)
```

**Specific Implementation Tasks:**

1. Create `mk_prompt_store(rootdir, extension='txt')` using:
   - `dol.Files` as base store
   - `dol.filt_iter.suffixes()` to filter by extension
   - `dol.wrap_kvs` with `key_of_id`/`id_of_key` to transform keys
   - `dol.add_ipython_key_completions` for autocomplete

2. Create `mk_template_store(rootdir)` that:
   - Handles multiple extensions (.txt, .jinja2, .mustache)
   - Uses `dol.filt_iter` to include all valid template extensions
   - Stores extension info in metadata for engine detection

3. Create `mk_metadata_store(rootdir, for_type='prompts')` using:
   - `dol.Files` with JSON serialization
   - `dol.wrap_kvs` with `obj_of_data=json.loads`, `data_of_obj=json.dumps`
   - Returns `MutableMapping[str, dict]`

4. Create default storage paths using platform-appropriate directories:
   - Use `pathlib.Path.home()` to get home directory
   - macOS/Linux: `~/.local/share/pyrompt/collections/`
   - Windows: `%LOCALAPPDATA%/pyrompt/collections/`

**Key dol concepts to apply:**
- `Files`: Base filesystem store
- `wrap_kvs`: Add encoding/decoding layers
- `filt_iter`: Filter keys by pattern/extension
- `KeyCodecs`: Transform keys (hide extensions, add prefixes)
- `add_ipython_key_completions`: Enable tab completion

#### 1.2 Extension-Based Codec (stores.py)

Create a codec that routes to different serializers based on file extension:

```python
from dol import wrap_kvs
import json
import pickle

# Pattern for extension-based routing:
def extension_codec(extension: str):
    """Return appropriate codec functions for extension"""
    codecs = {
        '.json': (json.dumps, json.loads),
        '.pkl': (pickle.dumps, pickle.loads),
        '.txt': (str, str),
    }
    data_of_obj, obj_of_data = codecs.get(extension, (str, str))
    return data_of_obj, obj_of_data

# Use in wrap_kvs:
def mk_multi_format_store(rootdir):
    base = Files(rootdir)
    # Detect extension from key and apply appropriate codec
    # (More complex - may need custom Store subclass)
```

### Phase 2: Template Engine Plugin System

**Goal:** Create extensible template engine registry with auto-detection.

#### 2.1 Base Template Engine (engines/__init__.py)

Define abstract base class and registry:

```python
from typing import Protocol, Dict, List, Optional
from abc import ABC, abstractmethod

class TemplateEngine(ABC):
    """Abstract base for template engines"""
    
    name: str  # e.g., 'format', 'jinja2', 'mustache'
    extensions: List[str]  # e.g., ['.jinja2', '.j2']
    
    @abstractmethod
    def parse_template(self, template_str: str) -> dict:
        """
        Parse template to extract structure.
        
        Returns:
            dict with keys:
                - 'placeholders': list of parameter names
                - 'defaults': dict of default values
                - 'metadata': any engine-specific info
        """
        pass
    
    @abstractmethod
    def render(self, template_str: str, **kwargs) -> str:
        """Render template with provided values"""
        pass
    
    def detect(self, content: str) -> bool:
        """
        Detect if content uses this engine based on syntax.
        
        Used when extension doesn't clearly indicate engine.
        """
        return False  # Default: rely on extension

# Global registry
_ENGINE_REGISTRY: Dict[str, TemplateEngine] = {}

def register_engine(engine: TemplateEngine):
    """Register a template engine"""
    _ENGINE_REGISTRY[engine.name] = engine
    
def get_engine(name: str) -> Optional[TemplateEngine]:
    """Get engine by name"""
    return _ENGINE_REGISTRY.get(name)

def detect_engine(content: str, extension: str = None) -> TemplateEngine:
    """
    Detect appropriate engine for content.
    
    Priority:
    1. Extension match (if provided)
    2. Content-based detection
    3. Default (format)
    """
    # Check extension first
    if extension:
        for engine in _ENGINE_REGISTRY.values():
            if extension in engine.extensions:
                return engine
    
    # Try content detection
    for engine in _ENGINE_REGISTRY.values():
        if engine.detect(content):
            return engine
    
    # Default to format engine
    return get_engine('format')
```

**Key concepts:**
- Use ABC for the base class
- Registry pattern for plugins
- Detection by extension then content
- Graceful fallback to default

#### 2.2 Format String Engine (engines/format_engine.py)

Implement Python format string support with custom default parsing:

```python
import re
from typing import Dict, List
from . import TemplateEngine, register_engine

class FormatEngine(TemplateEngine):
    """Python format string engine with extended default syntax"""
    
    name = 'format'
    extensions = ['.txt', '']  # Default for files without special extension
    
    def parse_template(self, template_str: str) -> dict:
        """
        Extract placeholders and defaults from format string.
        
        Supports custom syntax: {name:default_value}
        Standard format specs (e.g., {value:.2f}) are preserved.
        """
        # Pattern to find {placeholder} or {placeholder:default}
        # Must distinguish from format specs like {value:.2f}
        pattern = r'\{([^}:]+)(?::([^}]+))?\}'
        
        placeholders = []
        defaults = {}
        
        for match in re.finditer(pattern, template_str):
            name = match.group(1).strip()
            default_val = match.group(2)
            
            placeholders.append(name)
            if default_val and not self._is_format_spec(default_val):
                defaults[name] = default_val
        
        return {
            'placeholders': list(set(placeholders)),  # Unique
            'defaults': defaults,
            'metadata': {'syntax': 'python_format'}
        }
    
    def _is_format_spec(self, spec: str) -> bool:
        """Check if string is a format spec (e.g., '.2f' vs 'default_val')"""
        # Format specs typically start with format characters
        return spec.startswith(('.', '<', '>', '^', '=', '+', '-', ' ', '0')) or \
               spec.isdigit()
    
    def render(self, template_str: str, **kwargs) -> str:
        """Render using str.format()"""
        return template_str.format(**kwargs)
    
    def detect(self, content: str) -> bool:
        """Detect format strings by single braces"""
        return '{' in content and '{{' not in content

# Register on import
register_engine(FormatEngine())
```

**Key implementation details:**
- Use `re.finditer` to parse placeholders
- Distinguish format specs from default values
- Return structured parse result
- Simple `str.format()` for rendering

#### 2.3 Jinja2 Engine (engines/jinja2_engine.py)

```python
from typing import Optional
from . import TemplateEngine, register_engine

try:
    import jinja2
    HAVE_JINJA2 = True
except ImportError:
    HAVE_JINJA2 = False

class Jinja2Engine(TemplateEngine):
    """Jinja2 template engine"""
    
    name = 'jinja2'
    extensions = ['.jinja2', '.j2', '.jinja']
    
    def __init__(self):
        if not HAVE_JINJA2:
            raise ImportError("jinja2 not installed. Install with: pip install jinja2")
        self.env = jinja2.Environment()
    
    def parse_template(self, template_str: str) -> dict:
        """
        Parse Jinja2 template to extract variables.
        
        Uses Jinja2's meta.find_undeclared_variables()
        """
        ast = self.env.parse(template_str)
        placeholders = list(jinja2.meta.find_undeclared_variables(ast))
        
        # Jinja2 defaults are typically in the template via | default filter
        # Can't easily extract these, so return empty defaults
        # User should provide defaults when calling render
        
        return {
            'placeholders': placeholders,
            'defaults': {},  # Handled by Jinja2 itself
            'metadata': {'syntax': 'jinja2'}
        }
    
    def render(self, template_str: str, **kwargs) -> str:
        """Render Jinja2 template"""
        template = self.env.from_string(template_str)
        return template.render(**kwargs)
    
    def detect(self, content: str) -> bool:
        """Detect Jinja2 by {{ }} and {% %}"""
        return ('{{' in content and '}}' in content) or \
               ('{%' in content and '%}' in content)

# Register if available
if HAVE_JINJA2:
    register_engine(Jinja2Engine())
```

#### 2.4 Mustache Engine (engines/mustache_engine.py)

Similar pattern using `pystache`:

```python
from . import TemplateEngine, register_engine

try:
    import pystache
    HAVE_PYSTACHE = True
except ImportError:
    HAVE_PYSTACHE = False

class MustacheEngine(TemplateEngine):
    """Mustache template engine"""
    
    name = 'mustache'
    extensions = ['.mustache', '.hbs']
    
    def __init__(self):
        if not HAVE_PYSTACHE:
            raise ImportError("pystache not installed")
        self.renderer = pystache.Renderer()
    
    def parse_template(self, template_str: str) -> dict:
        """Parse Mustache template for variables"""
        # Use pystache's parser to find variables
        parsed = pystache.parse(template_str)
        # Extract variable names from parsed structure
        # (Implementation depends on pystache's parse tree format)
        placeholders = self._extract_vars(parsed)
        
        return {
            'placeholders': placeholders,
            'defaults': {},  # Mustache uses empty string for missing
            'metadata': {'syntax': 'mustache'}
        }
    
    def render(self, template_str: str, **kwargs) -> str:
        """Render Mustache template"""
        return self.renderer.render(template_str, kwargs)
    
    def detect(self, content: str) -> bool:
        """Detect Mustache by {{var}} without spaces"""
        return '{{' in content and '{{' in content.replace('{{ ', '{{')

if HAVE_PYSTACHE:
    register_engine(MustacheEngine())
```

### Phase 3: Collections with Template Support

**Goal:** Implement PromptCollection and TemplateCollection with engine integration.

#### 3.1 Base Collection Class (base.py)

```python
from typing import Optional, MutableMapping, Mapping
from collections.abc import MutableMapping as MutableMappingABC
import os
from pathlib import Path

from dol import wrap_kvs, add_ipython_key_completions
from .stores import mk_prompt_store, mk_metadata_store
from .engines import detect_engine, get_engine

class PromptCollection(MutableMappingABC):
    """
    Collection of prompts with MutableMapping interface.
    
    Backed by file storage via dol, with optional metadata.
    """
    
    def __init__(
        self,
        collection_name: str,
        *,
        base_path: Optional[str] = None,
        with_metadata: bool = False,
        store_factory: Optional[callable] = None
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
        self.base_path = base_path or self._default_base_path()
        self.collection_dir = Path(self.base_path) / 'collections' / collection_name
        
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
    
    @staticmethod
    def _default_base_path() -> str:
        """Get platform-appropriate base path"""
        if os.name == 'nt':  # Windows
            base = os.getenv('LOCALAPPDATA')
            return os.path.join(base, 'pyrompt')
        else:  # macOS/Linux
            return os.path.expanduser('~/.local/share/pyrompt')
    
    # MutableMapping interface
    def __getitem__(self, key: str) -> str:
        return self._store[key]
    
    def __setitem__(self, key: str, value: str):
        self._store[key] = value
    
    def __delitem__(self, key: str):
        del self._store[key]
    
    def __iter__(self):
        return iter(self._store)
    
    def __len__(self):
        return len(self._store)
    
    def __repr__(self):
        return f"PromptCollection('{self.collection_name}', {len(self)} prompts)"
```

#### 3.2 Template Collection with Engine Support (base.py)

```python
from .engines import detect_engine, get_engine, TemplateEngine

class TemplateCollection(MutableMappingABC):
    """
    Collection of templates with multi-engine support.
    
    Automatically detects template engine based on file extension
    or content. Provides rendering capabilities.
    """
    
    def __init__(
        self,
        collection_name: str,
        *,
        base_path: Optional[str] = None,
        with_metadata: bool = False,
        default_engine: str = 'format'
    ):
        self.collection_name = collection_name
        self.base_path = base_path or PromptCollection._default_base_path()
        self.collection_dir = Path(self.base_path) / 'collections' / collection_name
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
        """Detect which engine to use for a template"""
        # Extract extension(s) from key
        parts = key.split('.')
        if len(parts) > 1:
            # Try longest extension first (e.g., .jinja2.txt -> .jinja2)
            for i in range(len(parts) - 1):
                ext = '.' + '.'.join(parts[i+1:])
                engine = get_engine_by_extension(ext)
                if engine:
                    return engine
        
        # Fall back to content detection
        if key in self._store:
            content = self._store[key]
            return detect_engine(content)
        
        # Use default
        return get_engine(self.default_engine)
    
    def render(self, key: str, **kwargs) -> str:
        """
        Render a template with provided values.
        
        Args:
            key: Template key
            **kwargs: Values to inject
            
        Returns:
            Rendered template string
        """
        template_str = self._store[key]
        engine = self._detect_engine_for_key(key)
        return engine.render(template_str, **kwargs)
    
    def parse(self, key: str) -> dict:
        """
        Parse a template to extract structure.
        
        Returns dict with placeholders, defaults, metadata.
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
        
        Returns:
            Callable function that invokes LLM with rendered template
        """
        from oa import prompt_function
        
        template_str = self._store[key]
        parse_info = self.parse(key)
        
        return prompt_function(
            template_str,
            defaults=parse_info.get('defaults', {}),
            **prompt_func_kwargs
        )
    
    # MutableMapping interface delegates to _store
    def __getitem__(self, key: str) -> str:
        return self._store[key]
    
    def __setitem__(self, key: str, value: str):
        self._store[key] = value
    
    def __delitem__(self, key: str):
        del self._store[key]
    
    def __iter__(self):
        return iter(self._store)
    
    def __len__(self):
        return len(self._store)
```

**Key implementation points:**
- Use `Path` for cross-platform paths
- Detect engine per template based on extension and content
- Provide `render()` method as convenience
- Provide `to_prompt_function()` to integrate with `oa`
- Delegate MutableMapping to underlying dol store

### Phase 4: Integration with oa.prompt_function

**Goal:** Seamless conversion of templates to AI-enabled functions.

#### 4.1 Template to Function Conversion (base.py)

```python
# Add to TemplateCollection:

def create_prompt_functions(
    self,
    keys: Optional[List[str]] = None,
    **common_kwargs
) -> 'PromptFuncs':
    """
    Create PromptFuncs collection from templates.
    
    Args:
        keys: Template keys to include (None = all)
        **common_kwargs: Common kwargs for all prompt_functions
    
    Returns:
        PromptFuncs object with functions for each template
    """
    from oa import PromptFuncs
    
    keys = keys or list(self._store.keys())
    
    # Create dict of template_name -> template_string
    template_dict = {k: self._store[k] for k in keys}
    
    return PromptFuncs(
        template_store=template_dict,
        **common_kwargs
    )
```

#### 4.2 JSON Output Functions (base.py)

```python
# Add to TemplateCollection:

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
    """
    from oa import prompt_json_function
    
    template_str = self._store[key]
    parse_info = self.parse(key)
    
    return prompt_json_function(
        template_str,
        json_schema=json_schema,
        defaults=parse_info.get('defaults', {}),
        **prompt_func_kwargs
    )
```

### Phase 5: Semantic Search

**Goal:** Enable semantic search over prompts using embeddings.

#### 5.1 Semantic Index (search.py)

```python
from typing import List, Tuple, Optional, Dict
from collections.abc import Mapping
import numpy as np

from oa import embeddings
from oa.batch_embeddings import compute_embeddings

class SemanticIndex:
    """
    Semantic search index for prompts/templates.
    
    Uses oa.embeddings to create vector representations,
    then supports similarity search.
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
        """
        self.collection = collection
        self.auto_update = auto_update
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        
        # Storage for embeddings
        self._embeddings: Dict[str, np.ndarray] = {}
        
        # Build initial index
        self.rebuild()
    
    def rebuild(self):
        """Rebuild entire index from current collection"""
        keys = list(self.collection.keys())
        texts = [self.collection[k] for k in keys]
        
        if not texts:
            self._embeddings = {}
            return
        
        # Use batch computation for efficiency
        _, embedding_vectors = compute_embeddings(
            segments=texts,
            batch_size=self.batch_size,
            model=self.embedding_model,
            verbosity=0
        )
        
        # Store embeddings
        self._embeddings = {
            key: np.array(emb)
            for key, emb in zip(keys, embedding_vectors)
        }
    
    def add(self, key: str):
        """Add single item to index"""
        text = self.collection[key]
        # Single embedding computation
        emb = embeddings(text, model=self.embedding_model)
        self._embeddings[key] = np.array(emb)
    
    def remove(self, key: str):
        """Remove item from index"""
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
        """
        if not self._embeddings:
            return []
        
        # Get query embedding
        query_emb = np.array(embeddings(query, model=self.embedding_model))
        
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
        """Check if item matches metadata filters"""
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
```

**Key concepts:**
- Use `oa.embeddings` for single embeddings
- Use `oa.batch_embeddings.compute_embeddings` for bulk operations
- Implement cosine similarity for search
- Support metadata filtering
- Support diversity (MMR-like) to avoid redundant results

### Phase 6: GitHub Integration

**Goal:** Enable sharing collections via GitHub repositories.

#### 6.1 GitHub Collection (github_integration.py)

```python
from typing import Optional
import os
import tempfile
import shutil
from pathlib import Path

# Use GitHub API library
try:
    from github import Github, Repository
    HAVE_GITHUB = True
except ImportError:
    HAVE_GITHUB = False

class GitHubPromptCollection:
    """
    Prompt collection backed by GitHub repository.
    
    Repos must end with '_pyrompt' suffix.
    Contains prompts/ and/or templates/ directories.
    """
    
    def __init__(
        self,
        repo: str,
        *,
        token: Optional[str] = None,
        readonly: bool = False,
        branch: str = 'main',
        local_cache: Optional[str] = None
    ):
        """
        Create GitHub-backed collection.
        
        Args:
            repo: Repository name (user/repo_name_pyrompt)
            token: GitHub token (for write access)
            readonly: Whether collection is read-only
            branch: Branch to use
            local_cache: Local cache directory
        """
        if not HAVE_GITHUB:
            raise ImportError("PyGithub required. Install: pip install PyGithub")
        
        if not repo.endswith('_pyrompt'):
            raise ValueError("Repository must end with '_pyrompt'")
        
        self.repo_name = repo
        self.readonly = readonly
        self.branch = branch
        
        # Setup GitHub client
        self.gh = Github(token) if token else Github()
        self.repo = self.gh.get_repo(repo)
        
        # Local cache
        self.local_cache = local_cache or self._default_cache_path()
        self._ensure_local_cache()
    
    def _default_cache_path(self) -> str:
        """Get default cache path for this repo"""
        cache_base = Path.home() / '.cache' / 'pyrompt' / 'github'
        return str(cache_base / self.repo_name.replace('/', '_'))
    
    def _ensure_local_cache(self):
        """Clone or pull repo to local cache"""
        if not os.path.exists(self.local_cache):
            # Clone repo
            import git
            git.Repo.clone_from(
                f"https://github.com/{self.repo_name}",
                self.local_cache,
                branch=self.branch
            )
        else:
            # Pull latest
            import git
            repo = git.Repo(self.local_cache)
            repo.remotes.origin.pull(self.branch)
    
    def sync(self):
        """Push local changes to GitHub (if not readonly)"""
        if self.readonly:
            raise PermissionError("Collection is readonly")
        
        import git
        repo = git.Repo(self.local_cache)
        
        # Add all changes
        repo.git.add(A=True)
        
        # Commit
        if repo.is_dirty():
            repo.index.commit("Update prompts via pyrompt")
            
            # Push
            origin = repo.remote('origin')
            origin.push(self.branch)
    
    # Provide MutableMapping interface over local cache
    # (Implementation similar to PromptCollection but uses local_cache path)
```

#### 6.2 Discovery Functions (github_integration.py)

```python
def discover_prompt_collections(
    search_term: Optional[str] = None,
    min_stars: int = 0,
    language: str = 'Python'
) -> List[dict]:
    """
    Discover public *_pyrompt repositories on GitHub.
    
    Args:
        search_term: Search query (e.g., "python data")
        min_stars: Minimum star count
        language: Programming language filter
    
    Returns:
        List of dicts with repo info (name, description, stars, url)
    """
    if not HAVE_GITHUB:
        raise ImportError("PyGithub required")
    
    gh = Github()
    
    # Build search query
    query_parts = []
    if search_term:
        query_parts.append(search_term)
    query_parts.append('_pyrompt in:name')
    if language:
        query_parts.append(f'language:{language}')
    if min_stars:
        query_parts.append(f'stars:>={min_stars}')
    
    query = ' '.join(query_parts)
    
    # Search
    repos = gh.search_repositories(query, sort='stars', order='desc')
    
    # Format results
    results = []
    for repo in repos:
        results.append({
            'name': repo.full_name,
            'description': repo.description,
            'stars': repo.stargazers_count,
            'forks': repo.forks_count,
            'url': repo.html_url,
            'updated': repo.updated_at
        })
    
    return results

def fork_collection(
    source: str,
    target: str,
    token: str
) -> GitHubPromptCollection:
    """
    Fork a collection to your account.
    
    Args:
        source: Source repo (user/repo_pyrompt)
        target: Target repo (your_user/repo_pyrompt)
        token: GitHub token with repo permissions
    
    Returns:
        GitHubPromptCollection for the new fork
    """
    gh = Github(token)
    source_repo = gh.get_repo(source)
    
    # Create fork
    fork = source_repo.create_fork()
    
    # Return collection for fork
    return GitHubPromptCollection(
        repo=fork.full_name,
        token=token
    )
```

### Phase 7: Collection Mall

**Goal:** Group multiple collections together.

#### 7.1 PromptMall Implementation (mall.py)

```python
from typing import Dict, Optional, List
from collections.abc import Mapping

class PromptMall(Mapping):
    """
    Collection of collections - a "mall" of stores.
    
    Provides nested access: mall['collection_name']['prompt_key']
    """
    
    def __init__(
        self,
        workspace_name: str,
        *,
        base_path: Optional[str] = None,
        collection_names: Optional[List[str]] = None
    ):
        """
        Create a mall (collection of collections).
        
        Args:
            workspace_name: Name for this workspace
            base_path: Root directory
            collection_names: Initial collection names to create
        """
        self.workspace_name = workspace_name
        self.base_path = base_path
        
        # Storage for collections
        self._collections: Dict[str, PromptCollection] = {}
        
        # Create initial collections
        if collection_names:
            for name in collection_names:
                self._collections[name] = PromptCollection(
                    f"{workspace_name}_{name}",
                    base_path=base_path
                )
    
    def add_collection(
        self,
        name: str,
        collection_type: str = 'prompt'
    ):
        """Add a new collection to the mall"""
        if collection_type == 'prompt':
            self._collections[name] = PromptCollection(
                f"{self.workspace_name}_{name}",
                base_path=self.base_path
            )
        elif collection_type == 'template':
            self._collections[name] = TemplateCollection(
                f"{self.workspace_name}_{name}",
                base_path=self.base_path
            )
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
    
    def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        **search_kwargs
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Search across multiple collections.
        
        Args:
            query: Search query
            collections: Collection names to search (None = all)
            **search_kwargs: Additional args for SemanticIndex.search
        
        Returns:
            Dict mapping collection_name -> search results
        """
        collections = collections or list(self._collections.keys())
        
        results = {}
        for coll_name in collections:
            if coll_name not in self._collections:
                continue
            
            coll = self._collections[coll_name]
            index = SemanticIndex(coll)
            results[coll_name] = index.search(query, **search_kwargs)
        
        return results
    
    # Mapping interface
    def __getitem__(self, key: str):
        if key not in self._collections:
            # Auto-create collection on first access
            self.add_collection(key)
        return self._collections[key]
    
    def __iter__(self):
        return iter(self._collections)
    
    def __len__(self):
        return len(self._collections)
```

### Phase 8: Testing Strategy

Create comprehensive tests using pytest:

#### 8.1 Test Structure

```
tests/
├── test_stores.py          # Test dol-based storage
├── test_engines.py         # Test each template engine
├── test_collections.py     # Test PromptCollection, TemplateCollection
├── test_search.py          # Test semantic search
├── test_github.py          # Test GitHub integration (with mocks)
├── test_mall.py            # Test PromptMall
└── conftest.py             # Pytest fixtures
```

#### 8.2 Key Test Patterns

```python
# conftest.py
import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_collection_dir():
    """Provide temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_prompts():
    """Sample prompts for testing"""
    return {
        'greeting': 'Hello, {name}!',
        'farewell': 'Goodbye, {name}. See you {when}!',
        'system': 'You are a helpful assistant.'
    }

@pytest.fixture
def sample_templates():
    """Sample templates with various engines"""
    return {
        'simple.txt': 'Hello {name}',
        'advanced.jinja2': 'Hello {{ name|default("friend") }}',
        'logic.mustache': 'Hello {{name}}'
    }

# test_collections.py
def test_prompt_collection_basic(temp_collection_dir, sample_prompts):
    """Test basic CRUD operations"""
    coll = PromptCollection('test', base_path=temp_collection_dir)
    
    # Create
    coll['greeting'] = sample_prompts['greeting']
    
    # Read
    assert coll['greeting'] == sample_prompts['greeting']
    
    # Update
    coll['greeting'] = 'Hi {name}!'
    assert coll['greeting'] == 'Hi {name}!'
    
    # Delete
    del coll['greeting']
    assert 'greeting' not in coll

def test_template_collection_render(temp_collection_dir):
    """Test template rendering with different engines"""
    coll = TemplateCollection('test', base_path=temp_collection_dir)
    
    # Format engine
    coll['simple.txt'] = 'Hello {name}'
    assert coll.render('simple.txt', name='Alice') == 'Hello Alice'
    
    # Jinja2 engine (if available)
    try:
        coll['advanced.jinja2'] = 'Hello {{ name }}'
        assert coll.render('advanced.jinja2', name='Bob') == 'Hello Bob'
    except ImportError:
        pytest.skip("Jinja2 not installed")

def test_template_to_prompt_function(temp_collection_dir):
    """Test conversion to prompt function"""
    coll = TemplateCollection('test', base_path=temp_collection_dir)
    coll['greet'] = 'Hello {name}!'
    
    # Create function (mock prompt_func to avoid API calls)
    func = coll.to_prompt_function(
        'greet',
        prompt_func=lambda x: f"AI says: {x}"
    )
    
    result = func(name='Alice')
    assert 'Hello Alice' in result
```

## Development Workflow

### Step-by-Step Implementation Order

1. **Phase 1.1**: Basic file storage with dol
   - Test with simple dict-like operations
   - Ensure extensions are hidden from users

2. **Phase 2.1-2.2**: Base template engine + Format engine
   - Test placeholder extraction
   - Test default value parsing
   - Test rendering

3. **Phase 3.1**: PromptCollection
   - Test CRUD operations
   - Test metadata support

4. **Phase 3.2**: TemplateCollection
   - Test with Format engine only
   - Test engine detection

5. **Phase 2.3-2.4**: Additional engines (Jinja2, Mustache)
   - Make optional dependencies
   - Test graceful degradation

6. **Phase 4**: Integration with oa
   - Test to_prompt_function
   - Test to_prompt_json_function

7. **Phase 5**: Semantic search
   - Start with simple similarity
   - Add filters and diversity

8. **Phase 6**: GitHub integration
   - Mock GitHub API in tests
   - Test clone, sync, discovery

9. **Phase 7**: Mall
   - Test nested access
   - Test cross-collection search

## Key i2 and dol Tools Reference

### dol Tools

- **`dol.Files`**: Base filesystem store
  ```python
  from dol import Files
  store = Files('/path/to/dir', max_levels=0)  # Flat directory
  ```

- **`dol.wrap_kvs`**: Add encoding/decoding layers
  ```python
  from dol import wrap_kvs
  store = wrap_kvs(
      base_store,
      key_of_id=lambda k: k.replace('.txt', ''),
      id_of_key=lambda k: f'{k}.txt',
      obj_of_data=json.loads,
      data_of_obj=json.dumps
  )
  ```

- **`dol.filt_iter`**: Filter keys
  ```python
  from dol import filt_iter
  filtered = filt_iter(store, filt=filt_iter.suffixes('.txt', '.md'))
  ```

- **`dol.KeyCodecs`**: Transform keys
  ```python
  from dol import KeyCodecs
  codec = KeyCodecs(
      key_of_id=str.lower,
      id_of_key=str.upper
  )
  ```

- **`dol.add_ipython_key_completions`**: Enable tab completion
  ```python
  from dol import add_ipython_key_completions
  store = add_ipython_key_completions(store)
  ```

### i2 Tools

- **`i2.Sig`**: Manipulate function signatures
  ```python
  from i2 import Sig
  sig = Sig(my_function)
  new_sig = sig.ch_defaults(x=10, y=20)
  ```

- **`i2.Wrapper`**: Create function wrappers
  ```python
  from i2 import Wrapper
  wrapped = Wrapper(func, ingress=preprocess, egress=postprocess)
  ```

- **`i2.signatures`**: Extract and modify signatures
  ```python
  from i2.signatures import call_forgivingly, ensure_params
  ```

## Package Structure

```python
# pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pyrompt"
version = "0.1.0"
description = "Flexible prompt and template management with multi-engine support"
dependencies = [
    "dol>=0.2.40",
    "i2>=0.0.1",
    "oa>=0.1.0"
]

[project.optional-dependencies]
jinja2 = ["jinja2>=3.0"]
mustache = ["pystache>=0.6"]
github = ["PyGithub>=1.55", "gitpython>=3.1"]
search = ["imbed>=0.1.0", "numpy>=1.20"]
all = ["pyrompt[jinja2,mustache,github,search]"]
full = ["pyrompt[all]"]

[project.scripts]
pyrompt = "pyrompt.cli:main"
```

## Success Criteria

- [ ] Basic PromptCollection works with file storage
- [ ] TemplateCollection supports multiple engines
- [ ] Engine detection works by extension and content
- [ ] Integration with oa.prompt_function works
- [ ] Semantic search returns relevant results
- [ ] GitHub integration can clone and sync
- [ ] All tests pass
- [ ] Documentation is clear
- [ ] Examples run without errors

## Future Enhancements

1. **More template engines**: Handlebars, Guidance DSL
2. **Database backends**: MongoDB, PostgreSQL
3. **Advanced search**: Hybrid search, reranking
4. **Template versioning**: Track changes over time
5. **Collaboration**: Share and comment on prompts
6. **Analytics**: Track usage, effectiveness
7. **CLI tool**: Full command-line interface
8. **VS Code extension**: Edit prompts in IDE

---

This implementation plan provides a comprehensive roadmap for building pyrompt with proper use of dol, i2, and oa tools while maintaining extensibility and clean architecture.
