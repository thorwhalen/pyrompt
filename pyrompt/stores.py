"""
Storage layer for pyrompt using dol abstractions.

Provides clean Mapping/MutableMapping interfaces to filesystem storage.
"""

import os
from pathlib import Path
from typing import Optional

from dol import Files, TextFiles, JsonFiles, wrap_kvs, filt_iter, add_ipython_key_completions


def get_default_base_path() -> str:
    """
    Get platform-appropriate base path for pyrompt data.

    Returns:
        - Windows: %LOCALAPPDATA%/pyrompt
        - macOS/Linux: ~/.local/share/pyrompt
    """
    if os.name == 'nt':  # Windows
        base = os.getenv('LOCALAPPDATA')
        if not base:
            base = os.path.expanduser('~')
        return os.path.join(base, 'pyrompt')
    else:  # macOS/Linux
        # Check for XDG_DATA_HOME first
        xdg_data = os.getenv('XDG_DATA_HOME')
        if xdg_data:
            return os.path.join(xdg_data, 'pyrompt')
        return os.path.expanduser('~/.local/share/pyrompt')


def mk_prompt_store(rootdir: str, extension: str = 'txt'):
    """
    Create a file store for prompts with clean interface.

    - Keys don't show extension
    - Values are strings (not bytes)
    - Supports tab completion

    Args:
        rootdir: Directory to store prompts
        extension: File extension (without dot)

    Returns:
        MutableMapping[str, str] with file-backed storage

    Example:
        >>> store = mk_prompt_store('/tmp/prompts')
        >>> store['greeting'] = "Hello, {name}!"
        >>> print(store['greeting'])
        Hello, {name}!
    """
    # Ensure directory exists
    Path(rootdir).mkdir(parents=True, exist_ok=True)

    # Start with text files
    base = TextFiles(rootdir, max_levels=0)

    # Filter for specific extension
    ext_with_dot = f'.{extension}'
    filtered = filt_iter(base, filt=filt_iter.suffixes(ext_with_dot))

    # Hide extension from users
    store = wrap_kvs(
        filtered,
        key_of_id=lambda k: k.replace(ext_with_dot, ''),
        id_of_key=lambda k: f'{k}{ext_with_dot}'
    )

    # Add tab completion support
    return add_ipython_key_completions(store)


def mk_template_store(rootdir: str, extensions: Optional[list] = None):
    """
    Create a template store that handles multiple extensions.

    Keeps the extension in the key so we can detect template engine.

    Args:
        rootdir: Directory to store templates
        extensions: List of extensions to support (with dots)
                   Defaults to ['.txt', '.jinja2', '.j2', '.jinja', '.mustache', '.hbs']

    Returns:
        MutableMapping[str, str] with file-backed storage

    Example:
        >>> store = mk_template_store('/tmp/templates')
        >>> store['greeting.jinja2'] = "Hello {{ name }}!"
        >>> store['simple.txt'] = "Hello {name}!"
    """
    # Ensure directory exists
    Path(rootdir).mkdir(parents=True, exist_ok=True)

    if extensions is None:
        extensions = ['.txt', '.jinja2', '.j2', '.jinja', '.mustache', '.hbs']

    # Start with text files
    base = TextFiles(rootdir, max_levels=0)

    # Filter for template extensions
    filtered = filt_iter(base, filt=filt_iter.suffixes(*extensions))

    # Add tab completion (keep extensions visible)
    return add_ipython_key_completions(filtered)


def mk_metadata_store(rootdir: str, for_type: str = 'prompts'):
    """
    Create a JSON metadata store.

    Args:
        rootdir: Directory to store metadata
        for_type: Type of metadata (just for documentation)

    Returns:
        MutableMapping[str, dict] with JSON file storage

    Example:
        >>> store = mk_metadata_store('/tmp/meta')
        >>> store['greeting'] = {'author': 'thor', 'version': '1.0'}
        >>> meta = store['greeting']
    """
    # Ensure directory exists
    Path(rootdir).mkdir(parents=True, exist_ok=True)

    # JsonFiles already handles JSON serialization
    return JsonFiles(rootdir, max_levels=0)


def mk_collection_path(
    collection_name: str,
    base_path: Optional[str] = None
) -> Path:
    """
    Get the path for a collection.

    Args:
        collection_name: Name of the collection
        base_path: Optional base path (uses default if None)

    Returns:
        Path object for the collection directory
    """
    if base_path is None:
        base_path = get_default_base_path()

    base_path = os.getenv('PYROMPT_DATA_DIR', base_path)

    collection_path = Path(base_path) / 'collections' / collection_name
    return collection_path


# Utility function to create all stores for a collection
def mk_collection_stores(
    collection_name: str,
    base_path: Optional[str] = None,
    create_metadata: bool = False
) -> dict:
    """
    Create all stores for a collection.

    Args:
        collection_name: Name of the collection
        base_path: Optional base path
        create_metadata: Whether to create metadata stores

    Returns:
        Dict with keys: 'prompts', 'templates', and optionally 'prompt_meta', 'template_meta'
    """
    collection_path = mk_collection_path(collection_name, base_path)

    stores = {
        'prompts': mk_prompt_store(str(collection_path / 'prompts')),
        'templates': mk_template_store(str(collection_path / 'templates')),
    }

    if create_metadata:
        stores['prompt_meta'] = mk_metadata_store(
            str(collection_path / '_prompt_meta'),
            for_type='prompts'
        )
        stores['template_meta'] = mk_metadata_store(
            str(collection_path / '_template_meta'),
            for_type='templates'
        )

    return stores
