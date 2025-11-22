"""
pyrompt: Python Prompt Management

A flexible framework for managing, sharing, and searching prompts and prompt templates
with support for multiple templating languages and idiomatic Python interfaces.

Quick Start:
    >>> from pyrompt import PromptCollection, TemplateCollection
    >>> prompts = PromptCollection('my_project')
    >>> templates = TemplateCollection('my_project')
    >>> prompts['system'] = "You are a helpful Python expert."
    >>> templates['greeting'] = "Hello, {name}!"
    >>> print(templates.render('greeting', name='Alice'))
    Hello, Alice!

Main Components:
    - PromptCollection: Store and manage prompts
    - TemplateCollection: Store and manage templates with multi-engine support
    - PromptMall: Collection of collections
    - SemanticIndex: Semantic search over prompts
    - GitHubPromptCollection: GitHub-backed collections
"""

__version__ = '0.0.4'

# Core collections
from pyrompt.base import PromptCollection, TemplateCollection

# Collection of collections
from pyrompt.mall import PromptMall

# Template engines
from pyrompt.engines import (
    TemplateEngine,
    register_engine,
    get_engine,
    list_engines,
    detect_engine,
)

# Semantic search (optional - requires numpy + oa)
try:
    from pyrompt.search import SemanticIndex
    _HAVE_SEARCH = True
except ImportError:
    _HAVE_SEARCH = False
    SemanticIndex = None

# GitHub integration (optional - requires PyGithub + gitpython)
try:
    from pyrompt.github_integration import (
        GitHubPromptCollection,
        discover_prompt_collections,
        fork_collection,
        clone_collection,
    )
    _HAVE_GITHUB = True
except ImportError:
    _HAVE_GITHUB = False
    GitHubPromptCollection = None
    discover_prompt_collections = None
    fork_collection = None
    clone_collection = None

# Storage utilities
from pyrompt.stores import get_default_base_path

# Utility functions
from pyrompt.util import (
    quick_setup,
    import_from_dict,
    export_to_dict,
    validate_template,
    merge_collections,
    list_available_engines,
    render_template_file,
    create_project_structure,
    get_stats,
)


__all__ = [
    # Core
    'PromptCollection',
    'TemplateCollection',
    'PromptMall',

    # Template engines
    'TemplateEngine',
    'register_engine',
    'get_engine',
    'list_engines',
    'detect_engine',

    # Search (optional)
    'SemanticIndex',

    # GitHub (optional)
    'GitHubPromptCollection',
    'discover_prompt_collections',
    'fork_collection',
    'clone_collection',

    # Utilities
    'get_default_base_path',
    'quick_setup',
    'import_from_dict',
    'export_to_dict',
    'validate_template',
    'merge_collections',
    'list_available_engines',
    'render_template_file',
    'create_project_structure',
    'get_stats',
]


def _check_optional_dependencies():
    """Check which optional dependencies are available."""
    status = {
        'search': _HAVE_SEARCH,
        'github': _HAVE_GITHUB,
    }
    return status
