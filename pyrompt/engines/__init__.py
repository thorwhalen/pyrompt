"""
Template engine plugin system for pyrompt.

Provides extensible template engine support with auto-detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class TemplateEngine(ABC):
    """
    Abstract base class for template engines.

    Template engines parse and render templates with different syntaxes.
    They must provide:
    - name: Unique identifier (e.g., 'format', 'jinja2')
    - extensions: List of file extensions (e.g., ['.jinja2', '.j2'])
    - parse_template: Extract placeholders and defaults
    - render: Render template with values
    - detect: Optionally detect if content uses this engine
    """

    name: str  # e.g., 'format', 'jinja2', 'mustache'
    extensions: List[str]  # e.g., ['.jinja2', '.j2']

    @abstractmethod
    def parse_template(self, template_str: str) -> dict:
        """
        Parse template to extract structure.

        Args:
            template_str: Template content

        Returns:
            dict with keys:
                - 'placeholders': list of parameter names
                - 'defaults': dict of default values
                - 'metadata': any engine-specific info
        """
        pass

    @abstractmethod
    def render(self, template_str: str, **kwargs) -> str:
        """
        Render template with provided values.

        Args:
            template_str: Template content
            **kwargs: Values to inject

        Returns:
            Rendered string
        """
        pass

    def detect(self, content: str) -> bool:
        """
        Detect if content uses this engine based on syntax.

        Used when extension doesn't clearly indicate engine.
        Override to provide content-based detection.

        Args:
            content: Template content

        Returns:
            True if content appears to use this engine
        """
        return False  # Default: rely on extension


# Global registry
_ENGINE_REGISTRY: Dict[str, TemplateEngine] = {}


def register_engine(engine: TemplateEngine):
    """
    Register a template engine.

    Args:
        engine: TemplateEngine instance to register
    """
    _ENGINE_REGISTRY[engine.name] = engine


def get_engine(name: str) -> Optional[TemplateEngine]:
    """
    Get engine by name.

    Args:
        name: Engine name (e.g., 'format', 'jinja2')

    Returns:
        TemplateEngine instance or None if not found
    """
    return _ENGINE_REGISTRY.get(name)


def get_engine_by_extension(extension: str) -> Optional[TemplateEngine]:
    """
    Get engine that handles a given extension.

    Args:
        extension: File extension (e.g., '.jinja2')

    Returns:
        TemplateEngine instance or None if not found
    """
    for engine in _ENGINE_REGISTRY.values():
        if extension in engine.extensions:
            return engine
    return None


def detect_engine(content: str, extension: str = None) -> TemplateEngine:
    """
    Detect appropriate engine for content.

    Priority:
    1. Extension match (if provided)
    2. Content-based detection
    3. Default (format)

    Args:
        content: Template content
        extension: Optional file extension

    Returns:
        TemplateEngine instance (never None, falls back to 'format')
    """
    # Check extension first
    if extension:
        engine = get_engine_by_extension(extension)
        if engine:
            return engine

    # Try content detection
    for engine in _ENGINE_REGISTRY.values():
        if engine.detect(content):
            return engine

    # Default to format engine
    default_engine = get_engine('format')
    if default_engine is None:
        raise RuntimeError("No default 'format' engine registered")
    return default_engine


def list_engines() -> List[str]:
    """
    List all registered engine names.

    Returns:
        List of engine names
    """
    return list(_ENGINE_REGISTRY.keys())


# Import and register built-in engines
# These imports will auto-register the engines
from pyrompt.engines.format_engine import FormatEngine  # noqa: E402

# Optional engines (only register if dependencies are available)
try:
    from pyrompt.engines.jinja2_engine import Jinja2Engine  # noqa: F401, E402
except ImportError:
    pass  # Jinja2 not available

try:
    from pyrompt.engines.mustache_engine import MustacheEngine  # noqa: F401, E402
except ImportError:
    pass  # Pystache not available


__all__ = [
    'TemplateEngine',
    'register_engine',
    'get_engine',
    'get_engine_by_extension',
    'detect_engine',
    'list_engines',
]
