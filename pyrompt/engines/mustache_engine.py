"""
Mustache template engine.

Logic-less templates with sections and partials.
"""

import re
from typing import List

from pyrompt.engines import TemplateEngine, register_engine

try:
    import pystache
    HAVE_PYSTACHE = True
except ImportError:
    HAVE_PYSTACHE = False


class MustacheEngine(TemplateEngine):
    """
    Mustache template engine.

    Syntax: {{variable}}, {{#section}}, {{^inverted}}, {{!comment}}
    Logic-less templating for cross-platform compatibility.

    Examples:
        >>> # Requires pystache installed
        >>> engine = MustacheEngine()
        >>> template = "Hello {{name}}!"
        >>> engine.render(template, name="Alice")
        'Hello Alice!'
        >>> template = "{{#premium}}Thank you!{{/premium}}"
        >>> engine.render(template, premium=True)
        'Thank you!'
    """

    name = 'mustache'
    extensions = ['.mustache', '.hbs']

    def __init__(self):
        if not HAVE_PYSTACHE:
            raise ImportError(
                "pystache not installed. Install with: pip install pystache"
            )
        self.renderer = pystache.Renderer()

    def parse_template(self, template_str: str) -> dict:
        """
        Parse Mustache template for variables.

        Extracts variable names from {{var}} and {{#section}} tags.

        Args:
            template_str: Template content

        Returns:
            dict with 'placeholders', 'defaults', 'metadata'
        """
        placeholders = self._extract_vars(template_str)

        return {
            'placeholders': sorted(list(set(placeholders))),  # Unique, sorted
            'defaults': {},  # Mustache uses empty string for missing
            'metadata': {'syntax': 'mustache'}
        }

    def _extract_vars(self, template_str: str) -> List[str]:
        """
        Extract variable names from Mustache template.

        Args:
            template_str: Template content

        Returns:
            List of variable names
        """
        placeholders = []

        # Pattern for {{variable}}
        var_pattern = r'\{\{([^#^/!>&\{][^}]*)\}\}'
        variables = re.findall(var_pattern, template_str)
        placeholders.extend(v.strip() for v in variables)

        # Pattern for {{#section}} and {{^inverted}}
        section_pattern = r'\{\{[#^]([^}]+)\}\}'
        sections = re.findall(section_pattern, template_str)
        placeholders.extend(s.strip() for s in sections)

        return placeholders

    def render(self, template_str: str, **kwargs) -> str:
        """
        Render Mustache template.

        Args:
            template_str: Template content
            **kwargs: Values to inject

        Returns:
            Rendered string
        """
        return self.renderer.render(template_str, kwargs)

    def detect(self, content: str) -> bool:
        """
        Detect Mustache by {{var}} without spaces.

        Distinguishes from Jinja2 by lack of spaces: {{var}} vs {{ var }}.

        Args:
            content: Template content

        Returns:
            True if content appears to use Mustache syntax
        """
        # Look for {{something}} pattern
        # Mustache typically has no spaces: {{name}}
        # Jinja2 typically has spaces: {{ name }}
        if '{{' not in content:
            return False

        # Check for Mustache-specific syntax
        mustache_patterns = [
            r'\{\{#',  # Sections: {{#section}}
            r'\{\{\^',  # Inverted: {{^inverted}}
            r'\{\{!',  # Comments: {{!comment}}
            r'\{\{>',  # Partials: {{>partial}}
        ]

        for pattern in mustache_patterns:
            if re.search(pattern, content):
                return True

        # If no Mustache-specific patterns, check for tight braces (no spaces)
        # This is a heuristic and not foolproof
        tight_braces = re.search(r'\{\{[^\s{]', content)
        spaced_braces = re.search(r'\{\{\s', content)

        return tight_braces and not spaced_braces


# Register if available
if HAVE_PYSTACHE:
    register_engine(MustacheEngine())
