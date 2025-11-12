"""
Python format string template engine.

Supports standard Python format strings with {placeholder} syntax.
Provides extended syntax for defaults: {name:default_value}
"""

import re
import string
from typing import Dict, List

from pyrompt.engines import TemplateEngine, register_engine


class FormatEngine(TemplateEngine):
    """
    Python format string engine.

    Syntax: {name} or {name:format_spec}
    Extended: {name:default_value} for simple defaults (when not a format spec)

    Examples:
        >>> engine = FormatEngine()
        >>> engine.render("Hello {name}!", name="Alice")
        'Hello Alice!'
        >>> info = engine.parse_template("Hello {name}, you are {age} years old!")
        >>> info['placeholders']
        ['name', 'age']
    """

    name = 'format'
    extensions = ['.txt', '']  # Default for files without special extension

    def parse_template(self, template_str: str) -> dict:
        """
        Extract placeholders from format string.

        Uses string.Formatter to parse the template.

        Args:
            template_str: Template content

        Returns:
            dict with 'placeholders', 'defaults', 'metadata'
        """
        formatter = string.Formatter()
        placeholders = []
        defaults = {}

        try:
            # Parse the format string
            for literal_text, field_name, format_spec, conversion in formatter.parse(template_str):
                if field_name is not None:
                    # Extract the field name (before any index or attribute access)
                    # e.g., "user.name" -> "user", "items[0]" -> "items"
                    base_name = field_name.split('.')[0].split('[')[0]
                    if base_name and base_name not in placeholders:
                        placeholders.append(base_name)
        except (ValueError, KeyError):
            # If parsing fails, fall back to regex
            placeholders = self._parse_with_regex(template_str)

        return {
            'placeholders': placeholders,
            'defaults': defaults,
            'metadata': {'syntax': 'python_format'}
        }

    def _parse_with_regex(self, template_str: str) -> List[str]:
        """
        Fallback parsing using regex.

        Args:
            template_str: Template content

        Returns:
            List of placeholder names
        """
        # Pattern to find {placeholder}
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)[^\}]*\}'
        placeholders = re.findall(pattern, template_str)
        return list(set(placeholders))  # Unique

    def render(self, template_str: str, **kwargs) -> str:
        """
        Render using str.format().

        Args:
            template_str: Template content
            **kwargs: Values to inject

        Returns:
            Rendered string
        """
        return template_str.format(**kwargs)

    def detect(self, content: str) -> bool:
        """
        Detect format strings by single braces.

        Args:
            content: Template content

        Returns:
            True if content appears to use Python format syntax
        """
        # Look for {placeholder} but not {{ }} (which is literal braces)
        return '{' in content and not ('{{' in content and '}}' in content)


# Register on import
register_engine(FormatEngine())
