"""
Jinja2 template engine.

Provides rich templating with conditionals, loops, and filters.
"""

from typing import Optional

from pyrompt.engines import TemplateEngine, register_engine

try:
    import jinja2
    HAVE_JINJA2 = True
except ImportError:
    HAVE_JINJA2 = False


class Jinja2Engine(TemplateEngine):
    """
    Jinja2 template engine.

    Syntax: {{ variable }}, {% if %}, {% for %}, etc.
    Supports filters: {{ name|default('friend') }}

    Examples:
        >>> # Requires jinja2 installed
        >>> engine = Jinja2Engine()
        >>> template = "Hello {{ name|default('friend') }}!"
        >>> engine.render(template, name="Alice")
        'Hello Alice!'
        >>> engine.render(template)
        'Hello friend!'
    """

    name = 'jinja2'
    extensions = ['.jinja2', '.j2', '.jinja']

    def __init__(self):
        if not HAVE_JINJA2:
            raise ImportError(
                "jinja2 not installed. Install with: pip install jinja2"
            )
        # Create Jinja2 environment with safe defaults
        self.env = jinja2.Environment(
            autoescape=False,  # Don't auto-escape (we're not rendering HTML typically)
            undefined=jinja2.Undefined  # Allow undefined variables
        )

    def parse_template(self, template_str: str) -> dict:
        """
        Parse Jinja2 template to extract variables.

        Uses Jinja2's meta.find_undeclared_variables().

        Args:
            template_str: Template content

        Returns:
            dict with 'placeholders', 'defaults', 'metadata'
        """
        try:
            ast = self.env.parse(template_str)
            placeholders = list(jinja2.meta.find_undeclared_variables(ast))

            # Jinja2 defaults are typically in the template via |default filter
            # We can't easily extract these statically, so return empty defaults
            # Users should provide defaults when calling render

            return {
                'placeholders': sorted(placeholders),  # Sort for consistency
                'defaults': {},  # Handled by Jinja2 itself via |default filter
                'metadata': {'syntax': 'jinja2'}
            }
        except jinja2.TemplateSyntaxError as e:
            # If parsing fails, return minimal info
            return {
                'placeholders': [],
                'defaults': {},
                'metadata': {'syntax': 'jinja2', 'parse_error': str(e)}
            }

    def render(self, template_str: str, **kwargs) -> str:
        """
        Render Jinja2 template.

        Args:
            template_str: Template content
            **kwargs: Values to inject

        Returns:
            Rendered string

        Raises:
            jinja2.TemplateError: If template has errors
        """
        template = self.env.from_string(template_str)
        return template.render(**kwargs)

    def detect(self, content: str) -> bool:
        """
        Detect Jinja2 by {{ }} and {% %}.

        Args:
            content: Template content

        Returns:
            True if content appears to use Jinja2 syntax
        """
        return (
            ('{{' in content and '}}' in content) or
            ('{%' in content and '%}' in content) or
            ('{#' in content and '#}' in content)  # Comments
        )


# Register if available
if HAVE_JINJA2:
    register_engine(Jinja2Engine())
