"""
Template engines example.

Shows how to use different template engines:
- Python format strings
- Jinja2 (if installed)
- Mustache (if installed)
"""

from pyrompt import TemplateCollection

templates = TemplateCollection('engines_example')

# 1. Python format strings (always available)
print("=== Python Format Strings ===\n")
templates['format.txt'] = "Name: {name}\nAge: {age}\nRole: {role}"

result = templates.render('format.txt', name='Alice', age=30, role='Developer')
print(result)
print()

# 2. Jinja2 templates (if installed)
try:
    import jinja2

    print("=== Jinja2 Templates ===\n")

    templates['jinja.jinja2'] = """
Hello {{ name|default('friend') }}!

{% if premium %}
Thank you for being a premium member!
{% else %}
Consider upgrading to premium for more features.
{% endif %}

{% if items %}
Your items:
{% for item in items %}
  {{ loop.index }}. {{ item }}
{% endfor %}
{% endif %}
"""

    result = templates.render(
        'jinja.jinja2',
        name='Bob',
        premium=True,
        items=['Item 1', 'Item 2', 'Item 3']
    )
    print(result)

except ImportError:
    print("Jinja2 not installed. Install with: pip install jinja2\n")

# 3. Mustache templates (if installed)
try:
    import pystache

    print("=== Mustache Templates ===\n")

    templates['mustache.mustache'] = """
Hello {{name}}!

{{#features}}
Feature: {{.}}
{{/features}}

{{^premium}}
Upgrade to premium for more features!
{{/premium}}
"""

    result = templates.render(
        'mustache.mustache',
        name='Charlie',
        features=['Feature A', 'Feature B', 'Feature C'],
        premium=False
    )
    print(result)

except ImportError:
    print("Pystache not installed. Install with: pip install pystache\n")

print("✓ Template engines example complete!")
