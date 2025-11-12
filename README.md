# pyrompt: Python Prompt Management

> A flexible framework for managing, sharing, and searching prompts and prompt templates with support for multiple templating languages and idiomatic Python interfaces.

## Overview

**pyrompt** makes it easy to:
- 📝 Manage collections of prompts and templates with dict-like interfaces
- 🎨 Use multiple template formats (Jinja2, Mustache, Python format strings, etc.)
- 🔌 Extend with custom template languages via plugin system
- 🔍 Search prompts semantically using embeddings
- 🌐 Share and discover prompt collections via GitHub
- 🎯 Create AI-enabled functions from templates using `oa.prompt_function`
- 💾 Store prompts locally, on GitHub, or custom backends
- 🏗️ Build composable prompt management systems

All data sources use Python's `Mapping` and `MutableMapping` interfaces (via `dol`), and all template types are registered in an extensible plugin system.

## Quick Start

```python
from pyrompt import PromptCollection, TemplateCollection

# Create collections
prompts = PromptCollection('my_project')
templates = TemplateCollection('my_project')

# Store a simple prompt
prompts['system'] = "You are a helpful Python expert."

# Store templates (default: Python format strings)
templates['greeting'] = "Hello, {name}! Welcome to {place}."

# Use template with format syntax
print(templates['greeting'].format(name='Alice', place='Wonderland'))
# Output: Hello, Alice! Welcome to Wonderland.

# Create an AI function from a template
from oa import prompt_function

greet = prompt_function(templates['greeting'])
response = greet(name='Bob', place='the future')
```

## Installation

```bash
# Basic installation (Python format strings only)
pip install pyrompt

# With Jinja2 support (recommended)
pip install pyrompt[jinja2]

# With all template engines
pip install pyrompt[all]

# Full installation with semantic search
pip install pyrompt[full]
```

**Dependencies:**
- `dol` - Storage abstractions (Mapping/MutableMapping interfaces)
- `oa` - AI operations (prompt_function, embeddings)
- `i2` - Function signature manipulation and wrapping
- `jinja2` (optional) - Jinja2 templating support
- `pystache` (optional) - Mustache templating support
- `imbed` (optional) - Advanced semantic search

## Template Languages

pyrompt supports multiple template formats with an extensible plugin system. Choose the right tool for your needs:

### Default: Python Format Strings

Simple and built-in, perfect for basic substitution:

```python
templates['basic'] = "Hello, {name}! You are {age} years old."
```

**Pros:** Native Python, no dependencies, simple
**Cons:** No conditionals, no loops, no default values in template

### Jinja2 (Recommended for Complex Prompts)

Rich templating with conditionals, loops, and filters:

```python
templates['advanced.jinja2'] = """
Hello, {{ name|default('friend') }}!

{% if role %}
You are logged in as: {{ role }}
{% else %}
You are browsing as a guest.
{% endif %}

{% if items %}
Your items:
{% for item in items %}
  - {{ item }}
{% endfor %}
{% endif %}
"""

# Use with Jinja2 engine
from pyrompt.engines import Jinja2Engine
result = templates.render('advanced.jinja2', 
    name='Alice', 
    role='admin',
    items=['prompt1', 'prompt2']
)
```

**Pros:** Powerful, widely used, great for complex prompts
**Cons:** Requires jinja2 dependency, more complex syntax

### Mustache (Logic-less Templates)

Cross-platform, simple conditional sections:

```python
templates['mustache.mustache'] = """
Hello {{name}}!

{{#premium}}
Thank you for being a premium member!
{{/premium}}

{{^premium}}
Consider upgrading to premium.
{{/premium}}

{{#features}}
- {{.}}
{{/features}}
"""

# Mustache automatically handles sections and loops
result = templates.render('mustache.mustache',
    name='Bob',
    premium=True,
    features=['feature1', 'feature2']
)
```

**Pros:** Simple, logic-less, cross-language standard
**Cons:** Limited power, requires pystache dependency

### Template Auto-Detection

pyrompt automatically detects template types by extension:

```python
# These are automatically parsed with the right engine:
templates['simple.txt'] = "Hello {name}"           # Python format
templates['complex.jinja2'] = "Hello {{ name }}"   # Jinja2
templates['logic.mustache'] = "Hello {{name}}"     # Mustache

# Sub-extensions also work:
templates['code.jinja2.txt'] = "..."  # Detected as Jinja2
templates['data.mustache.md'] = "..."  # Detected as Mustache
```

## Core Concepts

### Collections with Metadata

```python
from pyrompt import PromptCollection

prompts = PromptCollection('my_project', with_metadata=True)

# Store a prompt
prompts['data_analyst'] = "You are an expert data analyst specializing in Python."

# Add metadata
prompts.meta['data_analyst'] = {
    'author': 'thor',
    'version': '1.0',
    'tags': ['system', 'data', 'python'],
    'template_engine': 'format',  # or 'jinja2', 'mustache', etc.
    'created': '2025-11-12',
    'description': 'System prompt for data analysis tasks'
}
```

### Creating AI Functions

Transform templates into callable AI functions:

```python
from pyrompt import TemplateCollection
from oa import prompt_function, prompt_json_function

templates = TemplateCollection('my_project')

# Simple prompt function
templates['code_review'] = """
Review this {language} code:

{code}

Focus on: {focus_areas}
"""

review_code = prompt_function(
    templates['code_review'],
    defaults={'focus_areas': 'correctness and readability'}
)

result = review_code(
    language='python',
    code='def hello(): return "world"'
)

# JSON-returning function
templates['extract_entities'] = """
Extract entities from this text: {text}

Return a JSON object with keys: people, places, organizations
"""

extract = prompt_json_function(
    templates['extract_entities'],
    json_schema={
        "name": "entities",
        "schema": {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "places": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
)

entities = extract(text="Alice works at Google in New York.")
# Returns: {"people": ["Alice"], "places": ["New York"], "organizations": ["Google"]}
```

### Using PromptFuncs for Collections

```python
from oa import PromptFuncs

# Create a collection of AI-enabled functions
funcs = PromptFuncs(
    template_store=templates,
    # Automatically creates functions for all templates
)

# Or from a dict
funcs = PromptFuncs(
    template_store={
        "haiku": "Write a haiku about {subject}. Only output the haiku.",
        "summarize": "Summarize this in {n_words} words: {text}"
    }
)

# Use them
print(funcs.haiku(subject="Python"))
print(funcs.summarize(text="Long text...", n_words="50"))
```

## Storage Locations

Default storage:
- **macOS/Linux**: `~/.local/share/pyrompt/collections/{collection_name}/`
- **Windows**: `%LOCALAPPDATA%/pyrompt/collections/{collection_name}/`

Structure:
```
my_project/
├── prompts/                    # Simple prompts
├── templates/                  # Templates (auto-detected by extension)
│   ├── simple.txt             # Python format
│   ├── advanced.jinja2        # Jinja2
│   └── logic.mustache         # Mustache
├── _prompt_meta/              # JSON metadata
└── _template_meta/            # JSON metadata
```

## Custom Storage Location

```python
from pyrompt import PromptCollection

# Use a specific directory
prompts = PromptCollection('my_project', base_path='/custom/path')

# Or environment variable
import os
os.environ['PYROMPT_DATA_DIR'] = '/my/prompts'
prompts = PromptCollection('my_project')
```

## GitHub Integration

### Publishing Your Collection

```python
from pyrompt import GitHubPromptCollection

gh_prompts = GitHubPromptCollection(
    repo='username/my_project_pyrompt',  # Must end with '_pyrompt'
    token='your_github_token'
)

gh_prompts['sql_expert'] = "You are an expert in SQL optimization."
gh_prompts.sync()  # Commits and pushes
```

### Discovering Collections

```python
from pyrompt import discover_prompt_collections

collections = discover_prompt_collections(
    search_term='python data',
    min_stars=10
)

for repo in collections:
    print(f"{repo['name']}: {repo['description']}")
```

## Semantic Search

Build semantic indices for intelligent discovery:

```python
from pyrompt import PromptCollection, SemanticIndex

prompts = PromptCollection('my_project')

# Add prompts
prompts['python_expert'] = "You are a Python programming expert."
prompts['data_viz'] = "You specialize in data visualizations."
prompts['ml_engineer'] = "You are a machine learning engineer."

# Build semantic index (uses oa.embeddings)
index = SemanticIndex(prompts)

# Search semantically
results = index.search("help with pandas dataframes", top_k=3)
for name, score in results:
    print(f"{name} (score: {score:.3f})")
```

### Batch Embeddings

For large collections:

```python
from oa.batch_embeddings import compute_embeddings

prompt_keys = list(prompts.keys())
prompt_texts = [prompts[k] for k in prompt_keys]

# Compute in batches (uses OpenAI batch API)
result_texts, embeddings = compute_embeddings(
    segments=prompt_texts,
    batch_size=100,
    verbosity=1
)

# Store embeddings
for key, embedding in zip(prompt_keys, embeddings):
    prompts.embeddings[key] = embedding
```

## Extending with Custom Template Engines

Register your own template engines:

```python
from pyrompt.engines import TemplateEngine, register_engine

class MyCustomEngine(TemplateEngine):
    """Custom template engine"""
    
    name = "custom"
    extensions = ['.custom', '.cust']
    
    def parse_template(self, template_str: str) -> dict:
        """Extract placeholders and defaults"""
        # Your parsing logic
        return {
            'placeholders': ['name', 'value'],
            'defaults': {'value': '42'}
        }
    
    def render(self, template_str: str, **kwargs) -> str:
        """Render template with values"""
        # Your rendering logic
        return template_str  # simplified
    
    def detect(self, content: str) -> bool:
        """Detect if content uses this engine"""
        return '<<' in content and '>>' in content

# Register it
register_engine(MyCustomEngine())

# Now use it
templates['my_template.custom'] = "Hello <<name>>, value is <<value>>"
```

## Collection of Collections (Malls)

Organize multiple collections:

```python
from pyrompt import PromptMall

mall = PromptMall('my_workspace')

# Access different collections
mall['system_prompts']['python_expert'] = "You are a Python expert."
mall['templates']['code_review'] = "Review this {language} code..."
mall['personas']['analyst'] = "You are a data analyst..."

# List collections
list(mall.keys())  # ['system_prompts', 'templates', 'personas']

# Search across collections
results = mall.search('python', collections=['system_prompts', 'personas'])
```

## Real-World Example

```python
from pyrompt import PromptCollection, TemplateCollection
from oa import prompt_function, PromptFuncs

# Set up collections
system_prompts = PromptCollection('system')
templates = TemplateCollection('templates')

# System prompts
system_prompts['code_reviewer'] = """You are an expert code reviewer.
Focus on: correctness, efficiency, readability, and best practices."""

# Jinja2 template with conditionals
templates['review_pr.jinja2'] = """
Review this pull request:

**Title**: {{ pr_title }}
**Description**: {{ pr_description }}

{% if changed_files %}
**Changed Files**:
{{ changed_files }}
{% endif %}

Provide:
1. Summary of changes
2. Potential issues
{% if include_suggestions %}
3. Suggestions for improvement
{% endif %}
"""

# Create AI function
review_pr = prompt_function(
    templates['review_pr.jinja2'],
    prompt_func_kwargs={'system': system_prompts['code_reviewer']}
)

# Use it
result = review_pr(
    pr_title="Add new feature",
    pr_description="Implements user authentication",
    changed_files="auth.py, models.py",
    include_suggestions=True
)
```

## Template Format Comparison

| Feature | Format | Jinja2 | Mustache |
|---------|--------|--------|----------|
| Syntax | `{var}` | `{{ var }}` | `{{var}}` |
| Conditionals | No | Yes (`{% if %}`) | Yes (`{{#var}}`) |
| Loops | No | Yes (`{% for %}`) | Yes (`{{#array}}`) |
| Defaults | No | Yes (`\|default`) | Via sections |
| Filters | No | Yes | No |
| Dependencies | None | jinja2 | pystache |
| Complexity | Simple | Medium | Simple |
| Best For | Basic substitution | Complex prompts | Cross-platform |

**Recommendation:** Start with Python format strings for simple prompts. Graduate to Jinja2 when you need conditionals, loops, or default values. Use Mustache if you need cross-language compatibility.

## Architecture Principles

Following `i2mint` philosophy, `pyrompt`:

1. **Facades everything**: Storage via `dol.Mapping`/`MutableMapping`, templates via unified interface
2. **Plugin system**: Register new template engines without modifying core
3. **Composable**: Build complex workflows from simple components
4. **Type-hinted**: Full type coverage for IDE support
5. **Zero magic**: Explicit interfaces, clear abstractions
6. **Open-closed**: Extend functionality without changing existing code

## Implementation Stack

**Core dependencies:**
- `dol`: Storage abstractions and key-value transformations
  - `dol.Files`, `dol.wrap_kvs`, `dol.filt_iter`, `dol.KeyCodecs`
- `i2`: Function manipulation and signature handling
  - `i2.Sig`, `i2.Wrapper`, `i2.signatures`
- `oa`: AI operations
  - `oa.prompt_function`, `oa.prompt_json_function`
  - `oa.embeddings`, `oa.PromptFuncs`

**Optional integrations:**
- `imbed`: Semantic search and clustering
- `jinja2`: Advanced templating
- `pystache`: Mustache templates

## CLI Usage (Planned)

```bash
# Create a new collection
pyrompt new my_project

# Add a prompt
pyrompt add my_project/greeting "Hello, {name}!"

# List templates
pyrompt list my_project --templates

# Test a template
pyrompt render my_project/greeting.jinja2 --name Alice

# Search prompts
pyrompt search my_project "help with databases"

# Sync to GitHub
pyrompt sync my_project --repo username/my_project_pyrompt

# Discover collections
pyrompt discover --search "python" --min-stars 10
```

## Contributing

We welcome contributions! Priority areas:

- Additional template engines (Handlebars, Guidance DSL, etc.)
- Enhanced semantic search algorithms
- Prompt optimization tools
- Template testing utilities
- Collection templates for common use cases

See `IMPLEMENTATION_PLAN.md` for development roadmap.

## License

MIT License - See LICENSE file for details

---

**Key Features:**
- 🎨 Multiple template formats (Format, Jinja2, Mustache, extensible)
- 🔌 Plugin architecture for custom engines
- 🤖 Direct integration with `oa.prompt_function` and `oa.prompt_json_function`
- 📦 Clean storage via `dol` abstractions
- 🔍 Semantic search with `oa.embeddings`
- 🌐 GitHub sharing and discovery
- 🏗️ Built on `i2mint` principles: facades, composition, extensibility
