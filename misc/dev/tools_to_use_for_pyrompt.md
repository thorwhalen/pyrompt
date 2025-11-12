# Tools to Use for pyrompt

This document describes the i2mint ecosystem tools and patterns that will be useful for implementing pyrompt, with concrete examples and usage patterns.

## Table of Contents

1. [Core Storage Tools (dol)](#core-storage-tools-dol)
2. [Function Signature Tools (i2)](#function-signature-tools-i2)
3. [AI Operations (oa)](#ai-operations-oa)
4. [Function-Store Connections (larder)](#function-store-connections-larder)
5. [GitHub Integration (hubcap)](#github-integration-hubcap)
6. [URL Caching (graze)](#url-caching-graze)
7. [Semantic Embeddings (imbed)](#semantic-embeddings-imbed)
8. [DAG Workflows (meshed)](#dag-workflows-meshed)
9. [HuggingFace Integration (hfdol)](#huggingface-integration-hfdol)

---

## Core Storage Tools (dol)

### Overview
`dol` provides Mapping/MutableMapping interfaces to various data sources. This is the foundation for pyrompt's storage layer.

### Key Tools

#### 1. Files - Basic File Storage

```python
from dol import Files

# Create a file store (keys are relative paths, values are bytes)
store = Files('/path/to/prompts', max_levels=0)  # max_levels=0 for flat directory

# Use like a dict
store['greeting.txt'] = b"Hello, {name}!"
content = store['greeting.txt']
list(store)  # ['greeting.txt', 'farewell.txt', ...]
del store['greeting.txt']
```

**Usage in pyrompt:**
- Base storage for prompt files
- Handles binary content (bytes)
- Provides dict-like interface to filesystem

#### 2. TextFiles - Text File Storage

```python
from dol import TextFiles

# Text files (keys are relative paths, values are strings)
text_store = TextFiles('/path/to/prompts')

# Work with strings directly
text_store['system.txt'] = "You are a helpful assistant."
prompt = text_store['system.txt']  # Returns string, not bytes
```

**Usage in pyrompt:**
- Primary storage for prompt text
- More convenient than Files for text content
- Automatic encoding/decoding

#### 3. JsonFiles - JSON Storage

```python
from dol import JsonFiles

# Store metadata as JSON
metadata_store = JsonFiles('/path/to/metadata')

metadata_store['greeting_meta'] = {
    'author': 'thor',
    'version': '1.0',
    'tags': ['greeting', 'friendly'],
    'created': '2025-11-12'
}

meta = metadata_store['greeting_meta']  # Returns dict
```

**Usage in pyrompt:**
- Store prompt metadata
- Template configuration
- Engine settings

#### 4. wrap_kvs - Add Transformations

```python
from dol import Files, wrap_kvs

# Hide file extensions from users
base_store = Files('/path/to/prompts')

# Add key transformations
store = wrap_kvs(
    base_store,
    key_of_id=lambda k: k.replace('.txt', ''),  # file path -> user key
    id_of_key=lambda k: f'{k}.txt',              # user key -> file path
)

# Now users see clean keys
store['greeting'] = b"Hello!"  # Actually saves to greeting.txt
content = store['greeting']     # Actually reads from greeting.txt
list(store)                     # ['greeting', 'farewell'] (no .txt)
```

**Usage in pyrompt:**
- Hide file extensions from users
- Transform keys (e.g., for different template engines)
- Add encoding/decoding layers

**Advanced example with encoding:**

```python
from dol import wrap_kvs, Files
import json

# Store JSON objects as files
store = wrap_kvs(
    Files('/path/to/data'),
    key_of_id=lambda k: k.replace('.json', ''),
    id_of_key=lambda k: f'{k}.json',
    obj_of_data=json.loads,  # bytes -> Python object
    data_of_obj=lambda obj: json.dumps(obj, indent=2).encode()  # object -> bytes
)

# Now store Python objects directly
store['config'] = {'model': 'gpt-4', 'temperature': 0.7}
config = store['config']  # Returns dict, not bytes
```

#### 5. filt_iter - Filter Keys

```python
from dol import Files, filt_iter

store = Files('/path/to/templates')

# Filter by extension
txt_files = filt_iter(store, filt=filt_iter.suffixes('.txt'))
jinja_files = filt_iter(store, filt=filt_iter.suffixes('.jinja2'))

# Multiple extensions
template_files = filt_iter(
    store, 
    filt=filt_iter.suffixes('.txt', '.jinja2', '.mustache')
)

# Custom filter function
import re
python_files = filt_iter(
    store,
    filt=lambda k: re.match(r'.*\.py$', k)
)
```

**Usage in pyrompt:**
- Filter template files by extension
- Separate prompts from templates
- Create views of specific file types

#### 6. KeyCodecs - Transform Keys

```python
from dol import KeyCodecs, Files, wrap_kvs

# Remove extensions and normalize case
codec = KeyCodecs(
    key_of_id=lambda k: k.replace('.txt', '').lower(),
    id_of_key=lambda k: f'{k.lower()}.txt'
)

store = wrap_kvs(Files('/path'), **codec.to_wrap_kvs_kwargs())

# User sees normalized keys
store['MyPrompt'] = b"content"  # Saves to myprompt.txt
store['myprompt']               # Works!
```

**Usage in pyrompt:**
- Normalize template names
- Handle different extension types
- Create user-friendly key spaces

#### 7. add_ipython_key_completions - Tab Completion

```python
from dol import Files, add_ipython_key_completions

store = Files('/path/to/prompts')
store = add_ipython_key_completions(store)

# Now in IPython/Jupyter:
# store['gre<TAB>  -> shows: greeting, greeting_formal, greeting_casual
```

**Usage in pyrompt:**
- Enhanced developer experience
- Interactive exploration of collections
- Makes stores feel more native

### Complete Example: Prompt Store with dol

```python
from dol import Files, TextFiles, JsonFiles, wrap_kvs, filt_iter, add_ipython_key_completions
from pathlib import Path

def mk_prompt_store(rootdir: str):
    """
    Create a prompt store with clean interface.
    
    - Keys don't show .txt extension
    - Values are strings, not bytes
    - Supports tab completion
    """
    # Start with text files
    base = TextFiles(rootdir, max_levels=0)
    
    # Filter for .txt files only
    filtered = filt_iter(base, filt=filt_iter.suffixes('.txt'))
    
    # Hide .txt extension
    store = wrap_kvs(
        filtered,
        key_of_id=lambda k: k.replace('.txt', ''),
        id_of_key=lambda k: f'{k}.txt'
    )
    
    # Add tab completion
    return add_ipython_key_completions(store)

def mk_template_store(rootdir: str):
    """
    Create a template store that handles multiple extensions.
    
    Keeps the extension in the key so we can detect template engine.
    """
    base = TextFiles(rootdir, max_levels=0)
    
    # Filter for template extensions
    filtered = filt_iter(
        base,
        filt=filt_iter.suffixes('.txt', '.jinja2', '.mustache', '.j2')
    )
    
    return add_ipython_key_completions(filtered)

def mk_metadata_store(rootdir: str):
    """Create a JSON metadata store."""
    return JsonFiles(rootdir, max_levels=0)

# Usage
prompts = mk_prompt_store('~/.pyrompt/my_project/prompts')
templates = mk_template_store('~/.pyrompt/my_project/templates')
meta = mk_metadata_store('~/.pyrompt/my_project/_meta')

# Simple interface
prompts['system'] = "You are a helpful assistant."
templates['greeting.jinja2'] = "Hello {{ name|default('friend') }}!"
meta['system.json'] = {'author': 'thor', 'version': '1.0'}
```

### Pattern: Extension-Based Routing

```python
from dol import wrap_kvs, Files
import json
import pickle

def mk_multi_format_store(rootdir: str):
    """
    Store that automatically chooses codec based on extension.
    """
    base = Files(rootdir)
    
    def obj_of_data(data, key):
        """Decode based on extension in key."""
        if key.endswith('.json'):
            return json.loads(data.decode())
        elif key.endswith('.pkl'):
            return pickle.loads(data)
        else:
            return data.decode()  # text
    
    def data_of_obj(obj, key):
        """Encode based on extension in key."""
        if key.endswith('.json'):
            return json.dumps(obj).encode()
        elif key.endswith('.pkl'):
            return pickle.dumps(obj)
        else:
            return str(obj).encode()
    
    # Note: This is a simplified example. In practice, you'd need
    # to pass the key through the codec functions, which wrap_kvs
    # doesn't support directly. You'd need a custom Store subclass.
    
    return base  # Simplified for illustration
```

---

## Function Signature Tools (i2)

### Overview
`i2` provides tools for manipulating function signatures, wrapping functions with transformations, and creating clean interfaces.

### Key Tools

#### 1. Sig - Signature Manipulation

```python
from i2 import Sig

def greet(name: str, greeting: str = "Hello"):
    return f"{greeting}, {name}!"

# Get signature
sig = Sig(greet)

print(sig)  # <Sig (name: str, greeting: str = 'Hello')>
print(list(sig))  # ['name', 'greeting']
print(sig.defaults)  # {'greeting': 'Hello'}

# Modify signature
new_sig = sig.ch_defaults(greeting="Hi")
new_sig = sig.ch_annotations(name=int)
new_sig = sig - ['greeting']  # Remove parameter

# Apply new signature to function
@new_sig
def greet_modified(name):
    return f"Hi, {name}!"
```

**Usage in pyrompt:**
- Extract template parameters
- Modify function signatures for prompt_function
- Create dynamic function interfaces

#### 2. Wrap - Function Wrapping with Ingress/Egress

```python
from i2 import wrap

def process_data(x: int, y: int) -> int:
    return x + y

# Ingress: Transform inputs before function
def ingress(x: str, y: str):
    """Convert strings to ints"""
    return (int(x), int(y)), {}

# Egress: Transform output after function
def egress(result):
    """Format output"""
    return f"Result: {result}"

# Wrap the function
wrapped = wrap(process_data, ingress=ingress, egress=egress)

# Now it accepts strings
result = wrapped("5", "10")  # "Result: 15"
```

**Usage in pyrompt:**
- Pre-process template parameters
- Post-process rendered templates
- Add validation layers

#### 3. Ingress Class - Complex Input Transformation

```python
from i2 import Ingress, wrap

def my_function(a: int, b: int):
    return a * b

def kwargs_trans(outer_kwargs):
    """Transform outer interface to inner interface"""
    return dict(
        a=outer_kwargs['x'] * 2,
        b=outer_kwargs['y'] + 1
    )

ingress = Ingress(
    inner_sig=my_function,
    kwargs_trans=kwargs_trans,
    outer_sig='x: int, y: int'  # New interface
)

wrapped = wrap(my_function, ingress=ingress)

# Call with new interface
result = wrapped(x=3, y=4)  # (3*2) * (4+1) = 30
```

**Usage in pyrompt:**
- Map template parameters to function parameters
- Handle parameter name transformations
- Create flexible interfaces for templates

### Complete Example: Template Function Wrapper

```python
from i2 import wrap, Sig

def create_template_function(template_str: str, template_engine):
    """
    Wrap a template string into a callable function.
    
    The resulting function will have parameters matching the template.
    """
    # Parse template to get parameters
    params = template_engine.parse_template(template_str)
    placeholders = params['placeholders']
    defaults = params.get('defaults', {})
    
    # Create signature for the function
    sig_parts = []
    for param in placeholders:
        if param in defaults:
            sig_parts.append(f"{param}='{defaults[param]}'")
        else:
            sig_parts.append(param)
    
    sig_str = ', '.join(sig_parts)
    outer_sig = Sig(sig_str)
    
    # Create the rendering function
    def render_template(**kwargs):
        return template_engine.render(template_str, **kwargs)
    
    # Apply the signature
    return outer_sig(render_template)

# Example usage
template = "Hello {name}, welcome to {place}!"

class SimpleEngine:
    def parse_template(self, s):
        import re
        placeholders = re.findall(r'\{(\w+)\}', s)
        return {'placeholders': placeholders}
    
    def render(self, template, **kwargs):
        return template.format(**kwargs)

engine = SimpleEngine()
greet_func = create_template_function(template, engine)

print(Sig(greet_func))  # <Sig (name, place)>
result = greet_func(name="Alice", place="Wonderland")
# "Hello Alice, welcome to Wonderland!"
```

---

## AI Operations (oa)

### Overview
`oa` provides a clean interface to OpenAI's API, with functions for prompts, embeddings, and batch processing.

### Key Tools

#### 1. prompt_function - Create AI Functions from Templates

```python
from oa import prompt_function

# Simple prompt function
template = "Explain {concept} in simple terms."

explain = prompt_function(template)

# Use it
result = explain(concept="quantum computing")
# Returns: AI-generated explanation

# With system prompt
explain = prompt_function(
    template,
    prompt_func_kwargs={'system': "You are a physics teacher."}
)

# With defaults
explain = prompt_function(
    template,
    defaults={'concept': 'relativity'}
)
result = explain()  # Uses default concept
```

**Usage in pyrompt:**
- Core integration with templates
- Transform templates into callable functions
- Primary way users interact with templates

#### 2. prompt_json_function - Structured Output

```python
from oa import prompt_json_function

template = "Extract key entities from: {text}"

extract = prompt_json_function(
    template,
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

result = extract(text="Alice works at Google in New York.")
# Returns: {"people": ["Alice"], "places": ["New York"], "organizations": ["Google"]}
```

**Usage in pyrompt:**
- Templates that return structured data
- Validation and parsing
- Type-safe template outputs

#### 3. PromptFuncs - Collection of Functions

```python
from oa import PromptFuncs

# From a dict of templates
templates = {
    "summarize": "Summarize this in {n_words} words: {text}",
    "haiku": "Write a haiku about {subject}",
    "translate": "Translate to {language}: {text}"
}

funcs = PromptFuncs(template_store=templates)

# Use dot notation
print(funcs.haiku(subject="Python"))
summary = funcs.summarize(text="Long text...", n_words=50)
```

**Usage in pyrompt:**
- Collections of prompt functions
- Clean namespace for templates
- Integration with TemplateCollection

#### 4. embeddings - Generate Embeddings

```python
from oa import embeddings

# Single text
text = "semantic embeddings"
embedding = embeddings(text)  # Returns list of floats

# Multiple texts
texts = ["first text", "second text", "third text"]
vectors = embeddings(texts)  # List of lists

# Dictionary
texts = {"id1": "text 1", "id2": "text 2"}
vectors = embeddings(texts)  # Dict of id -> vector

# With options
embeddings(
    texts,
    model="text-embedding-3-small",
    dimensions=512,  # Reduce dimensionality
    batch_size=100
)
```

**Usage in pyrompt:**
- Semantic search over prompts/templates
- Building indices
- Similarity matching

#### 5. compute_embeddings (Batch API)

```python
from oa.batch_embeddings import compute_embeddings

# For large collections
segments = {
    f"prompt_{i}": f"This is prompt {i}"
    for i in range(1000)
}

# Uses OpenAI batch API (cheaper, async)
result_texts, embeddings = compute_embeddings(
    segments,
    batch_size=100,
    verbosity=1,
    poll_interval=60  # Check every minute
)

# Store embeddings
for key, embedding in zip(result_texts, embeddings):
    embedding_store[key] = embedding
```

**Usage in pyrompt:**
- Bulk embedding generation for large collections
- Cost-effective for initial index building
- Background processing

### Complete Example: Template Collection with AI Functions

```python
from oa import prompt_function, PromptFuncs
from dol import TextFiles

class TemplateCollection:
    """A collection that can create AI functions from templates."""
    
    def __init__(self, rootdir: str):
        self.store = TextFiles(rootdir)
        self._func_cache = {}
    
    def __getitem__(self, key):
        """Get template text."""
        return self.store[key]
    
    def __setitem__(self, key, value):
        """Set template text."""
        self.store[key] = value
        # Invalidate cache
        if key in self._func_cache:
            del self._func_cache[key]
    
    def to_function(self, key, **prompt_func_kwargs):
        """Convert template to callable AI function."""
        if key not in self._func_cache:
            template = self.store[key]
            self._func_cache[key] = prompt_function(
                template,
                **prompt_func_kwargs
            )
        return self._func_cache[key]
    
    def get_funcs(self, keys=None):
        """Get a PromptFuncs collection."""
        if keys is None:
            keys = list(self.store)
        
        template_dict = {k: self.store[k] for k in keys}
        return PromptFuncs(template_store=template_dict)

# Usage
templates = TemplateCollection('~/.pyrompt/templates')

# Store templates
templates['code_review'] = """
Review this {language} code:

{code}

Focus on: {focus_areas}
"""

# Convert to function
review = templates.to_function(
    'code_review',
    defaults={'focus_areas': 'correctness and readability'}
)

result = review(
    language='python',
    code='def hello(): return "world"'
)

# Or use collection
funcs = templates.get_funcs()
result = funcs.code_review(language='python', code='...')
```

---

## Function-Store Connections (larder)

### Overview
`larder` provides the CRUDE (CRUD-Execution) pattern for connecting functions to stores, enabling persistence of inputs and outputs.

### Key Concepts

#### 1. store_on_output - Persist Function Outputs

```python
from larder import store_on_output

output_store = {}

@store_on_output('result', store=output_store)
def compute(x, y):
    return x + y

result = compute(2, 3)  # Returns 5
output_store['result']  # Also contains 5

# With dynamic naming
@store_on_output(store=output_store, save_name_param='save_as')
def compute(x, y, save_as='default'):
    return x + y

compute(2, 3, save_as='sum_result')
output_store['sum_result']  # 5
```

**Usage in pyrompt:**
- Cache template renders
- Store embedding results
- Persist AI function outputs

#### 2. prepare_for_crude_dispatch - Input Resolution

```python
from larder import prepare_for_crude_dispatch

# Stores (malls)
mall = {
    'templates': {'greeting': 'Hello {name}!'},
    'names': {'user1': 'Alice', 'user2': 'Bob'},
    'outputs': {}
}

def render(template, name):
    return template.format(name=name)

# Wrap to resolve from stores
render_from_stores = prepare_for_crude_dispatch(
    render,
    param_to_mall_map={'template': 'templates', 'name': 'names'},
    mall=mall,
    output_store='outputs'
)

# Call with keys instead of values
result = render_from_stores(
    template='greeting',
    name='user1',
    save_name='greeting_alice'
)
# Returns: "Hello Alice!"
# mall['outputs']['greeting_alice'] contains the result
```

**Usage in pyrompt:**
- API endpoints that work with keys
- Store-based dispatch
- Separation of data and logic

### Complete Example: Template Rendering with Persistence

```python
from larder import store_on_output, prepare_for_crude_dispatch
from dol import TextFiles, JsonFiles

class TemplateProcessor:
    """Process templates with automatic storage."""
    
    def __init__(self, rootdir: str):
        self.templates = TextFiles(f'{rootdir}/templates')
        self.results = TextFiles(f'{rootdir}/results')
        self.metadata = JsonFiles(f'{rootdir}/metadata')
    
    def render_template(self, template_name: str, **kwargs):
        """Render a template by name."""
        template = self.templates[template_name]
        return template.format(**kwargs)
    
    def render_with_storage(self, template_name: str, result_name: str, **kwargs):
        """Render and store result."""
        @store_on_output(result_name, store=self.results)
        def _render():
            return self.render_template(template_name, **kwargs)
        
        result = _render()
        
        # Also store metadata
        self.metadata[result_name] = {
            'template': template_name,
            'parameters': kwargs,
            'timestamp': str(datetime.now())
        }
        
        return result

# Usage
processor = TemplateProcessor('~/.pyrompt/workspace')
processor.templates['greeting'] = 'Hello {name}!'

result = processor.render_with_storage(
    'greeting',
    'greeting_alice',
    name='Alice'
)
# Result stored in results/greeting_alice.txt
# Metadata stored in metadata/greeting_alice.json
```

---

## GitHub Integration (hubcap)

### Overview
`hubcap` provides dict-like access to GitHub repositories, making it easy to read and write files.

### Key Patterns

#### 1. Reading from GitHub

```python
from hubcap import GithubReader

# Connect to a user/org
reader = GithubReader('thorwhalen')

# List repositories
repos = list(reader)

# Access a repository
repo = reader['pyrompt']

# List branches
branches = list(repo)

# Access a branch
branch = repo['main']

# List files
files = list(branch)  # ['/README.md', '/pyproject.toml', ...]

# Read file content
content = branch['/README.md']  # Returns bytes
```

**Usage in pyrompt:**
- Discover prompt collections on GitHub
- Read templates from repositories
- Clone collections locally

#### 2. Pattern: GitHub-Based Collection Discovery

```python
from hubcap import GithubReader, get_repository_info
import re

def discover_prompt_collections(
    search_org: str = None,
    suffix: str = '_pyrompt',
    min_stars: int = 0
):
    """
    Discover prompt collections on GitHub.
    
    Collections should be named {name}_pyrompt for discovery.
    """
    if search_org:
        reader = GithubReader(search_org)
        repos = [r for r in reader if r.endswith(suffix)]
    else:
        # Would need GitHub API search here
        # For now, assume we have repo names
        repos = []
    
    collections = []
    for repo_name in repos:
        info = get_repository_info(f'{search_org}/{repo_name}')
        if info['stargazers_count'] >= min_stars:
            collections.append({
                'name': repo_name,
                'url': info['html_url'],
                'description': info['description'],
                'stars': info['stargazers_count']
            })
    
    return collections
```

#### 3. Writing to GitHub

```python
from hubcap import GitHubRepoManager
from dol import Files

class GitHubPromptCollection:
    """A prompt collection backed by GitHub."""
    
    def __init__(self, repo: str, token: str, branch: str = 'main'):
        self.manager = GitHubRepoManager(repo, token)
        self.branch = branch
        self.local_cache = Files('/tmp/pyrompt_cache')
    
    def __getitem__(self, key):
        """Get prompt from GitHub."""
        path = f'/prompts/{key}.txt'
        content = self.manager.read_file(path, branch=self.branch)
        return content.decode()
    
    def __setitem__(self, key, value):
        """Save prompt to GitHub."""
        path = f'/prompts/{key}.txt'
        self.manager.write_file(
            path,
            value.encode(),
            branch=self.branch,
            message=f"Update {key}"
        )
    
    def sync(self, message: str = "Update prompts"):
        """Commit and push changes."""
        self.manager.commit_and_push(message)
```

**Usage in pyrompt:**
- Publish collections to GitHub
- Collaborative editing
- Version control for prompts

---

## URL Caching (graze)

### Overview
`graze` provides transparent caching of URL content, perfect for fetching remote templates or data.

### Key Patterns

#### 1. Basic URL Caching

```python
from graze import Graze

# Create cache
g = Graze('~/.pyrompt/cache')

# Fetch URL (caches automatically)
url = 'https://example.com/template.txt'
content = g[url]  # Bytes

# Second access uses cache
content_again = g[url]  # Fast, from cache

# Check if cached
if url in g:
    print("Already cached")

# Delete cache
del g[url]
```

**Usage in pyrompt:**
- Fetch remote templates
- Cache external prompt libraries
- Offline-capable after first fetch

#### 2. TTL (Time-To-Live) Caching

```python
from graze import GrazeWithDataRefresh

# Re-fetch if older than 1 hour
g = GrazeWithDataRefresh(
    rootdir='~/.pyrompt/cache',
    time_to_live=3600,  # seconds
    on_error='ignore'   # Use stale data if refresh fails
)

content = g[url]  # Fresh or cached
```

**Usage in pyrompt:**
- Auto-update shared collections
- Balance freshness vs. performance
- Handle flaky connections

#### 3. Pattern: Remote Template Loading

```python
from graze import Graze
from dol import wrap_kvs

class RemoteTemplateLoader:
    """Load templates from URLs with caching."""
    
    def __init__(self, cache_dir: str = '~/.pyrompt/remote_cache'):
        self.cache = Graze(cache_dir)
    
    def load_template(self, url: str) -> str:
        """Load and decode template from URL."""
        content = self.cache[url]
        return content.decode('utf-8')
    
    def load_collection(self, base_url: str, manifest_file: str = 'manifest.json'):
        """
        Load a collection from a URL.
        
        Expects a manifest.json with template names and paths.
        """
        import json
        
        manifest_url = f"{base_url.rstrip('/')}/{manifest_file}"
        manifest_content = self.cache[manifest_url]
        manifest = json.loads(manifest_content)
        
        templates = {}
        for name, path in manifest['templates'].items():
            url = f"{base_url.rstrip('/')}/{path}"
            templates[name] = self.load_template(url)
        
        return templates

# Usage
loader = RemoteTemplateLoader()

# Load single template
template = loader.load_template('https://prompts.example.com/greeting.txt')

# Load collection
templates = loader.load_collection('https://prompts.example.com/collection')
```

**Usage in pyrompt:**
- Share collections via HTTP
- CDN-backed templates
- Import from gists/pastebin

---

## Semantic Embeddings (imbed)

### Overview
`imbed` provides semantic search and clustering capabilities using embeddings.

### Key Patterns

#### 1. Basic Semantic Search

```python
from oa import embeddings
import numpy as np

class SemanticIndex:
    """Simple semantic search index."""
    
    def __init__(self, texts: dict):
        """
        Args:
            texts: Dict of id -> text
        """
        self.texts = texts
        self.ids = list(texts.keys())
        
        # Compute embeddings
        text_list = [texts[k] for k in self.ids]
        self.vectors = embeddings(text_list)
        self.vectors = np.array(self.vectors)
    
    def search(self, query: str, top_k: int = 5):
        """Search for similar texts."""
        # Get query embedding
        query_vec = np.array(embeddings(query))
        
        # Compute cosine similarities
        similarities = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.ids[i], similarities[i])
            for i in top_indices
        ]
        
        return results

# Usage with prompts
prompts = {
    'python_expert': "You are a Python expert.",
    'data_viz': "You specialize in data visualizations.",
    'ml_engineer': "You are a machine learning engineer.",
}

index = SemanticIndex(prompts)
results = index.search("help with pandas dataframes", top_k=2)
# [('python_expert', 0.82), ('ml_engineer', 0.71)]
```

**Usage in pyrompt:**
- Semantic prompt discovery
- Find similar templates
- Intelligent suggestions

#### 2. Batch Embedding with Persistence

```python
from oa.batch_embeddings import compute_embeddings
from dol import PickleFiles

class PersistentSemanticIndex:
    """Semantic index with persistent storage."""
    
    def __init__(self, texts_store, embeddings_dir: str):
        self.texts = texts_store
        self.embeddings_store = PickleFiles(embeddings_dir)
    
    def build_index(self, keys=None):
        """Build index for given keys (or all if None)."""
        if keys is None:
            keys = list(self.texts)
        
        # Get texts that don't have embeddings yet
        missing_keys = [k for k in keys if k not in self.embeddings_store]
        
        if missing_keys:
            texts = {k: self.texts[k] for k in missing_keys}
            
            # Batch compute embeddings
            result_keys, vectors = compute_embeddings(
                texts,
                batch_size=100,
                verbosity=1
            )
            
            # Store embeddings
            for key, vec in zip(result_keys, vectors):
                self.embeddings_store[key] = vec
    
    def search(self, query: str, top_k: int = 5):
        """Search with persistent embeddings."""
        import numpy as np
        
        # Load all embeddings
        ids = list(self.embeddings_store)
        vectors = np.array([self.embeddings_store[k] for k in ids])
        
        # Get query embedding
        query_vec = np.array(embeddings(query))
        
        # Compute similarities
        similarities = np.dot(vectors, query_vec) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            (ids[i], similarities[i])
            for i in top_indices
        ]

# Usage
from dol import TextFiles

texts = TextFiles('~/.pyrompt/prompts')
index = PersistentSemanticIndex(
    texts,
    '~/.pyrompt/embeddings'
)

# Build index (only computes missing embeddings)
index.build_index()

# Search
results = index.search("database queries")
```

**Usage in pyrompt:**
- Efficient large-scale search
- Incremental index updates
- Background index building

---

## DAG Workflows (meshed)

### Overview
`meshed` provides DAG (Directed Acyclic Graph) workflow capabilities for complex processing pipelines.

### Key Patterns

#### 1. Simple Pipeline

```python
from meshed import DAG, FuncNode

# Define processing steps
def load_template(name):
    return f"Template: {name}"

def render_template(template, **params):
    return template + str(params)

def post_process(rendered):
    return rendered.upper()

# Create DAG
dag = DAG([
    FuncNode(load_template, out='template'),
    FuncNode(
        render_template,
        bind={'template': 'template'},
        out='rendered'
    ),
    FuncNode(
        post_process,
        bind={'rendered': 'rendered'},
        out='final'
    )
])

# Execute
result = dag(name='greeting', params={'name': 'Alice'})
```

**Usage in pyrompt:**
- Complex template processing
- Multi-stage rendering
- Conditional logic

#### 2. Pattern: Template Processing Pipeline

```python
from meshed import DAG, FuncNode

class TemplatePipeline:
    """Complex template processing with DAG."""
    
    def __init__(self):
        self.dag = None
    
    def build_pipeline(self, steps):
        """
        Build a processing pipeline.
        
        Args:
            steps: List of (name, func, bind, out) tuples
        """
        nodes = []
        for name, func, bind, out in steps:
            nodes.append(
                FuncNode(func, name=name, bind=bind, out=out)
            )
        
        self.dag = DAG(nodes)
        return self
    
    def __call__(self, **inputs):
        """Execute pipeline."""
        return self.dag(**inputs)

# Example: Multi-engine template processing
def detect_engine(template_key):
    if template_key.endswith('.jinja2'):
        return 'jinja2'
    elif template_key.endswith('.mustache'):
        return 'mustache'
    return 'format'

def load_template(template_key, templates_store):
    return templates_store[template_key]

def select_renderer(engine_type):
    renderers = {
        'jinja2': jinja2_render,
        'mustache': mustache_render,
        'format': format_render
    }
    return renderers[engine_type]

def render(template, renderer, params):
    return renderer(template, **params)

# Build pipeline
pipeline = TemplatePipeline().build_pipeline([
    ('detect', detect_engine, {'template_key': 'template_key'}, 'engine'),
    ('load', load_template, 
     {'template_key': 'template_key', 'templates_store': 'store'}, 
     'template'),
    ('select', select_renderer, {'engine_type': 'engine'}, 'renderer'),
    ('render', render, 
     {'template': 'template', 'renderer': 'renderer', 'params': 'params'}, 
     'output')
])

# Use
result = pipeline(
    template_key='greeting.jinja2',
    store=templates,
    params={'name': 'Alice'}
)
```

**Usage in pyrompt:**
- Multi-engine support
- Complex transformations
- Composable processing

---

## HuggingFace Integration (hfdol)

### Overview
`hfdol` provides a Mapping interface to HuggingFace resources.

### Key Patterns

#### 1. Pattern: HuggingFace Template Library

```python
from hfdol import HfFilesStore

class HuggingFaceTemplateLibrary:
    """Access templates hosted on HuggingFace."""
    
    def __init__(self, repo_id: str):
        """
        Args:
            repo_id: HF repo like 'username/prompt-templates'
        """
        self.store = HfFilesStore(repo_id)
    
    def list_templates(self):
        """List available templates."""
        return [k for k in self.store if k.endswith('.txt')]
    
    def get_template(self, name: str):
        """Get template content."""
        return self.store[name].decode('utf-8')
    
    def search_templates(self, keyword: str):
        """Search templates by keyword."""
        return [
            k for k in self.list_templates()
            if keyword.lower() in k.lower()
        ]

# Usage
lib = HuggingFaceTemplateLibrary('org/prompt-templates')
templates = lib.list_templates()
template = lib.get_template('code-review.txt')
```

**Usage in pyrompt:**
- Shared template libraries
- Model-specific prompts
- Community collections

---

## Summary: Key Patterns for pyrompt

### 1. Storage Architecture

```python
# Base storage pattern
from dol import TextFiles, JsonFiles, wrap_kvs, add_ipython_key_completions

def mk_collection_stores(rootdir: str):
    """Create all stores for a collection."""
    return {
        'prompts': add_ipython_key_completions(
            wrap_kvs(
                TextFiles(f'{rootdir}/prompts'),
                key_of_id=lambda k: k.replace('.txt', ''),
                id_of_key=lambda k: f'{k}.txt'
            )
        ),
        'templates': add_ipython_key_completions(
            TextFiles(f'{rootdir}/templates')
        ),
        'metadata': JsonFiles(f'{rootdir}/_meta'),
        'embeddings': PickleFiles(f'{rootdir}/_embeddings')
    }
```

### 2. Template Functions

```python
# Template to function pattern
from i2 import Sig
from oa import prompt_function

def template_to_function(template_str, engine, **kwargs):
    """Convert template to callable function."""
    # Parse to get signature
    params = engine.parse_template(template_str)
    
    # Create rendering function
    def render(**values):
        return engine.render(template_str, **values)
    
    # Apply signature and convert to AI function
    sig = Sig.from_objs(params['placeholders'])
    render = sig(render)
    
    return prompt_function(render, **kwargs)
```

### 3. Semantic Search

```python
# Semantic index pattern
from oa import embeddings
import numpy as np

class SemanticPromptIndex:
    def __init__(self, prompt_store):
        self.prompts = prompt_store
        self.build_index()
    
    def build_index(self):
        keys = list(self.prompts)
        texts = [self.prompts[k] for k in keys]
        self.vectors = np.array(embeddings(texts))
        self.keys = keys
    
    def search(self, query, top_k=5):
        query_vec = np.array(embeddings(query))
        sims = np.dot(self.vectors, query_vec)
        top = np.argsort(sims)[-top_k:][::-1]
        return [(self.keys[i], sims[i]) for i in top]
```

### 4. GitHub Integration

```python
# GitHub collection pattern
from hubcap import GithubReader

class GitHubCollection:
    def __init__(self, repo):
        self.reader = GithubReader(repo)
    
    def sync_to_local(self, local_store):
        """Sync GitHub repo to local store."""
        branch = self.reader['main']
        for path in branch:
            if path.endswith('.txt'):
                key = path.split('/')[-1].replace('.txt', '')
                content = branch[path].decode()
                local_store[key] = content
```

### 5. Remote Loading

```python
# Remote template pattern
from graze import Graze

class RemoteTemplateManager:
    def __init__(self):
        self.cache = Graze('~/.pyrompt/cache')
    
    def load_from_url(self, url):
        return self.cache[url].decode()
    
    def load_collection(self, manifest_url):
        import json
        manifest = json.loads(self.cache[manifest_url])
        return {
            k: self.load_from_url(manifest['base'] + v)
            for k, v in manifest['templates'].items()
        }
```

---

## Tool Selection Guide

### Use **dol** when:
- Building storage layers
- Need dict-like interfaces to files
- Want key transformations
- Need persistence

### Use **i2** when:
- Manipulating function signatures
- Creating wrappers
- Transforming inputs/outputs
- Building dynamic interfaces

### Use **oa** when:
- Calling OpenAI API
- Creating AI functions
- Generating embeddings
- Need structured outputs

### Use **larder** when:
- Connecting functions to stores
- Need input resolution from keys
- Want output persistence
- Building API layers

### Use **hubcap** when:
- Working with GitHub repos
- Sharing collections
- Reading remote templates
- Collaborative editing

### Use **graze** when:
- Fetching remote content
- Need caching
- Want offline capability
- Working with URLs

### Use **imbed** when:
- Building semantic search
- Clustering prompts
- Similarity matching
- Creating indices

### Use **meshed** when:
- Complex pipelines
- Multi-stage processing
- Conditional flows
- Composable workflows

---

## Next Steps

1. **Start with dol**: Build core storage layer
2. **Add i2**: Create template function wrappers
3. **Integrate oa**: Connect to OpenAI API
4. **Layer on larder**: Add CRUDE patterns for APIs
5. **Enhance with search**: Use imbed for semantic discovery
6. **Enable sharing**: Add hubcap for GitHub integration
7. **Support remote**: Use graze for URL-based templates
8. **Add workflows**: Use meshed for complex processing
