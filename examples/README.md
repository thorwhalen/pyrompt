# pyrompt Examples

This directory contains example scripts demonstrating various features of pyrompt.

## Running the Examples

```bash
# Install pyrompt first
pip install -e .

# Run examples
python examples/01_basic_usage.py
python examples/02_template_engines.py
python examples/03_utilities.py
```

## Examples Overview

### 01_basic_usage.py
- Creating prompt and template collections
- Adding and retrieving prompts
- Rendering templates
- Parsing templates to extract parameters
- Basic collection operations

### 02_template_engines.py
- Using Python format strings
- Using Jinja2 templates (if installed)
- Using Mustache templates (if installed)
- Comparing different template syntaxes
- Engine auto-detection

### 03_utilities.py
- Quick project setup with `quick_setup()`
- Bulk import/export with `import_from_dict()` and `export_to_dict()`
- Template validation with `validate_template()`
- Collection statistics with `get_stats()`
- Merging collections with `merge_collections()`
- Listing available engines

## Optional Dependencies

Some examples require optional dependencies:

```bash
# For Jinja2 examples
pip install jinja2

# For Mustache examples
pip install pystache

# For all features
pip install pyrompt[all]
```

## More Examples

For interactive examples, see the Jupyter notebook:
- `misc/pyrompt_demo.ipynb` - Comprehensive interactive demo

## CLI Examples

The pyrompt CLI provides command-line access to collections:

```bash
# Create a new collection
pyrompt new my_project

# List prompts
pyrompt list my_project

# Add a prompt
echo "You are a helpful assistant." | pyrompt add my_project system

# Show a prompt
pyrompt show my_project system

# Render a template
pyrompt render my_project greeting --params '{"name": "Alice"}'

# Show statistics
pyrompt stats my_project

# List available engines
pyrompt engines
```

See `pyrompt --help` for full CLI documentation.
