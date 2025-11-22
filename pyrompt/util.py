"""
Utility functions and convenience helpers for pyrompt.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os


def quick_setup(
    project_name: str,
    with_templates: bool = True,
    with_metadata: bool = False,
    base_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick setup for a new pyrompt project.

    Creates both prompt and template collections with sensible defaults.

    Args:
        project_name: Name of the project
        with_templates: Whether to create a template collection
        with_metadata: Whether to enable metadata
        base_path: Optional base path

    Returns:
        Dict with 'prompts' and optionally 'templates' keys

    Example:
        >>> collections = quick_setup('my_ai_project')
        >>> collections['prompts']['system'] = "You are a helpful assistant."
        >>> collections['templates']['greeting'] = "Hello {name}!"
    """
    from pyrompt import PromptCollection, TemplateCollection

    result = {
        'prompts': PromptCollection(
            project_name,
            base_path=base_path,
            with_metadata=with_metadata
        )
    }

    if with_templates:
        result['templates'] = TemplateCollection(
            project_name,
            base_path=base_path,
            with_metadata=with_metadata
        )

    return result


def import_from_dict(
    collection,
    prompts: Dict[str, str],
    metadata: Optional[Dict[str, dict]] = None
):
    """
    Import prompts from a dictionary.

    Args:
        collection: PromptCollection or TemplateCollection
        prompts: Dict mapping keys to prompt/template strings
        metadata: Optional dict mapping keys to metadata dicts

    Example:
        >>> prompts = PromptCollection('my_project')
        >>> import_from_dict(prompts, {
        ...     'system': 'You are helpful.',
        ...     'user': 'Hello!'
        ... })
    """
    for key, value in prompts.items():
        collection[key] = value

    if metadata and hasattr(collection, 'meta') and collection.meta is not None:
        for key, meta in metadata.items():
            if key in collection:
                collection.meta[key] = meta


def export_to_dict(
    collection,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Export collection to a dictionary.

    Args:
        collection: PromptCollection or TemplateCollection
        include_metadata: Whether to include metadata

    Returns:
        Dict with 'prompts' and optionally 'metadata' keys

    Example:
        >>> prompts = PromptCollection('my_project')
        >>> data = export_to_dict(prompts)
        >>> print(data['prompts'])
    """
    result = {
        'prompts': {key: collection[key] for key in collection}
    }

    if include_metadata and hasattr(collection, 'meta') and collection.meta is not None:
        result['metadata'] = {key: collection.meta[key] for key in collection.meta}

    return result


def validate_template(
    template_str: str,
    engine_name: str = 'format',
    required_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate a template string.

    Args:
        template_str: Template content
        engine_name: Engine to use ('format', 'jinja2', 'mustache')
        required_params: Optional list of required parameter names

    Returns:
        Dict with 'valid', 'errors', 'warnings', 'placeholders'

    Example:
        >>> result = validate_template("Hello {name}!", required_params=['name'])
        >>> assert result['valid']
        >>> result = validate_template("Hello {name}!", required_params=['name', 'age'])
        >>> assert not result['valid']
    """
    from pyrompt.engines import get_engine

    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'placeholders': []
    }

    # Get engine
    engine = get_engine(engine_name)
    if engine is None:
        result['valid'] = False
        result['errors'].append(f"Engine '{engine_name}' not found")
        return result

    # Try to parse
    try:
        parsed = engine.parse_template(template_str)
        result['placeholders'] = parsed['placeholders']
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Parse error: {str(e)}")
        return result

    # Check required params
    if required_params:
        missing = set(required_params) - set(result['placeholders'])
        if missing:
            result['valid'] = False
            result['errors'].append(f"Missing required parameters: {missing}")

        extra = set(result['placeholders']) - set(required_params)
        if extra:
            result['warnings'].append(f"Extra parameters not in required list: {extra}")

    return result


def merge_collections(
    target,
    *sources,
    conflict_strategy: str = 'skip'
):
    """
    Merge multiple collections into a target collection.

    Args:
        target: Target collection to merge into
        *sources: Source collections to merge from
        conflict_strategy: How to handle conflicts ('skip', 'overwrite', 'error')

    Example:
        >>> target = PromptCollection('merged')
        >>> source1 = PromptCollection('source1')
        >>> source2 = PromptCollection('source2')
        >>> merge_collections(target, source1, source2, conflict_strategy='skip')
    """
    for source in sources:
        for key in source:
            if key in target:
                if conflict_strategy == 'skip':
                    continue
                elif conflict_strategy == 'error':
                    raise ValueError(f"Key '{key}' already exists in target")
                # else: overwrite
            target[key] = source[key]


def list_available_engines() -> Dict[str, Dict[str, Any]]:
    """
    List all available template engines with their details.

    Returns:
        Dict mapping engine name to info dict

    Example:
        >>> engines = list_available_engines()
        >>> print(engines['format']['extensions'])
        ['.txt', '']
    """
    from pyrompt.engines import list_engines, get_engine

    result = {}
    for name in list_engines():
        engine = get_engine(name)
        result[name] = {
            'name': engine.name,
            'extensions': engine.extensions,
            'available': True
        }

    return result


def render_template_file(
    file_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Render a template file directly.

    Args:
        file_path: Path to template file
        output_path: Optional path to save rendered output
        **kwargs: Template parameters

    Returns:
        Rendered string

    Example:
        >>> result = render_template_file(
        ...     'templates/greeting.txt',
        ...     name='Alice'
        ... )
    """
    from pyrompt.engines import detect_engine

    # Read template
    with open(file_path, 'r') as f:
        template_str = f.read()

    # Detect engine by extension
    extension = '.' + file_path.split('.')[-1] if '.' in file_path else ''
    engine = detect_engine(template_str, extension)

    # Render
    result = engine.render(template_str, **kwargs)

    # Save if output_path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(result)

    return result


def create_project_structure(
    project_name: str,
    base_path: Optional[str] = None,
    include_examples: bool = True
) -> str:
    """
    Create a complete project structure for prompt management.

    Creates directories and example files for a new pyrompt project.

    Args:
        project_name: Name of the project
        base_path: Optional base path
        include_examples: Whether to include example prompts

    Returns:
        Path to created project directory

    Example:
        >>> path = create_project_structure('my_ai_app')
        >>> print(f"Project created at: {path}")
    """
    from pyrompt import PromptCollection, TemplateCollection

    # Create collections
    prompts = PromptCollection(project_name, base_path=base_path, with_metadata=True)
    templates = TemplateCollection(project_name, base_path=base_path, with_metadata=True)

    if include_examples:
        # Add example prompts
        prompts['system_default'] = "You are a helpful AI assistant."
        prompts.meta['system_default'] = {
            'author': 'pyrompt',
            'description': 'Default system prompt',
            'tags': ['system', 'default']
        }

        # Add example templates
        templates['greeting.txt'] = "Hello {name}, welcome to {project}!"
        templates.meta['greeting.txt'] = {
            'author': 'pyrompt',
            'description': 'Simple greeting template',
            'parameters': ['name', 'project']
        }

    return str(prompts.collection_dir.parent)


def get_stats(collection) -> Dict[str, Any]:
    """
    Get statistics about a collection.

    Args:
        collection: PromptCollection or TemplateCollection

    Returns:
        Dict with statistics

    Example:
        >>> prompts = PromptCollection('my_project')
        >>> stats = get_stats(prompts)
        >>> print(stats['total_prompts'])
    """
    stats = {
        'total_prompts': len(collection),
        'has_metadata': hasattr(collection, 'meta') and collection.meta is not None,
        'keys': list(collection.keys()),
    }

    if stats['has_metadata']:
        stats['prompts_with_metadata'] = len(collection.meta)

    # Calculate sizes
    total_chars = sum(len(collection[k]) for k in collection)
    stats['total_characters'] = total_chars
    stats['average_length'] = total_chars / len(collection) if len(collection) > 0 else 0

    return stats


__all__ = [
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
