"""Tests for storage layer."""

import pytest
from pyrompt.stores import (
    mk_prompt_store,
    mk_template_store,
    mk_metadata_store,
    get_default_base_path,
)


def test_prompt_store_basic_operations(temp_dir, sample_prompts):
    """Test basic CRUD operations on prompt store."""
    store = mk_prompt_store(temp_dir)

    # Create
    store['greeting'] = sample_prompts['greeting']

    # Read
    assert store['greeting'] == sample_prompts['greeting']

    # Update
    store['greeting'] = 'Hi {name}!'
    assert store['greeting'] == 'Hi {name}!'

    # Delete
    del store['greeting']
    assert 'greeting' not in store


def test_prompt_store_iteration(temp_dir, sample_prompts):
    """Test iteration over prompt store."""
    store = mk_prompt_store(temp_dir)

    # Add multiple prompts
    for key, value in sample_prompts.items():
        store[key] = value

    # Test iteration
    keys = list(store)
    assert len(keys) == len(sample_prompts)
    assert set(keys) == set(sample_prompts.keys())


def test_template_store_multiple_extensions(temp_dir):
    """Test template store with multiple extensions."""
    store = mk_template_store(temp_dir)

    # Store templates with different extensions
    store['simple.txt'] = 'Hello {name}'
    store['jinja.jinja2'] = 'Hello {{ name }}'
    store['mustache.mustache'] = 'Hello {{name}}'

    # All should be accessible
    assert 'simple.txt' in store
    assert 'jinja.jinja2' in store
    assert 'mustache.mustache' in store


def test_metadata_store(temp_dir):
    """Test JSON metadata store."""
    store = mk_metadata_store(temp_dir)

    # Store metadata
    store['prompt1'] = {
        'author': 'thor',
        'version': '1.0',
        'tags': ['system', 'friendly']
    }

    # Retrieve metadata
    meta = store['prompt1']
    assert meta['author'] == 'thor'
    assert meta['version'] == '1.0'
    assert 'system' in meta['tags']


def test_default_base_path():
    """Test default base path generation."""
    path = get_default_base_path()
    assert path is not None
    assert isinstance(path, str)
    assert 'pyrompt' in path.lower()
