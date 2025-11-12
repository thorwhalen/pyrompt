"""Tests for PromptCollection and TemplateCollection."""

import pytest
from pyrompt import PromptCollection, TemplateCollection


def test_prompt_collection_basic(temp_dir, sample_prompts):
    """Test basic PromptCollection operations."""
    coll = PromptCollection('test', base_path=temp_dir)

    # Create
    coll['greeting'] = sample_prompts['greeting']

    # Read
    assert coll['greeting'] == sample_prompts['greeting']

    # Update
    coll['greeting'] = 'Hi {name}!'
    assert coll['greeting'] == 'Hi {name}!'

    # Delete
    del coll['greeting']
    assert 'greeting' not in coll

    # Test len
    coll['system'] = sample_prompts['system']
    assert len(coll) == 1


def test_prompt_collection_with_metadata(temp_dir):
    """Test PromptCollection with metadata."""
    coll = PromptCollection('test', base_path=temp_dir, with_metadata=True)

    # Add prompt
    coll['greeting'] = "Hello, {name}!"

    # Add metadata
    coll.meta['greeting'] = {
        'author': 'thor',
        'version': '1.0',
        'tags': ['friendly', 'greeting']
    }

    # Retrieve metadata
    meta = coll.meta['greeting']
    assert meta['author'] == 'thor'
    assert 'friendly' in meta['tags']


def test_template_collection_basic(temp_dir):
    """Test basic TemplateCollection operations."""
    coll = TemplateCollection('test', base_path=temp_dir)

    # Add template
    coll['greeting.txt'] = "Hello {name}!"

    # Read
    assert coll['greeting.txt'] == "Hello {name}!"

    # Render
    result = coll.render('greeting.txt', name='Alice')
    assert result == "Hello Alice!"


def test_template_collection_parse(temp_dir):
    """Test template parsing."""
    coll = TemplateCollection('test', base_path=temp_dir)

    coll['greeting'] = "Hello {name}, welcome to {place}!"

    parsed = coll.parse('greeting')
    assert 'placeholders' in parsed
    assert set(parsed['placeholders']) == {'name', 'place'}


def test_template_collection_jinja2_if_available(temp_dir):
    """Test TemplateCollection with Jinja2 templates."""
    pytest.importorskip("jinja2")

    coll = TemplateCollection('test', base_path=temp_dir)

    # Add Jinja2 template
    coll['greeting.jinja2'] = "Hello {{ name|default('friend') }}!"

    # Render with value
    result = coll.render('greeting.jinja2', name='Bob')
    assert result == "Hello Bob!"

    # Render without value (uses default)
    result = coll.render('greeting.jinja2')
    assert result == "Hello friend!"


def test_template_collection_to_prompt_function_mock(temp_dir):
    """Test converting template to function (mocked)."""
    coll = TemplateCollection('test', base_path=temp_dir)
    coll['greet'] = "Hello {name}!"

    # Note: Would need oa installed to actually test this
    # Here we just test that the method exists
    assert hasattr(coll, 'to_prompt_function')
    assert hasattr(coll, 'to_prompt_json_function')
    assert hasattr(coll, 'create_prompt_functions')


def test_prompt_collection_repr(temp_dir):
    """Test __repr__ for PromptCollection."""
    coll = PromptCollection('test', base_path=temp_dir)
    coll['prompt1'] = "Test prompt"

    repr_str = repr(coll)
    assert 'PromptCollection' in repr_str
    assert 'test' in repr_str


def test_template_collection_repr(temp_dir):
    """Test __repr__ for TemplateCollection."""
    coll = TemplateCollection('test', base_path=temp_dir)
    coll['template1.txt'] = "Test template"

    repr_str = repr(coll)
    assert 'TemplateCollection' in repr_str
    assert 'test' in repr_str
