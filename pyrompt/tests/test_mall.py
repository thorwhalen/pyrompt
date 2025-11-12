"""Tests for PromptMall."""

import pytest
from pyrompt import PromptMall


def test_prompt_mall_basic(temp_dir):
    """Test basic PromptMall operations."""
    mall = PromptMall('workspace', base_path=temp_dir)

    # Add collection implicitly
    mall['system']['python'] = "You are a Python expert."

    # Check it exists
    assert 'system' in mall
    assert 'python' in mall['system']

    # Add another collection
    mall['templates']['greeting'] = "Hello {name}!"

    # Check both exist
    assert len(mall) == 2
    assert 'system' in mall.keys()
    assert 'templates' in mall.keys()


def test_prompt_mall_explicit_add(temp_dir):
    """Test explicitly adding collections to mall."""
    mall = PromptMall('workspace', base_path=temp_dir)

    # Add collection explicitly
    mall.add_collection('personas', 'prompt')
    mall['personas']['analyst'] = "You are a data analyst."

    assert 'personas' in mall
    assert 'analyst' in mall['personas']


def test_prompt_mall_repr(temp_dir):
    """Test __repr__ for PromptMall."""
    mall = PromptMall('workspace', base_path=temp_dir)
    mall['system']['test'] = "Test"

    repr_str = repr(mall)
    assert 'PromptMall' in repr_str
    assert 'workspace' in repr_str


def test_prompt_mall_search_mock(temp_dir):
    """Test that mall.search exists."""
    mall = PromptMall('workspace', base_path=temp_dir)

    # Just check the method exists
    assert hasattr(mall, 'search')
