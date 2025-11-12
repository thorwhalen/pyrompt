"""Pytest fixtures for pyrompt tests."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return {
        'greeting': 'Hello, {name}!',
        'farewell': 'Goodbye, {name}. See you {when}!',
        'system': 'You are a helpful assistant.',
        'python_expert': 'You are a Python programming expert.',
        'data_analyst': 'You specialize in data analysis.',
    }


@pytest.fixture
def sample_templates():
    """Sample templates with various engines."""
    return {
        'simple.txt': 'Hello {name}',
        'advanced.jinja2': 'Hello {{ name|default("friend") }}',
        'logic.mustache': 'Hello {{name}}',
    }
