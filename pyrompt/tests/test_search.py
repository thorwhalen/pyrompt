"""Tests for semantic search functionality."""

import pytest


def test_semantic_index_basic():
    """Test basic semantic index operations."""
    pytest.importorskip("numpy")
    pytest.importorskip("oa")

    from pyrompt import PromptCollection, SemanticIndex

    # This test would require actual API calls, so we skip it in CI
    pytest.skip("Requires OpenAI API access")


def test_semantic_index_structure():
    """Test that SemanticIndex has the expected interface."""
    pytest.importorskip("numpy")
    pytest.importorskip("oa")

    from pyrompt.search import SemanticIndex

    # Check that the class exists and has expected methods
    assert hasattr(SemanticIndex, 'search')
    assert hasattr(SemanticIndex, 'rebuild')
    assert hasattr(SemanticIndex, 'add')
    assert hasattr(SemanticIndex, 'remove')
