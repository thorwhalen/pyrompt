"""Tests for template engines."""

import pytest
from pyrompt.engines import (
    get_engine,
    list_engines,
    detect_engine,
)
from pyrompt.engines.format_engine import FormatEngine


def test_format_engine_basic():
    """Test basic format engine operations."""
    engine = FormatEngine()

    # Test rendering
    template = "Hello {name}!"
    result = engine.render(template, name='Alice')
    assert result == "Hello Alice!"


def test_format_engine_parsing():
    """Test format engine template parsing."""
    engine = FormatEngine()

    template = "Hello {name}, you are {age} years old!"
    parsed = engine.parse_template(template)

    assert 'placeholders' in parsed
    assert set(parsed['placeholders']) == {'name', 'age'}


def test_format_engine_detection():
    """Test format engine content detection."""
    engine = FormatEngine()

    # Should detect format strings
    assert engine.detect("Hello {name}!")
    assert engine.detect("{x} + {y} = {z}")

    # Should not detect literal braces
    assert not engine.detect("Hello {{name}}!")


def test_get_engine():
    """Test getting engines by name."""
    engine = get_engine('format')
    assert engine is not None
    assert engine.name == 'format'


def test_list_engines():
    """Test listing available engines."""
    engines = list_engines()
    assert 'format' in engines


def test_detect_engine_by_content():
    """Test engine detection by content."""
    # Format strings
    engine = detect_engine("Hello {name}!")
    assert engine.name == 'format'

    # Try to detect Jinja2 (if available)
    try:
        import jinja2
        engine = detect_engine("Hello {{ name }}!")
        assert engine.name == 'jinja2'
    except ImportError:
        pass  # Jinja2 not available


def test_jinja2_engine_if_available():
    """Test Jinja2 engine if available."""
    pytest.importorskip("jinja2")

    from pyrompt.engines.jinja2_engine import Jinja2Engine

    engine = Jinja2Engine()

    # Test rendering
    template = "Hello {{ name }}!"
    result = engine.render(template, name='Bob')
    assert result == "Hello Bob!"

    # Test with default filter
    template = "Hello {{ name|default('friend') }}!"
    result = engine.render(template)
    assert result == "Hello friend!"


def test_mustache_engine_if_available():
    """Test Mustache engine if available."""
    pytest.importorskip("pystache")

    from pyrompt.engines.mustache_engine import MustacheEngine

    engine = MustacheEngine()

    # Test rendering
    template = "Hello {{name}}!"
    result = engine.render(template, name='Charlie')
    assert result == "Hello Charlie!"

    # Test sections
    template = "{{#premium}}Premium user{{/premium}}"
    result = engine.render(template, premium=True)
    assert "Premium user" in result
