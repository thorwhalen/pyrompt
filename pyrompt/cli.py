"""
Command-line interface for pyrompt.

Simple CLI for managing prompt collections.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional


def cmd_new(args):
    """Create a new collection."""
    from pyrompt.util import create_project_structure

    path = create_project_structure(
        args.name,
        base_path=args.path,
        include_examples=not args.no_examples
    )
    print(f"✓ Created project '{args.name}' at: {path}")


def cmd_list(args):
    """List prompts in a collection."""
    from pyrompt import PromptCollection, TemplateCollection

    if args.templates:
        coll = TemplateCollection(args.name, base_path=args.path)
        print(f"Templates in '{args.name}':")
    else:
        coll = PromptCollection(args.name, base_path=args.path)
        print(f"Prompts in '{args.name}':")

    if len(coll) == 0:
        print("  (empty)")
    else:
        for key in sorted(coll.keys()):
            preview = coll[key][:60].replace('\n', ' ')
            if len(coll[key]) > 60:
                preview += '...'
            print(f"  {key}: {preview}")


def cmd_show(args):
    """Show a specific prompt."""
    from pyrompt import PromptCollection, TemplateCollection

    if args.templates:
        coll = TemplateCollection(args.name, base_path=args.path)
    else:
        coll = PromptCollection(args.name, base_path=args.path)

    if args.key not in coll:
        print(f"Error: '{args.key}' not found in collection '{args.name}'")
        sys.exit(1)

    print(f"=== {args.key} ===")
    print(coll[args.key])

    if args.metadata and hasattr(coll, 'meta') and coll.meta and args.key in coll.meta:
        print(f"\n=== Metadata ===")
        import json
        print(json.dumps(coll.meta[args.key], indent=2))


def cmd_add(args):
    """Add a prompt to a collection."""
    from pyrompt import PromptCollection, TemplateCollection

    if args.templates:
        coll = TemplateCollection(args.name, base_path=args.path)
    else:
        coll = PromptCollection(args.name, base_path=args.path)

    # Read content from stdin or argument
    if args.content == '-':
        content = sys.stdin.read()
    elif args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    else:
        content = args.content

    coll[args.key] = content
    print(f"✓ Added '{args.key}' to collection '{args.name}'")


def cmd_render(args):
    """Render a template."""
    from pyrompt import TemplateCollection
    import json

    coll = TemplateCollection(args.name, base_path=args.path)

    if args.key not in coll:
        print(f"Error: Template '{args.key}' not found")
        sys.exit(1)

    # Parse parameters
    params = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError:
            # Try as key=value pairs
            for pair in args.params.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    params[k.strip()] = v.strip()

    # Render
    result = coll.render(args.key, **params)
    print(result)


def cmd_stats(args):
    """Show collection statistics."""
    from pyrompt import PromptCollection
    from pyrompt.util import get_stats

    coll = PromptCollection(args.name, base_path=args.path)
    stats = get_stats(coll)

    print(f"Collection: {args.name}")
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Average length: {stats['average_length']:.1f} chars")
    if stats['has_metadata']:
        print(f"  With metadata: {stats['prompts_with_metadata']}")


def cmd_engines(args):
    """List available template engines."""
    from pyrompt.util import list_available_engines

    engines = list_available_engines()
    print("Available template engines:")
    for name, info in engines.items():
        exts = ', '.join(info['extensions']) if info['extensions'] else '(none)'
        print(f"  {name}: {exts}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='pyrompt - Manage AI prompts and templates',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--path',
        help='Base path for collections (overrides default)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # new command
    new_parser = subparsers.add_parser('new', help='Create a new collection')
    new_parser.add_argument('name', help='Collection name')
    new_parser.add_argument('--no-examples', action='store_true',
                           help='Do not include example prompts')
    new_parser.set_defaults(func=cmd_new)

    # list command
    list_parser = subparsers.add_parser('list', help='List prompts in a collection')
    list_parser.add_argument('name', help='Collection name')
    list_parser.add_argument('--templates', action='store_true',
                            help='List templates instead of prompts')
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser('show', help='Show a specific prompt')
    show_parser.add_argument('name', help='Collection name')
    show_parser.add_argument('key', help='Prompt key')
    show_parser.add_argument('--templates', action='store_true',
                            help='Show from templates')
    show_parser.add_argument('--metadata', action='store_true',
                            help='Show metadata')
    show_parser.set_defaults(func=cmd_show)

    # add command
    add_parser = subparsers.add_parser('add', help='Add a prompt')
    add_parser.add_argument('name', help='Collection name')
    add_parser.add_argument('key', help='Prompt key')
    add_parser.add_argument('content', nargs='?', default='-',
                           help='Prompt content (use - for stdin)')
    add_parser.add_argument('--file', help='Read from file')
    add_parser.add_argument('--templates', action='store_true',
                           help='Add to templates')
    add_parser.set_defaults(func=cmd_add)

    # render command
    render_parser = subparsers.add_parser('render', help='Render a template')
    render_parser.add_argument('name', help='Collection name')
    render_parser.add_argument('key', help='Template key')
    render_parser.add_argument('--params', help='JSON params or key=value pairs')
    render_parser.set_defaults(func=cmd_render)

    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    stats_parser.add_argument('name', help='Collection name')
    stats_parser.set_defaults(func=cmd_stats)

    # engines command
    engines_parser = subparsers.add_parser('engines', help='List available engines')
    engines_parser.set_defaults(func=cmd_engines)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n✗ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Error: {e}")
        if '--debug' in sys.argv:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
