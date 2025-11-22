"""
Utilities and helpers example.

Shows how to use pyrompt utility functions for common tasks.
"""

from pyrompt import PromptCollection
from pyrompt.util import (
    quick_setup,
    import_from_dict,
    export_to_dict,
    validate_template,
    merge_collections,
    get_stats,
    list_available_engines,
)

print("=== Quick Setup ===\n")
# Quick setup creates both prompts and templates
collections = quick_setup('util_example')
print(f"Created collections: {list(collections.keys())}\n")

print("=== Import from Dictionary ===\n")
# Bulk import from dict
prompts_data = {
    'python': "You are a Python expert.",
    'javascript': "You are a JavaScript expert.",
    'rust': "You are a Rust expert.",
}
import_from_dict(collections['prompts'], prompts_data)
print(f"Imported {len(prompts_data)} prompts\n")

print("=== Template Validation ===\n")
# Validate a template
result = validate_template(
    "Hello {name}, you are {age} years old!",
    engine_name='format',
    required_params=['name', 'age']
)
print(f"Valid: {result['valid']}")
print(f"Placeholders: {result['placeholders']}")
print()

# Validate with missing params
result = validate_template(
    "Hello {name}!",
    required_params=['name', 'age']  # age is missing
)
print(f"Valid (missing param): {result['valid']}")
print(f"Errors: {result['errors']}\n")

print("=== Collection Statistics ===\n")
stats = get_stats(collections['prompts'])
print(f"Total prompts: {stats['total_prompts']}")
print(f"Total characters: {stats['total_characters']}")
print(f"Average length: {stats['average_length']:.1f} chars\n")

print("=== Available Engines ===\n")
engines = list_available_engines()
for name, info in engines.items():
    print(f"{name}: {info['extensions']}")
print()

print("=== Export to Dictionary ===\n")
data = export_to_dict(collections['prompts'], include_metadata=False)
print(f"Exported {len(data['prompts'])} prompts")
print(f"Keys: {list(data['prompts'].keys())}\n")

print("=== Merge Collections ===\n")
# Create two source collections
source1 = PromptCollection('merge_source1')
source1['prompt1'] = "First prompt"
source1['prompt2'] = "Second prompt"

source2 = PromptCollection('merge_source2')
source2['prompt3'] = "Third prompt"
source2['prompt4'] = "Fourth prompt"

# Merge into target
target = PromptCollection('merge_target')
merge_collections(target, source1, source2)
print(f"Merged collections: {len(target)} total prompts")
print(f"Keys: {list(target.keys())}\n")

print("✓ Utilities example complete!")
