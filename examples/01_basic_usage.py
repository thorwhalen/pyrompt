"""
Basic pyrompt usage example.

Shows how to:
- Create collections
- Add prompts and templates
- Render templates
- Use metadata
"""

from pyrompt import PromptCollection, TemplateCollection

# Create collections
prompts = PromptCollection('basic_example')
templates = TemplateCollection('basic_example')

# Add simple prompts
prompts['system'] = "You are a helpful AI assistant specialized in Python."
prompts['creative'] = "You are a creative writing assistant."

# Add templates
templates['greeting.txt'] = "Hello {name}, welcome to {place}!"
templates['question.txt'] = "Please answer this question: {question}\n\nProvide details."

# Render templates
print("=== Rendered Templates ===\n")
result = templates.render('greeting.txt', name='Alice', place='Python Land')
print(f"Greeting: {result}\n")

result = templates.render('question.txt', question='What is the meaning of life?')
print(f"Question:\n{result}\n")

# Parse template to see what parameters it needs
print("=== Template Parameters ===\n")
info = templates.parse('greeting.txt')
print(f"Parameters for 'greeting.txt': {info['placeholders']}\n")

# List all prompts
print("=== All Prompts ===\n")
for key in prompts:
    print(f"{key}: {prompts[key][:50]}...")

print("\n✓ Basic example complete!")
