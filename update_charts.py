import re

# Read the HTML file
with open('dashboard_example.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all instances of "responsive: true," with "responsive: true,\n                maintainAspectRatio: false,"
# but only if maintainAspectRatio is not already present
pattern = r'(\s+)responsive: true,(?!\s*maintainAspectRatio)'
replacement = r'\1responsive: true,\1maintainAspectRatio: false,'

updated_content = re.sub(pattern, replacement, content)

# Write the updated content back
with open('dashboard_example.html', 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Updated all charts to be more responsive and flexible!")