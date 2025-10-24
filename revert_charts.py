import re

# Read the HTML file
with open('dashboard_example.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove maintainAspectRatio: false lines
content = re.sub(r'\s+maintainAspectRatio: false,\s*\n', '\n', content)

# Restore the Learning Effectiveness chart options to original
learning_chart_pattern = r'(options: \{[\s\S]*?)maxTicksLimit: 4,\s*font: \{\s*size: 11\s*\}([\s\S]*?\}[\s\S]*?\}\s*\}\s*\})'
learning_chart_replacement = r'\1\2'
content = re.sub(learning_chart_pattern, learning_chart_replacement, content)

# Restore Content Quality radar chart options
radar_pattern = r'(options: \{[\s\S]*?)font: \{\s*size: 10\s*\}[\s\S]*?pointLabels: \{[\s\S]*?font: \{\s*size: 11\s*\}[\s\S]*?\}[\s\S]*?labels: \{[\s\S]*?font: \{\s*size: 12\s*\}[\s\S]*?\}([\s\S]*?\}[\s\S]*?\})'
radar_replacement = r'\1\2'
content = re.sub(radar_pattern, radar_replacement, content)

# Write the updated content back
with open('dashboard_example.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Reverted all responsive chart changes!")