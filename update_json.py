import yaml
import json

YAML_FILE = 'config2.yaml'
JSON_FILE= 'test.json'

# Read the full YAML file
with open( YAML_FILE, 'r') as yaml_file:
    yaml_content = yaml.safe_load(yaml_file)

# Extract the 'uat' section
uat_content = yaml_content.get('uat', {})

# Read the existing JSON content
with open( JSON_FILE, 'r') as json_file:
    json_content = json.load(json_file)

# Update the JSON content with the extracted 'uat' section
json_content.update(uat_content)

# Write the updated content back to the JSON file
with open(JSON_FILE, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)
