import yaml
import json

# Define the input and output file paths
input_file = 'D:/PYTHON/POC/config2.yaml'  # Path to your YAML file
output_file = 'D:/PYTHON/POC/output.json'  # Path to save the converted JSON file

# Read the YAML file
with open(input_file, 'r') as yaml_file:
    yaml_content = yaml.safe_load(yaml_file)

# Convert YAML content to JSON and write it to a file
with open(output_file, 'w') as json_file:
    json.dump(yaml_content, json_file, indent=4)

print(f"Converted YAML to JSON and saved as {output_file}")