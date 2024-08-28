import yaml
import json

input_file = 'D:/PYTHON/POC/config2.yaml'  # Path to your YAML file
output_file = 'D:/PYTHON/POC/test.json'  # Path to save the converted JSON file

# Load the YAML file
with open(input_file, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# Load the JSON file
with open(output_file, 'r') as json_file:
    json_data = json.load(json_file)

# Override JSON values with YAML values
for env, values in yaml_data.items():
    if env in json_data:
        json_data[env].update(values)

# Save the updated JSON data back to the file
with open(output_file, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("JSON file updated successfully.")
