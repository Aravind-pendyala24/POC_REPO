#!/bin/bash

#Run in debug mode
set -x

# Define the YAML file path
yaml_file="/home/aravind/test/config1.yaml"

# Read the YAML file line by line
while IFS= read -r line; do
  # Remove leading and trailing spaces
  line=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')

  # Check if the line contains an app version (key-value pair without '-code')
  if [[ $line =~ ^[a-z-]+:[[:space:]]*[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    app_version=$(echo "$line" | awk -F': ' '{print $2}')

    # Derive the corresponding code key
    code_key=$(echo "$line" | awk -F': ' '{print $1"-code"}')
    
    # Find the corresponding code value
    app_code=$(grep "$code_key" "$yaml_file" | awk -F': ' '{print $2}')

    # Execute the other script with app version and code as arguments
    sh /home/aravind/test/script.sh "$app_version" "$app_code"
  fi
done < <(grep -v '^uat:' "$yaml_file")

#this is working
