#!/bin/bash

# Define the YAML file path
yaml_file="config.yaml"

# Initialize variables to store the current app and its code
app=""
code=""

# Read the YAML file line by line
while IFS= read -r line; do
  # Remove leading and trailing spaces
  line=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')

  # Check if the line contains an app version
  if [[ $line =~ ^[a-z_]+:[[:space:]]*[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    app=$(echo "$line" | awk -F': ' '{print $1}')
    app_version=$(echo "$line" | awk -F': ' '{print $2}')

  # Check if the line contains a code corresponding to the last app version found
  elif [[ $line =~ ^[a-z_]+code:[[:space:]]*[A-Z]+ ]]; then
    code=$(echo "$line" | awk -F': ' '{print $2}')

    # Execute the other script with app version and code as arguments
    ./other_script.sh "$app_version" "$code"
  fi
done < <(grep -v '^uat:' "$yaml_file")

#this is working
