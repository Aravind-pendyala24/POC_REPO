#!/bin/bash

# Path to the YAML file
yaml_file="/home/aravind/test/test.yaml"

# Iterate over each component in the YAML
components=$(yq e '.uat | keys' $yaml_file)

# Convert the yq output into an array
components_array=($(echo "$components" | tr -d '[],'))

# Iterate through each component
for component in "${components_array[@]}"; do
    tag=$(yq e ".uat.${component}.tag" $yaml_file)
    code=$(yq e ".uat.${component}.code" $yaml_file)

    # Execute the shell script with tag and code as arguments
    ./home/aravind/test/script.sh "$tag" "$code"
done
