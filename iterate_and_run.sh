#!/bin/bash

# Path to the YAML file
yaml_file="/home/aravind/test/test.yaml"

# Iterate over each component in the YAML
yq eval '.uat | to_entries[] | .key' $yaml_file | while read component; do
    tag=$(yq eval ".uat.${component}.tag" $yaml_file)
    code=$(yq eval ".uat.${component}.code" $yaml_file)

    # Execute the shell script with tag and code as arguments
    ./home/aravind/test/script.sh "$tag" "$code"
done
