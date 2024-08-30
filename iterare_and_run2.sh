#!/bin/bash

# Define the YAML file path
yaml_file="D:/PYTHON\POC"

# Use awk to parse the yaml file
awk '
/^[[:space:]]*[^[:space:]:]+:/ { key=$1; sub(":$", "", key) }
/^[[:space:]]+tag:/ { tag=$2 }
/^[[:space:]]+portalcode:/ { portalcode=$2; print key, tag, portalcode }
' "$yaml_file" | while read -r app tag portalcode; do
  # Pass the tag and portalcode as arguments to another script
  ./another_script.sh "$tag" "$portalcode"
done


#this is working