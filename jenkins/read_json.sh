#!/bin/bash

# Define paths to YAML files and the JSON file
UAT_YAML="/home/aravind/test/POC2/uat.yaml"        # Update with actual path to uat.yaml
STAGING_YAML="/home/aravind/test/POC2/staging.yaml" # Update with actual path to staging.yaml
JSON_FILE="/home/aravind/test/POC2/test.json"     # Update with actual path to the JSON file

# Function to read YAML and extract values
read_yaml() {
  local yaml_file="$1"
  local key="$2"
  grep "^ *$key:" "$yaml_file" | sed -E "s/^ *$key: *(.*)/\1/"
}

# Update JSON file with values from YAML files
update_json() {
  # Load existing JSON file into a variable
  json_content=$(cat "$JSON_FILE")

  # Extract values from the YAML files
  uat_ac_disc=$(read_yaml "$UAT_YAML" "ac-disc")
  uat_ac_admin=$(read_yaml "$UAT_YAML" "ac-admin")
  uat_ar_user=$(read_yaml "$UAT_YAML" "ar-user")
  uat_as_admin=$(read_yaml "$UAT_YAML" "as-admin")
  staging_ac_disc=$(read_yaml "$STAGING_YAML" "ac-disc")
  staging_ac_admin=$(read_yaml "$STAGING_YAML" "ac-admin")
  staging_ar_user=$(read_yaml "$STAGING_YAML" "ar-user")
  staging_as_admin=$(read_yaml "$STAGING_YAML" "as-admin")

  # Update JSON content using `jq` (a command-line JSON processor)
  updated_json=$(echo "$json_content" | jq \
    --arg uat_ac_disc "$uat_ac_disc" \
    --arg uat_ac_admin "$uat_ac_admin" \
    --arg uat_ar_user "$uat_ar_user" \
    --arg uat_as_admin "$uat_as_admin" \
    --arg staging_ac_disc "$staging_ac_disc" \
    --arg staging_ac_admin "$staging_ac_admin" \
    --arg staging_ar_user "$staging_ar_user" \
    --arg staging_as_admin "$staging_as_admin" \
    '.uat."ac-disc" = $uat_ac_disc |
     .uat."ac-admin" = $uat_ac_admin |
     .uat."ar-user" = $uat_ar_user |
     .uat."as-admin" = $uat_as_admin |
     .staging."ac-disc" = $staging_ac_disc |
     .staging."ac-admin" = $staging_ac_admin |
     .staging."ar-user" = $staging_ar_user |
     .staging."as-admin" = $staging_as_admin')

  # Write updated JSON content back to the file
  echo "$updated_json" > "$JSON_FILE"
}

# Run the function to update the JSON file
update_json

# Commit and push changes if the JSON file is updated
#if [ -n "$(git status --porcelain "$JSON_FILE")" ]; then
 # git add "$JSON_FILE"
 # git commit -m "Update JSON file with latest YAML changes"
  #git push origin main  # Replace 'main' with the correct branch if necessary
#else
 # echo "No changes detected in the JSON file."
#fi
