#!/bin/bash

#set +x

new_tag_txt=$1
portal_code=$2
file_name=/poc/scripts/xml/list.xml

export TZ="America/Chicago"

formatted_date=$(date "+%m/%d/%Y %H:%M %Z")

#lock_file="/poc/scripts/xml/list.lock"

flock -n "$lock_file" -c "
       awk -i inplace -v portal_code=\"${portal_code}\" -v new_tag_txt=\"${new_tag_txt}\" '{if (\$0 ~ portal_code) gsub(/tag=\"[^\"]*\"/, \"tag=\\\"\"new_tag_txt \"\\\"\"); print}' \"${file_name}\"

       awk -i inplace -v portal_code=\"${portal_code}\" -v date_text=\"${formatted_date}\" '{if (\$0 ~ portal_code) gsub(/ldeploy=\"[^\"]*\"/, \"ldeploy=\\\"\"date_text \"\\\"\"); print}' \"${file_name}\"
"

#rm -f "$lock_file"
