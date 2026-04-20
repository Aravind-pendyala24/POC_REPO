#!/bin/bash

set -e

BASE_BRANCH=$1
CHANGELOG_FILE=${2:-CHANGELOG.md}
VERSION_FILE=${3:-version.json}

echo "🔍 Validating changelog between versions..."

# Fetch base branch
git fetch origin "$BASE_BRANCH"

# Read versions
CURRENT=$(jq -r '.current_version' "$VERSION_FILE")
PREVIOUS=$(jq -r '.previous_version' "$VERSION_FILE")

echo "Current version: $CURRENT"
echo "Previous version: $PREVIOUS"

# Ensure CHANGELOG.md changed
CHANGED_FILES=$(git diff --name-only origin/$BASE_BRANCH...HEAD)

echo "$CHANGED_FILES" | grep -q "$CHANGELOG_FILE" || {
  echo "❌ ERROR: $CHANGELOG_FILE not updated in PR"
  exit 1
}

echo "✅ CHANGELOG.md modified"

# Get diff of changelog
DIFF=$(git diff origin/$BASE_BRANCH...HEAD "$CHANGELOG_FILE")

# Extract added lines
ADDED=$(echo "$DIFF" | grep "^+" | grep -v "+++")

echo "🆕 Added lines:"
echo "$ADDED"

echo "$ADDED" | grep -E "^\+\- " > /dev/null || {
  echo "❌ ERROR: No valid changelog entries added"
  exit 1
}

# Validate placement between versions
echo "🔎 Checking placement between versions..."

awk -v curr="## Release $CURRENT" -v prev="## Release $PREVIOUS" '
BEGIN {in_section=0; valid=0}

$0 ~ curr {in_section=1; next}
$0 ~ prev {in_section=0}

in_section && /^- / {
  valid=1
}

END {
  if (valid==0) {
    print "❌ ERROR: No entries found between current and previous versions"
    exit 1
  } else {
    print "✅ Valid changelog entries found between versions"
  }
}
' "$CHANGELOG_FILE"

echo "🎉 Changelog validation passed"
