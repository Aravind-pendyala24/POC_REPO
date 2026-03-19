# Copilot agent instructions

## What this repository is
Terraform infrastructure code. Modules are not written here — they are 
consumed from our internal JFrog Artifactory Terraform registry.

## How modules are declared in .tf files
Every externally-sourced module follows this exact pattern:

    module "<label>" {
      source  = "artifactory.example.com/terraform-modules__local/<module-name>/generic"
      version = "<semver>"
    }

Only update the `version =` line. Never touch `source =` or any other field.

---

## How to find all modules that need updating

### Step 1 — Find every .tf file using an Artifactory source
Run this in the repo root:
    grep -rl "artifactory.example.com" --include="*.tf" .

### Step 2 — Extract module name and current version from each file
For each file returned, extract blocks matching:
    grep -A 5 'source.*artifactory' <file>

The module name is the third path segment in the source URL.
For example: `terraform-modules__local/networking/generic` → module name is `networking`

---

## How to fetch the latest version from Artifactory

Run this curl command (the token is available as the env var ARTIFACTORY_TOKEN,
the base URL as ARTIFACTORY_BASE_URL):

    curl -s \
      -H "X-JFrog-Art-Api: $ARTIFACTORY_TOKEN" \
      "$ARTIFACTORY_BASE_URL/api/storage/terraform-modules/<module-name>/"

The response is JSON. The `children` array contains objects like:
    { "uri": "/1.2.0", "folder": true }

Collect all entries where `folder` is true. Strip the leading `/` from each 
`uri`. Sort them as semantic versions (major.minor.patch). The last one after 
sorting is the latest.

Example sort logic you can use in the shell:
    echo "1.2.0 1.10.0 1.9.0" | tr ' ' '\n' | sort -V | tail -1
    → 1.10.0

---

## How to read the Confluence module consumption guide

    curl -s \
      -H "Authorization: Bearer $CONFLUENCE_TOKEN" \
      "$CONFLUENCE_BASE_URL/rest/api/content/PAGE_ID_HERE?expand=body.storage"

Replace PAGE_ID_HERE with the actual page ID: 98765 (update this with your real ID).

Read the response for any upgrade notes specific to each module before deciding 
whether to apply an update. Look for sections matching the module name.

---

## Version update rules

- Patch bump (1.2.0 → 1.2.5): apply automatically.
- Minor bump (1.2.0 → 1.4.0): apply automatically.
- Major bump (1.x.x → 2.x.x): DO NOT update the file. Record it in the PR 
  body under a "Needs manual review" section with the current and latest 
  version noted.
- If the current version already equals the latest: skip silently.

---

## How to update the file

Use sed to replace the version string in-place. Only match the version line 
inside the correct module block to avoid false replacements:

    sed -i 's/version *= *"<old_version>"/version = "<new_version>"/' <filepath>

After editing, verify the change looks correct with:
    grep -A 6 'module "<label>"' <filepath>

---

## Pull request requirements

After all files are updated, create a PR with:

**Title:** `chore: update terraform module versions to latest`

**Body must include:**

1. A markdown table:
   | Module | File | Old version | New version |
   |--------|------|-------------|-------------|

2. A "Skipped — major version bumps" section (if any):
   List each module, current version, available version, and a note saying 
   manual review is required before applying.

3. Footer line:
   > Versions sourced from JFrog Artifactory. 
   > Confluence module guide reviewed before all changes.

**Labels to apply:** `terraform`, `automated`, `dependencies`
```

---

## Step 3 — Start the agent session from the Agents tab

Navigate to your repository, click the **Agents** tab, then **New session**. Paste this prompt — nothing more:
```
Scan this repository for all Terraform module declarations that reference 
our JFrog Artifactory registry. For each module found, check the latest 
available version using the Artifactory REST API instructions in 
copilot-instructions.md. Read the Confluence module guide for upgrade notes. 
Apply all eligible version updates directly to the .tf files following the 
rules in copilot-instructions.md. Then open a pull request.
```

That is the entire prompt. Everything else — which files to read, how to call the API, how to sort versions, what to put in the PR — is already in `copilot-instructions.md`. The prompt just says "go do the thing."

---

## Step 4 — What the agent actually does (what you'll see in the Agents tab)

The activity feed in the Agents tab will show these steps as the agent works through them:
```
Reading copilot-instructions.md...
Running: grep -rl "artifactory.example.com" --include="*.tf" .
  Found: infra/networking/main.tf
  Found: infra/compute/main.tf
  Found: infra/database/main.tf

Reading infra/networking/main.tf...
  Module: networking, current version: 1.1.0

Running: curl -s -H "X-JFrog-Art-Api: ***" .../terraform-modules/networking/
  Versions found: 1.1.0, 1.2.0, 1.3.0, 1.3.2
  Latest: 1.3.2 → update required

Fetching Confluence page 98765 for upgrade notes on networking...
  No breaking changes found for minor/patch range.

Running: sed -i 's/version = "1.1.0"/version = "1.3.2"/' infra/networking/main.tf
Verified change in infra/networking/main.tf ✓

... (repeats for compute, database)

Module: security-baseline, current: 1.4.0, latest: 2.0.1
  → Major version bump detected. Skipping. Will flag in PR.

Creating branch: copilot/tf-module-updates-20250320
Committing 3 changed files...
Opening pull request...
