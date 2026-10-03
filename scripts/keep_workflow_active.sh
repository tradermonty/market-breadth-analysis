#!/usr/bin/env bash
# GitHub disables public scheduled workflows after 60 days without activity.
set -euo pipefail

default_branch="${1:?Usage: keep_workflow_active.sh DEFAULT_BRANCH}"
git fetch origin "$default_branch"
git checkout -B "$default_branch" FETCH_HEAD

last_commit=$(git log -1 --format=%ct)
now=$(date +%s)
if (( now - last_commit < 30 * 86400 )); then
    echo "Recent repository activity; no keepalive needed."
    exit 0
fi

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
# Commit the existing tree only, never generated reports or staged files.
tree=$(git rev-parse 'HEAD^{tree}')
commit=$(git commit-tree "$tree" -p HEAD -m 'Keep scheduled market breadth workflow active')
# A concurrent default-branch update safely rejects this non-forced push.
git push origin "$commit:refs/heads/$default_branch"
