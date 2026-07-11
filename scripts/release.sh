#!/usr/bin/env bash
# Release helper: verify the repo is release-ready, snapshot the environment,
# commit, and tag. Pushing the tag and creating the GitHub release are the
# explicit last step and can be skipped with --no-push.
#
# Usage: scripts/release.sh X.Y.Z [--no-push]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 X.Y.Z [--no-push]" >&2
    exit 2
fi

VERSION="$1"
NO_PUSH=0
if [[ "${2:-}" == "--no-push" ]]; then
    NO_PUSH=1
fi

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Version must be in X.Y.Z form (got: $VERSION)" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

echo "==> Checking working tree is clean"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit or stash changes before releasing." >&2
    exit 1
fi

echo "==> Checking __version__ was bumped to $VERSION"
CURRENT_VERSION="$(python -c "import nbody_pipeline; print(nbody_pipeline.__version__)")"
if [[ "$CURRENT_VERSION" != "$VERSION" ]]; then
    echo "nbody_pipeline.__version__ is '$CURRENT_VERSION', expected '$VERSION'." >&2
    echo "Bump it in nbody_pipeline/__init__.py before releasing." >&2
    exit 1
fi

echo "==> Checking CHANGELOG.md has a [$VERSION] section"
if ! grep -qE "^## \[$VERSION\]" CHANGELOG.md; then
    echo "CHANGELOG.md has no '## [$VERSION]' section." >&2
    echo "Move [Unreleased] entries into a dated [$VERSION] section first." >&2
    exit 1
fi

echo "==> Checking CITATION.cff version matches"
if ! grep -qE "^version: $VERSION$" CITATION.cff; then
    echo "CITATION.cff 'version:' does not match $VERSION." >&2
    exit 1
fi

echo "==> Refreshing requirements.lock"
python -m pip freeze --exclude-editable > requirements.lock

echo "==> Committing release artifacts"
git add requirements.lock
if [[ -n "$(git status --porcelain)" ]]; then
    git commit -m "chore: release v$VERSION"
else
    echo "Nothing to commit (requirements.lock unchanged)."
fi

echo "==> Tagging v$VERSION"
git tag -a "v$VERSION" -m "v$VERSION"

if [[ "$NO_PUSH" -eq 1 ]]; then
    echo "==> --no-push given: skipping tag push and GitHub release creation"
    echo "Release checks passed and v$VERSION tagged locally."
    exit 0
fi

echo "==> Pushing tag v$VERSION"
git push origin "v$VERSION"

echo "==> Creating GitHub release"
NOTES="$(awk "/^## \[$VERSION\]/{flag=1; next} /^## \[/{flag=0} flag" CHANGELOG.md)"
gh release create "v$VERSION" --title "v$VERSION" --notes "$NOTES"

echo "Released v$VERSION."
