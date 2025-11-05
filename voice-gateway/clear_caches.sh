#!/usr/bin/env bash

# Clears common build and interpreter caches within the repository.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

remove_path() {
    local relative_path="$1"
    local absolute_path="${ROOT_DIR}/${relative_path}"

    if [[ -e "${absolute_path}" ]]; then
        echo "Removing ${relative_path}"
        rm -rf "${absolute_path}"
    fi
}

echo "Clearing caches under ${ROOT_DIR}"

# Python bytecode caches
while IFS= read -r dir; do
    rel_path="${dir#"${ROOT_DIR}/"}"
    echo "Removing ${rel_path}"
    rm -rf "${dir}"
done < <(find "${ROOT_DIR}" -type d -name "__pycache__")

# Stale bytecode files
find "${ROOT_DIR}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -print -delete

# Tool-specific caches
remove_path ".pytest_cache"
remove_path ".mypy_cache"
remove_path ".ruff_cache"
remove_path ".cache"

# Build artefacts
remove_path "build"
remove_path "dist"

echo "Cache cleanup complete."
