#!/bin/bash

# This script runs pre-commit hooks only on staged files
# Usage: ./scripts/run-pre-commit-staged.sh

# Get list of staged files
STAGED_FILES=$(git diff --name-only --cached)

if [ -z "$STAGED_FILES" ]; then
  echo "No files are staged for commit."
  exit 0
fi

# Run pre-commit on only the staged files
echo "$STAGED_FILES" | xargs pre-commit run --files
