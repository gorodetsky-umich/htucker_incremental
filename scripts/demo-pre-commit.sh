#!/bin/bash

# This script demonstrates the difference between running pre-commit on all files
# versus running it only on staged files
# Usage: ./scripts/demo-pre-commit.sh

echo "==== DEMONSTRATION OF PRE-COMMIT BEHAVIOR ===="
echo ""
echo "1. Running pre-commit on ALL files:"
echo "--------------------------------------"
pre-commit run

echo ""
echo "2. Running pre-commit on STAGED files ONLY:"
echo "-------------------------------------------"
./scripts/run-pre-commit-staged.sh

echo ""
echo "==== EXPLANATION ===="
echo "The first command runs pre-commit on all files that match hook patterns,"
echo "regardless of whether they're staged for commit."
echo ""
echo "The second command runs pre-commit only on files that are staged for commit,"
echo "ignoring unstaged changes, which is often what you want."
