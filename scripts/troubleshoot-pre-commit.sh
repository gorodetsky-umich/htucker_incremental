#!/bin/bash
# This script helps troubleshoot and fix common pre-commit issues
# Usage: ./scripts/troubleshoot-pre-commit.sh

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pre-commit Troubleshooting Tool ===${NC}"
echo -e "${BLUE}This script will help diagnose and fix common pre-commit issues${NC}"
echo ""

# Check if pre-commit is installed
echo -e "${YELLOW}Checking if pre-commit is installed...${NC}"
if ! command -v pre-commit &> /dev/null; then
    echo -e "${RED}pre-commit is not installed!${NC}"
    echo -e "Installing pre-commit..."
    pip install pre-commit
    echo -e "${GREEN}pre-commit installed successfully!${NC}"
else
    echo -e "${GREEN}pre-commit is installed.${NC}"
fi

# Check if git hooks are installed
echo -e "\n${YELLOW}Checking if pre-commit hooks are installed...${NC}"
if [ ! -f .git/hooks/pre-commit ]; then
    echo -e "${RED}pre-commit hooks are not installed!${NC}"
    echo -e "Installing pre-commit hooks..."
    pre-commit install
    echo -e "${GREEN}pre-commit hooks installed successfully!${NC}"
else
    echo -e "${GREEN}pre-commit hooks are installed.${NC}"
fi

# Check if pre-commit config exists
echo -e "\n${YELLOW}Checking if .pre-commit-config.yaml exists...${NC}"
if [ ! -f .pre-commit-config.yaml ]; then
    echo -e "${RED}.pre-commit-config.yaml does not exist!${NC}"
    echo -e "Please create a .pre-commit-config.yaml file."
else
    echo -e "${GREEN}.pre-commit-config.yaml exists.${NC}"
fi

# Check for staged files
echo -e "\n${YELLOW}Checking for staged files...${NC}"
STAGED_FILES=$(git diff --name-only --cached)
if [ -z "$STAGED_FILES" ]; then
    echo -e "${RED}No files are staged for commit.${NC}"
    echo -e "${YELLOW}You need to stage files before running pre-commit:${NC}"
    echo -e "  git add <files>"
else
    echo -e "${GREEN}You have staged files that can be checked with pre-commit.${NC}"
fi

# Check for unstaged changes in staged files
echo -e "\n${YELLOW}Checking for unstaged changes in staged files...${NC}"
UNSTAGED_STAGED_FILES=$(git diff --name-only | grep -F "$(git diff --name-only --cached)")
if [ -n "$UNSTAGED_STAGED_FILES" ]; then
    echo -e "${RED}Warning: You have unstaged changes in files that are staged for commit.${NC}"
    echo -e "${YELLOW}This can cause confusion with pre-commit. Consider:${NC}"
    echo -e "  1. Staging all changes: git add -u"
    echo -e "  2. Using './scripts/run-pre-commit-staged.sh' to check only staged parts"
else
    echo -e "${GREEN}No unstaged changes in staged files.${NC}"
fi

# Offer to clean pre-commit cache
echo -e "\n${YELLOW}Would you like to clean the pre-commit cache? (y/n)${NC}"
read -r clean_cache
if [[ $clean_cache == "y" || $clean_cache == "Y" ]]; then
    echo -e "Cleaning pre-commit cache..."
    pre-commit clean
    echo -e "${GREEN}Cache cleaned successfully!${NC}"
fi

# Offer to run pre-commit on staged files only
echo -e "\n${YELLOW}Would you like to run pre-commit on staged files only? (y/n)${NC}"
read -r run_staged
if [[ $run_staged == "y" || $run_staged == "Y" ]]; then
    echo -e "Running pre-commit on staged files only..."
    ./scripts/run-pre-commit-staged.sh
fi

echo -e "\n${GREEN}Troubleshooting complete!${NC}"
echo -e "${BLUE}For more detailed information about pre-commit, see docs/pre-commit-guide.md${NC}"
