# Pre-commit Usage Guide

This document provides detailed information on how to use pre-commit effectively in this repository.

## Overview

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It helps ensure that code quality checks are run before code is committed to the repository, catching issues early in the development process.

## Default Behavior

By default, pre-commit will:
- Run all configured hooks on files that are staged for commit
- Block the commit if any hooks fail
- Allow fixing issues and re-running hooks

## Common Issues

### Issue: Pre-commit checks non-staged files

By default, when running `pre-commit run` manually, it will check **all** files that match the hook patterns, not just those that are staged for commit. This can be confusing when you have unstaged changes that you're not ready to commit yet.

### Solution: Run pre-commit only on staged files

We provide two solutions:

1. **Use the provided script**:
   ```bash
   ./scripts/run-pre-commit-staged.sh
   ```

2. **Use the git alias**:
   ```bash
   git pre-commit
   ```

Both of these will run pre-commit only on files that are staged for commit, ignoring unstaged changes.

## Advanced Usage

### Running specific hooks

To run a specific hook:
```bash
pre-commit run <hook-id>
```

For example:
```bash
pre-commit run black
```

### Running on specific files

To run pre-commit on specific files:
```bash
pre-commit run --files file1.py file2.py
```

### Temporarily skipping hooks

To skip specific hooks for a commit:
```bash
SKIP=flake8,black git commit -m "Your message"
```

### Updating hooks

To update all hooks to their latest versions:
```bash
pre-commit autoupdate
```

## Troubleshooting

If you encounter issues with pre-commit:

1. **Hook fails but code seems correct**:
   Try running the hook directly (e.g., `black file.py`) to see more detailed errors

2. **Pre-commit seems slow**:
   Consider running only specific hooks or on specific files

3. **Need to temporarily bypass pre-commit**:
   Use `git commit --no-verify` (but use sparingly)

For more information, see the [pre-commit documentation](https://pre-commit.com/).
