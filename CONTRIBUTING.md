# Contributing to Incremental Hierarchical Tucker Decomposition

Thank you for your interest in contributing to this project! This guide will help you get started with the contribution process.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please report it by creating an issue using the bug report template. Make sure to include:
- A clear and descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, etc.)

### Suggesting Features

Have an idea for a new feature? Please create an issue using the feature request template with:
- A clear description of the feature
- Use cases for the feature
- Any implementation ideas you might have

### Working on Issues

1. Comment on the issue you'd like to work on to express your interest
2. Fork the repository
3. Create a branch using the convention: `<branch_type>/<your_name>/<issue_number>`
4. Make your changes
5. Write tests if applicable
6. Update documentation if necessary
7. Submit a pull request

### Pull Request Process

1. Ensure your PR includes a description of the changes and references related issues
2. Make sure all tests pass
3. Update the documentation if needed
4. Wait for code review and address any feedback

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/dorukaks/htucker_incremental.git
cd htucker_incremental
```

2. Set up a virtual environment
```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

3. Install development dependencies
```bash
pip install -e .
pip install -r requirements.txt

# For development, install additional tools
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## Code Quality Tools

This repository uses several tools to ensure code quality:

### Automatic Linting

We use the following linting tools:
- **flake8** - for style guide enforcement
- **black** - for code formatting
- **isort** - for import sorting
- **mypy** - for type checking
- **pylint** - for additional checks

### Pre-commit Hooks

We use pre-commit hooks to automatically check your code before committing. To set up:

```bash
pre-commit install
```

This will run the linters on changed files when you try to commit. If there are issues, the commit will be blocked until you fix them.

#### Running Pre-commit on Staged Files Only

By default, pre-commit will run on the files you're trying to commit. However, if you want to manually run pre-commit only on staged files (ignoring unstaged changes), you can use:

```bash
# Using the provided script
./scripts/run-pre-commit-staged.sh

# Or using the git alias
git pre-commit
```

This is useful when you have modified files that are not yet ready to be committed but want to check only what's staged for the current commit.

For a comprehensive guide on using pre-commit in this repository, please see [docs/pre-commit-guide.md](docs/pre-commit-guide.md).

### Manual Linting

You can also run the linting tools manually:

```bash
# Format code with black
black htucker/ examples/

# Sort imports
isort htucker/ examples/

# Run flake8
flake8 htucker/ examples/

# Run type checking
mypy htucker/
```

## Running Tests

```bash
# Run the test file
python test_htucker.py
```

## Coding Conventions

- Follow PEP 8 style guidelines
- Write docstrings for functions and classes
- Include comments for complex code sections
- Write meaningful commit messages

## Versioning

We use [Semantic Versioning](https://semver.org/) (SemVer) for version numbers.

Thank you for contributing!
