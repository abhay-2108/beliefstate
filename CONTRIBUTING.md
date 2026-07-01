# Contributing to BeliefState

Thank you for considering contributing to BeliefState! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Report unacceptable behavior via [GitHub Security Advisories](https://github.com/AltioraLabs/beliefstate/security/advisories/new).

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/beliefstate.git
   cd beliefstate
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Install the package in editable mode with all dev and optional dependencies
pip install -e ".[dev,all]"
```

### Verify Everything Works

```bash
# Run the linter
ruff check .

# Run the formatter check
ruff format --check .

# Run the type checker
mypy beliefstate

# Run the tests
pytest
```

## Coding Standards

### Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 88 characters
- Target Python version: 3.10+

### Type Checking

- We use [mypy](https://mypy-lang.org/) in strict mode
- All public functions and classes must have type annotations
- Use `# type: ignore[...]` only when necessary (e.g., optional dependencies)

### Testing

- Write tests for new features and bug fixes
- Tests live in the `tests/` directory
- Use `pytest` with `pytest-asyncio` for async tests
- Aim for meaningful coverage, not just high percentages
- Mock external services (Redis, PostgreSQL, OpenAI, etc.) in tests

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove")
- Reference issue numbers where applicable (e.g., `Fix #42`)

Examples:
```
Add PostgreSQL store connection pooling
Fix contradictory belief detection for nested objects
Update README with new integration examples
```

### Changelog

Update `CHANGELOG.md` under the `[Unreleased]` section using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Add entries under the appropriate category:

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

## Pull Request Process

1. **Ensure quality** before submitting:
   ```bash
   ruff check .
   ruff format --check .
   mypy beliefstate
   pytest
   ```

2. **Mandatory Manual Verification**:
   - To ensure all contributions are genuine and tested, you **must** manually verify your changes and provide before-and-after visual proof:
     - **For Core Library Changes (`beliefstate/`)**: You must use the `test_package/` environment. Create a Python test file inside `test_package/` (or update an existing script) to demonstrate your fix or feature. Take a screenshot **BEFORE** your change (showing the bug/issue) and **AFTER** your change (showing it resolved/working).
     - **For Documentation and Dashboard UI Fixes**: Take a screenshot **BEFORE** and **AFTER** your fix of the documentation pages or the dashboard UI showing the issue and your fix.
   - You must embed these screenshots in your pull request as proof of verification.

3. **Fill out the PR template** completely, including:
   - Description of changes
   - Related issue number
   - Type of change (bug fix, feature, breaking change, etc.)
   - Your before-and-after verification screenshots.

4. **Keep PRs focused**: One logical change per PR. If you have multiple unrelated fixes, submit them as separate PRs.

5. **Update documentation** if your change affects the public API, README, or docstrings.

6. **Add tests** for new functionality and regression tests for bug fixes.

7. **Request a review** from a maintainer. Address review feedback promptly.

### PR Review Criteria

PRs will be reviewed for:

- Correctness and completeness
- Test coverage for new/changed behavior
- Code style consistency (ruff, mypy pass)
- Documentation updates where needed
- Backward compatibility (or clear migration path for breaking changes)

## Issue Guidelines

### Bug Reports

Use the **Bug Report** template. Include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Python version and OS
- Relevant package versions (`pip list | grep beliefstate`)

### Feature Requests

Use the **Feature Request** template. Include:

- A clear description of the problem you're trying to solve
- Your proposed solution
- Alternatives you've considered

### Questions

For general questions, please use [GitHub Discussions](https://github.com/AltioraLabs/beliefstate/discussions) if available, or open a feature request template and mark it as a question.

## Project Structure

```
beliefstate/
  adapters/        # Provider adapters (OpenAI, Anthropic, etc.)
  integrations/    # Framework integrations (FastAPI, Flask, LangChain, etc.)
  store/           # Storage backends (SQLite, Redis, PostgreSQL, Memory)
  tracker.py       # Core BeliefTracker class
  extractor.py     # Belief extraction from LLM responses
  detector.py      # Contradiction detection
  resolver.py      # Conflict resolution
  config.py        # TrackerConfig
  models.py        # Pydantic models (Belief, etc.)
  resilience.py    # Circuit breaker, retry wrappers
  dispatcher.py    # Async/sync task dispatchers
  judge.py         # LLM-based contradiction judging
  call.py          # LLMCall/LLMResponse data classes
  logging_utils.py # Structured event logging
tests/             # Test suite
docs/              # Documentation site
```

## Security

If you discover a security vulnerability, please **do not** open a public issue. Instead, report it via [GitHub Security Advisories](https://github.com/AltioraLabs/beliefstate/security/advisories/new). See [SECURITY.md](SECURITY.md) for details.

## License

By contributing to BeliefState, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
