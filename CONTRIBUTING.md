# Contributing to SAM3-CPU

Thank you for your interest in contributing to **SAM3-CPU**! Whether it's a bug
fix, a new feature, or improved documentation — we'd love your input.

Please read this guide before submitting a pull request.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Becoming a Contributor / Maintainer](#becoming-a-contributor--maintainer)

---

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By
participating you are expected to uphold this code. Please report unacceptable
behaviour to **p.aparajeya@gmail.com**.

---

## Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.10  |
| uv          | latest  |
| FFmpeg      | ≥ 5.0   |

### Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/sam3-cpu.git
cd sam3-cpu

# Run the setup script (creates venv, installs dependencies + model weights)
bash setup.sh

# Activate the virtual environment
source .venv/bin/activate
```

### Verify

```bash
uv run python -m pytest tests/ -v
```

All tests should pass before you begin making changes.

---

## Development Workflow

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. **Make your changes** — keep commits atomic and well-described.
4. **Add or update tests** for any new or changed functionality.
5. **Run the full test suite** (see [Testing](#testing)).
6. **Push** your branch and **open a Pull Request** against `main`.

---

## Coding Standards

- **Style** — Follow existing code style and conventions. Use clear,
  descriptive variable and function names.
- **Type hints** — Add type annotations to all public function signatures.
- **Docstrings** — Use Google-style docstrings for all public classes and
  functions.
- **Imports** — Group imports in order: stdlib → third-party → local. Use
  absolute imports.
- **Line length** — 88 characters (Black default).
- **No commented-out code** — Remove dead code; rely on version control for
  history.

---

## Testing

We use **pytest** for all tests.

```bash
# Run full suite
uv run python -m pytest tests/ -v

# Run a specific test file
uv run python -m pytest tests/test_video_prompter.py -v

# Run a single test
uv run python -m pytest tests/test_video_prompter.py::TestVideoPrompter::test_basic_init -v
```

### Test guidelines

- Place tests in the `tests/` directory.
- Name test files `test_*.py` and test functions `test_*`.
- Use `unittest.mock` to mock heavy I/O (model loading, video encoding).
- Aim for fast unit tests — avoid requiring GPU or large model weights.
- Every bug fix should include a regression test.

---

## Pull Request Process

1. **Fill out the PR template** — describe what changed, why, and how to test.
2. **Keep PRs focused** — one logical change per PR.
3. **Ensure all tests pass** — `uv run python -m pytest tests/ -v`.
4. **Update documentation** — especially the README for user-facing changes.
5. **Respond to review feedback** promptly.
6. A maintainer will review and merge once approved.

### Commit messages

Use clear, imperative-mood commit messages:

```
Add frame-range support to video_prompter

- Parse --frame-range and --time-range CLI flags
- Add _resolve_range() and _parse_timestamp() helpers
- Add 12 tests for range validation
```

---

## Issue Guidelines

Use [GitHub Issues](https://github.com/rhubarb-ai/sam3-cpu/issues) for:

- **Bug reports** — use the Bug Report template; include OS, Python version,
  full error output, and steps to reproduce.
- **Feature requests** — use the Feature Request template; describe the use
  case and expected behaviour.
- **Questions** — label with `question`; check existing issues and the README
  first.

---

## Becoming a Contributor / Maintainer

If you'd like to be added as a regular contributor or maintainer, please
[open a GitHub Issue](https://github.com/rhubarb-ai/sam3-cpu/issues/new) with
the title **"Contributor request"** and a brief description of your background
and intended contributions.

---

## License

By contributing to SAM3-CPU, you agree that your contributions will be licensed
under the same [SAM License](LICENSE) that covers the project.

---

Thank you for helping make SAM3-CPU better!
