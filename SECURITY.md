# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |

Only the latest release on the `main` branch is actively supported with
security updates.

## Reporting a Vulnerability

**Please do NOT open a public GitHub Issue for security vulnerabilities.**

If you discover a security vulnerability in SAM3-CPU, please report it
responsibly:

1. **Email**: Send details to **p.aparajeya@gmail.com** with the subject
   line: `[SECURITY] SAM3-CPU — <brief description>`.
2. **Include**:
   - A description of the vulnerability and its potential impact.
   - Steps to reproduce or a proof-of-concept (if possible).
   - The version / commit hash you tested against.
   - Your suggested fix (optional but appreciated).

## Response Timeline

| Step                  | Target      |
|-----------------------|-------------|
| Acknowledgement       | 48 hours    |
| Initial assessment    | 5 days      |
| Fix or mitigation     | 30 days     |
| Public disclosure     | After fix   |

We will work with you to understand the issue and coordinate a fix before any
public disclosure. You will be credited in the release notes unless you prefer
to remain anonymous.

## Scope

The following are **in scope**:

- Code in `sam3/` and supporting scripts.
- Dependencies installed by `setup.sh` / `pyproject.toml`.
- Model-loading and file-handling logic that could be exploited with crafted
  inputs.

The following are **out of scope**:

- Vulnerabilities in upstream SAM 3 model weights hosted by Meta.
- Issues in third-party dependencies that should be reported to those projects
  directly.
- Social engineering or phishing attacks.

## Best Practices for Users

- Always run `pip audit` or `uv pip audit` periodically to check for known
  vulnerabilities in dependencies.
- Pin dependency versions in production deployments.
- Do not expose the inference API to untrusted networks without authentication.

---

Thank you for helping keep SAM3-CPU and its users safe.
