# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly using [GitHub Security Advisories](https://github.com/AltioraLabs/beliefstate/security/advisories/new).

**Please do NOT report security vulnerabilities through public GitHub issues.**

### What to include

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested fixes (if applicable)

### Response timeline

- **Acknowledgment**: Within 48 hours of your report
- **Assessment**: Within 1 week, you will receive an initial assessment
- **Fix**: Critical vulnerabilities will be patched as soon as possible; lower severity issues will be addressed in the next release

### Safe harbor

We support safe harbor for security researchers who:

- Make a good faith effort to avoid privacy violations, data destruction, or disruption to our services
- Only interact with accounts you own or with explicit permission of the account holder
- Do not exploit a vulnerability beyond what is necessary to confirm its existence

We will not pursue legal action against researchers who follow these guidelines.

## Security Best Practices for Users

- Always use environment variables or a secrets manager for API keys — never hardcode them
- Use the principle of least privilege when configuring store backends (Redis, PostgreSQL)
- Keep your dependencies up to date (`pip install --upgrade`)
