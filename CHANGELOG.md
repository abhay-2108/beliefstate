# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub issue templates (bug report, feature request)
- Pull request template
- Dependabot configuration for automated dependency updates
- CHANGELOG.md

### Changed
- Split CI and lint workflows: `ci.yml` handles tests, `lint.yml` handles code quality
- Hardened store backends: lowercase normalization, conversation_id in field keys

### Fixed
- LlamaIndex callback handler test (mock import ordering)
- Unused mypy `type: ignore` comments in integrations

## [1.0.2] - 2026-06-20

### Added
- Per-session turn counters replacing broken `turn_counter` property
- GenericAdapter `inject_context` method
- `postgres` extra in `[all]` dependencies

### Fixed
- FastAPI `get_session_id` guarded behind `if HAS_FASTAPI:` import check
- Redis/Memory store field key includes `conversation_id`
- Lowercase normalization in PostgreSQL `get_by_key`/`remove_belief`

## [1.0.1] - 2026-06-18

### Fixed
- Pronoun mapping bias in belief extraction (separate user/assistant prompts)
- Ollama adapter compatibility
- Documentation improvements

## [1.0.0] - 2026-06-15

### Added
- Initial public release
- Core belief tracking: `BeliefTracker`, `BeliefExtractor`, `BeliefResolver`
- Contradiction detection with `ContradictionDetector` and `ContradictionJudge`
- Provider adapters: OpenAI, Anthropic, Gemini, Ollama, LiteLLM
- Store backends: SQLite, Redis, PostgreSQL, In-Memory
- Framework integrations: FastAPI, Flask, ASGI, WSGI, LangChain, LlamaIndex
- Resilience layer: `ResilientAdapterWrapper`, `CircuitBreaker`
- Dispatchers: Asyncio, Sync, Celery, RQ
- Structured logging via `TrackerEvent`
- Full test suite and documentation

[Unreleased]: https://github.com/abhay-2108/beliefstate/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/abhay-2108/beliefstate/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/abhay-2108/beliefstate/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/abhay-2108/beliefstate/releases/tag/v1.0.0
