# Changelog

## Unreleased

- Added per-call `max_tokens` override to `LLMResponder.respond()` and `call_responder()` (signature-filtered for legacy responder plugins).
- Raised long-form output ceilings: `AgentConfig.llm_long_max_tokens` (4096) is now used for chapter writing; the FoxTable coder passes 4096 for code generation and critique.
- Root-caused the 47% truncation rate: the responder's default `max_tokens=1024` cut VB.NET code blocks mid-flight (279 of 286 truncated cycles produced exactly the 500-char fallback) and capped novel chapters around 1,000 characters.
- Unclosed code blocks now feed the real partial code to critique instead of the first 500 characters of prose.
- Added a consecutive-failure cap for forced retrains: a domain that still truncates after 3 consecutive forced rounds falls back to normal task selection instead of looping forever.

## 0.2.1

- Added immutable release manifests, versioned runtime installs, atomic activation, and offline rollback.
- Added validated named backups with byte-preserving pre-restore snapshots and checksum verification.
- Extended CI to build, verify, clean-install, smoke-test, and retain release artifacts.
- Added proprietary package metadata, release documentation, and dependency update automation.

## 0.2.0

- Added atomic persistence, rolling backup, stale-writer detection, and file locks.
- Persisted interest, graph, growth, cognition, curiosity, analogy, social, and emotion state.
- Prevented chapter overwrites and rejected truncated chapter output.
- Added novel audits, snapshots, overwrite evidence, revision archives, and reviewed-candidate promotion.
- Raised chapter completeness to 90% of the target and rejected incomplete final sentences.
- Added safe work-title migration and fixed chapter-context discovery.
- Fixed curiosity, graph, social-learning, and growth-summary contracts.
- Preserved compatibility with minimal responder plugins.
- Added thinking-disabled retry for reasoning-only model responses.
- Added finite autonomous defaults and consecutive-failure circuit breakers.
- Added installable CLI, health checks, cross-platform CI, and security documentation.
