# memagent — Human-brain memory for LLM agents

[![CI](https://github.com/1476989162/memagent/actions/workflows/ci.yml/badge.svg)](https://github.com/1476989162/memagent/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/memagent-local)](https://pypi.org/project/memagent-local/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Chinese README:** [README.md](README.md)

---

## The problem

LLM agents that run for more than a few turns have a memory problem. Either you stuff the entire history into the context window (and the model starts hallucinating from the noise), or you drop everything and the agent forgets what it just learned.

Most agent memory libraries (MemGPT, Mem0, Zep, etc.) use **fixed forgetting parameters** — a constant decay rate, a hard cap on stored items, hand-tuned similarity thresholds. They're generic scaffolding, not a real memory.

**memagent** is different: every memory has its own forgetting curve, the parameters **self-tune at runtime** from actual recall behavior, and memories are **reconsolidated when recalled** (like how your brain rewrites a memory each time you think about it).

## What you get

- **Zero dependencies** in the core package (pure Python stdlib)
- **Self-learning forgetting curve** — each memory learns its own decay rate from usage
- **Hot / Warm / Cold tiered storage** — frequently recalled memories promote, rarely recalled ones compact into summaries
- **Sleep consolidation** — periodic replay that strengthens important memories and compresses the rest (the "REM sleep" of your agent)
- **Retrieval-induced forgetting** — querying a memory actually strengthens it (the testing effect)
- **Three entry points**: CLI, REST API, and MCP server (for Hermes, Claude, Cursor, etc.)
- **650+ tests** passing on Python 3.10-3.13, Windows + Linux

## Install

```bash
pip install memagent-local              # core, zero dependencies
pip install memagent-local[mcp]         # + MCP server (for AI agents)
pip install memagent-local[embed-local] # + local embeddings (sentence-transformers)
pip install memagent-local[embed-fastembed]  # + ONNX embeddings (lightweight, no torch)
```

Verify:

```bash
memagent --version     # should print 0.3.5
```

## Quick start

```python
from memagent import MemoryAgent

agent = MemoryAgent(persist="memory.json")
agent.remember("User prefers concise responses", importance=0.9)
agent.remember("Project uses FastAPI on port 8000", importance=0.6)

results = agent.retrieve("what does the user prefer?")
print(results)
```

The agent learns your memory patterns — important memories survive longer, rarely-used ones compress into Cold summaries.

## Entry points

| Entrypoint | How | When to use |
|---|---|---|
| **CLI** | `memagent` | Ad-hoc queries, diagnostics, exports |
| **REST** | `python -m memagent.server --port 8000` | Non-Python clients (Node, Go, browser) |
| **MCP** | `python -m memagent.mcp_server` | Native integration with Hermes, Claude, Cursor |

## Why "self-tuning"?

Traditional memory systems pick a decay rate once and never change. That means:

- Frequently-referenced memories decay at the same rate as one-time trivia
- The parameters don't adapt to *your* usage pattern

memagent's τ-learner and plasticity-learner observe actual retrieval behavior and adjust each memory's decay rate at runtime. Your memory learns *from your usage*, not from a hardcoded constant.

## License

[MIT](LICENSE) — use it in your product, modify it, sell what you build on top.

## Looking for early users

First release. If you try it and something breaks, please [open an issue](https://github.com/1476989162/memagent/issues). The first few real users get priority fixes — this project is currently dogfooded by the author.

## Links

- 📦 PyPI: https://pypi.org/project/memagent-local/
- 📝 Changelog: https://github.com/1476989162/memagent/blob/main/CHANGELOG.md
- 🧪 CI: https://github.com/1476989162/memagent/actions
- 🐛 Issues: https://github.com/1476989162/memagent/issues
