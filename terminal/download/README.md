# TerminalBench Raw Asset Guide

This release does not vendor upstream TerminalBench tasks, Docker build contexts, or generated Harbor task directories.

## Required External Assets

You need one of the following:

- an upstream TerminalBench checkout plus a Harbor task cache, or
- an internal mirror/artifact containing the source task directories used by Harbor.

Each source task directory must contain at least:

```text
instruction.md
task.toml
tests/
solution/
environment/
```

The immediate materializer defaults to:

```text
~/.cache/harbor/tasks
```

You can override it with `--cache-root`.

## Suggested Layout

```text
terminal/
  external/terminal-bench/
  datasets/                       # generated, ignored by git
  jobs/                           # generated, ignored by git
```

## Delayed Tasks

The delayed case pack in this release includes structured rewrite specs and review artifacts. It intentionally omits generated task directories. Use your own generated-task artifact or reconstruct the environment edits from `data/delayed/specs/` against the corresponding upstream TerminalBench source tasks.

## Secrets

Do not commit `.env` files, OpenAI/OpenRouter keys, Google service account JSON files, ADC files, or remote runner logs. Use `.env.example` as the template for local configuration.
