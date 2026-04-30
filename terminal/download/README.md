# TerminalBench Assets

The materializers need a Harbor task cache or TerminalBench task mirror with:

```text
instruction.md
task.toml
tests/
solution/
environment/
```

Default cache path:

```text
~/.cache/harbor/tasks
```

Override with `--cache-root`.

Delayed task directories are reconstructed from the upstream task plus `data/delayed/specs/`.
