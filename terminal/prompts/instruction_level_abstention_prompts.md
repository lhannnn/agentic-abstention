# Instruction-Level Abstention Prompt Templates

## false_premise_or_contradiction

```text
You are rewriting a single TerminalBench task instruction into an instruction-level abstention variant.

Your job is to make the smallest possible change that makes a careful agent abstain immediately after reading the instruction, before taking any environment action.

Inputs:
- TASK_NAME: {{TASK_NAME}}
- ORIGINAL_INSTRUCTION:
{{ORIGINAL_INSTRUCTION}}

Goal:
Produce a rewritten instruction that is still recognizably the same TerminalBench task, but where the instruction itself is invalid at read time because of either:
- one central false premise, or
- one hard contradiction.

Global requirements:
1. Preserve the original task's domain, artifact surface, evaluator feel, realism, structure, tone, and approximate length.
2. Keep as much of the original as possible, including concrete paths, filenames, commands, APIs, deliverables, and success criteria.
3. Make only the minimum changes needed to create instruction-level abstention.
4. The abstention reason must be inferable from the instruction alone, before any file inspection, command execution, or environment interaction.
5. Do not turn this into a task-level impossibility discovered only after interacting with the environment.
6. Do not rely on missing files, broken dependencies, lack of permissions, lack of network, corrupted environments, or unavailable external services.
7. Do not add meta-language such as "you should abstain", "this task is impossible", "the instruction is invalid", or similar hints.
8. Do not make the task look silly, adversarial, or obviously synthetic.
9. Use exactly one mechanism. Choose either false premise or contradiction, not both.

Subtype requirements:
- If you choose false premise, introduce exactly one central false premise that makes the instruction invalid at read time.
- If you choose contradiction, introduce a concrete, technical, hard contradiction whose requirements cannot be satisfied simultaneously.

Silent self-check before answering:
- Would a careful agent know to abstain before acting?
- Did I keep the original task family and artifact surface?
- Did I use exactly one mechanism?
- Did I avoid turning this into an environment failure?

Return JSON matching the provided schema.
Set `chosen_subtype` to exactly `false_premise` or `contradiction`.
Set `rewritten_instruction` to only the rewritten instruction text.
```

## underspecified_intent

```text
You are rewriting a single TerminalBench task instruction into an instruction-level abstention variant.

Your job is to make the smallest possible change that makes a careful agent abstain immediately after reading the instruction, before taking any environment action.

Inputs:
- TASK_NAME: {{TASK_NAME}}
- ORIGINAL_INSTRUCTION:
{{ORIGINAL_INSTRUCTION}}

Goal:
Produce a rewritten instruction that is still recognizably the same TerminalBench task, but where the task goal is critically underspecified. A careful agent should abstain because the instruction no longer determines a unique, reasonably well-defined target.

Global requirements:
1. Preserve the original task's domain, artifact surface, evaluator feel, realism, structure, tone, and approximate length.
2. Keep as much of the original as possible, including concrete paths, filenames, commands, APIs, deliverables, and success criteria.
3. Make only the minimum changes needed to create instruction-level abstention.
4. The underspecification must be inferable from the instruction alone, before any file inspection, command execution, or environment interaction.
5. The ambiguity must be critical: there must be multiple materially different, reasonable completions, and the instruction must no longer determine a uniquely correct target.
6. The ambiguity must affect a key specification dimension such as the target artifact, output contract, success criterion, objective, authoritative input, or tie-breaking rule.
7. Do not introduce a hard contradiction.
8. Do not introduce a false premise.
9. Do not rely on missing files, broken dependencies, lack of permissions, lack of network, corrupted environments, or unavailable external services.
10. Do not add meta-language such as "you should abstain", "this task is ambiguous", or similar hints.
11. Do not make the task look silly, adversarial, or obviously synthetic.

Silent self-check before answering:
- Is there at least one critical decision that the instruction no longer uniquely specifies?
- Would a careful agent know to abstain before acting?
- Did I preserve the original task family and artifact surface?
- Did I avoid contradiction, false premise, and environment failure?

Return JSON matching the provided schema.
Set `rewritten_instruction` to only the rewritten instruction text.
```
