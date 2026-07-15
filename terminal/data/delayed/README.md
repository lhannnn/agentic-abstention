# Delayed Abstention Data

Files:

- `manifest.jsonl`: full delayed/control metadata rows
- `manifest.accepted_delayed_21.jsonl`: accepted delayed-only evaluation subset
- `manifest.schema.json`: manifest schema
- `specs/`: one structured rewrite spec per delayed case
- `reviews/`: historical review artifacts and rewrite-only acceptance policy

Delayed cases are accepted on rewrite correctness; visible verifier
permissiveness is not an acceptance criterion.

Reconstruct runnable tasks from `specs/` and an upstream task mirror, or use a
separately distributed generated-task artifact.
