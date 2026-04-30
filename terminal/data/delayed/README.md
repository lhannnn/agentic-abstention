# Delayed Abstention Data

This directory contains the delayed-abstention metadata release.

Files:

- `manifest.jsonl`: full delayed/control metadata rows
- `manifest.accepted_delayed_21.jsonl`: accepted delayed-only evaluation subset
- `manifest.schema.json`: manifest schema
- `specs/`: one structured rewrite spec per delayed case
- `reviews/`: historical review artifacts and rewrite-only acceptance policy

The release acceptance rule is rewrite correctness only. Visible verifier permissiveness is intentionally ignored for delayed-case acceptance.

Generated task directories are not tracked. Use the specs and an upstream task mirror or a separate generated-task artifact to reconstruct runnable task dirs.
