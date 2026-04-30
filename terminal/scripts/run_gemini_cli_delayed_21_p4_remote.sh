#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/terminalbench"
JOB_NAME="gemini-cli-gemini3flashpreview-delayed-abstention-21-p4"
CONFIG="$ROOT/configs/gemini_cli_gemini3flashpreview_delayed_abstention_21_p4.json"
ENV_FILE="$ROOT/.env.gemini_vertex"
SECRET="$ROOT/secrets/google-agentic-abstention-sa.json"
JOB_DIR="$ROOT/jobs/$JOB_NAME"
MANIFEST="$ROOT/datasets/terminalbench_delayed_abstention_10/manifest.accepted_delayed_21.jsonl"
ANALYZER="$ROOT/scripts/analyze_delayed_abstention_job.py"
HARBOR="$ROOT/.venv312/bin/harbor"

cd "$ROOT"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

if [[ ! -x "$HARBOR" ]]; then
  echo "Missing executable Harbor binary: $HARBOR" >&2
  exit 2
fi
require_file "$CONFIG"
require_file "$ENV_FILE"
require_file "$SECRET"
require_file "$MANIFEST"
require_file "$ANALYZER"

if grep -Eq "^(GEMINI_API_KEY|GOOGLE_API_KEY)=" "$ENV_FILE"; then
  echo "Refusing to run: API key variable found in $ENV_FILE" >&2
  exit 2
fi

secret_mode="$(stat -c "%a" "$SECRET")"
if [[ "$secret_mode" != "600" ]]; then
  echo "Refusing to run: $SECRET must be chmod 600, got $secret_mode" >&2
  exit 2
fi

.venv312/bin/python - <<'PY'
import json
from harbor.models.job.config import JobConfig

config_path = "configs/gemini_cli_gemini3flashpreview_delayed_abstention_21_p4.json"
config = json.load(open(config_path))
JobConfig.model_validate(config)
agent = config["agents"][0]
manifest_tasks = [
    json.loads(line)["task_name"]
    for line in open("datasets/terminalbench_delayed_abstention_10/manifest.accepted_delayed_21.jsonl")
]
assert agent["name"] == "gemini-cli"
assert agent["model_name"] == "google/gemini-3-flash-preview"
assert "max_interaction_rounds" not in agent
assert config["n_concurrent_trials"] == 4
assert config["datasets"][0]["task_names"] == manifest_tasks
print("preflight_config_ok")
PY

"$HARBOR" run -y --env-file "$ENV_FILE" --config "$CONFIG"
python3 "$ANALYZER" --job-dir "$JOB_DIR" --manifest "$MANIFEST"

test -f "$JOB_DIR/delayed_abstention_summary.json"
test -f "$JOB_DIR/delayed_abstention_summary.md"

python3 - <<'PY'
import json

summary_path = "/workspace/terminalbench/jobs/gemini-cli-gemini3flashpreview-delayed-abstention-21-p4/delayed_abstention_summary.json"
summary = json.load(open(summary_path))
assert summary.get("task_count") == 21
assert summary.get("control_task_count", 0) == 0
print("gemini_cli_full_summary_ok")
PY

echo "gemini_cli_full_complete"
