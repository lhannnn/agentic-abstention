#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/terminalbench"
JOB_NAME="gemini-cli-gemini3flashpreview-delayed-abstention-canary"
CONFIG="$ROOT/configs/gemini_cli_gemini3flashpreview_delayed_abstention_canary.json"
ENV_FILE="$ROOT/.env.gemini_vertex"
SECRET="$ROOT/secrets/google-agentic-abstention-sa.json"
JOB_DIR="$ROOT/jobs/$JOB_NAME"
MANIFEST="$ROOT/datasets/terminalbench_delayed_abstention_10/manifest.accepted_delayed_gemini_cli_canary.jsonl"
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

config_path = "configs/gemini_cli_gemini3flashpreview_delayed_abstention_canary.json"
config = json.load(open(config_path))
JobConfig.model_validate(config)
agent = config["agents"][0]
assert agent["name"] == "gemini-cli"
assert agent["model_name"] == "google/gemini-3-flash-preview"
assert "max_interaction_rounds" not in agent
assert config["n_concurrent_trials"] == 1
assert config["datasets"][0]["task_names"] == ["gcode-to-text__delayed_observation"]
print("preflight_config_ok")
PY

"$HARBOR" run -y --env-file "$ENV_FILE" --config "$CONFIG"
python3 "$ANALYZER" --job-dir "$JOB_DIR" --manifest "$MANIFEST"

test -f "$JOB_DIR/delayed_abstention_summary.json"
test -f "$JOB_DIR/delayed_abstention_summary.md"
find "$JOB_DIR" -path "*/result.json" -print -quit | grep -q .

echo "gemini_cli_canary_complete"
