#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-$ROOT_DIR/external/WebShop}"

ITEMS_URL="${ITEMS_URL:-}"
ATTR_URL="${ATTR_URL:-}"
HUMAN_URL="${HUMAN_URL:-}"
DOCS_URL="${DOCS_URL:-}"

mkdir -p "$WEBSHOP_ROOT/data" "$WEBSHOP_ROOT/search_engine/resources"

download_one() {
  local url="$1"
  local output="$2"
  if [[ -z "$url" ]]; then
    echo "skip: no URL provided for $output"
    return 0
  fi
  curl -L --fail --retry 3 "$url" -o "$output"
}

download_one "$ITEMS_URL" "$WEBSHOP_ROOT/data/items_shuffle.json"
download_one "$ATTR_URL" "$WEBSHOP_ROOT/data/items_ins_v2.json"
download_one "$HUMAN_URL" "$WEBSHOP_ROOT/data/items_human_ins.json"
download_one "$DOCS_URL" "$WEBSHOP_ROOT/search_engine/resources/documents.jsonl"

echo "done: raw assets staged under $WEBSHOP_ROOT"
