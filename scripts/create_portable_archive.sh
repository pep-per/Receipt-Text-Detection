#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
STAMP="${PORTABLE_ARCHIVE_STAMP:-$(date +%Y%m%d)}"
DEFAULT_ARCHIVE="$PROJECT_PARENT/${PROJECT_NAME}-portable-no-data-${STAMP}.tar.zst"
ARCHIVE_PATH="${1:-$DEFAULT_ARCHIVE}"

if [[ "$ARCHIVE_PATH" != /* ]]; then
  ARCHIVE_PATH="$(pwd)/$ARCHIVE_PATH"
fi

if [[ "$ARCHIVE_PATH" == "$PROJECT_ROOT"/* ]]; then
  echo "Archive must be created outside the project directory: $ARCHIVE_PATH" >&2
  exit 1
fi

command -v tar >/dev/null
command -v zstd >/dev/null
command -v sha256sum >/dev/null

"$SCRIPT_DIR/verify_resume_assets.sh"

echo "Creating $ARCHIVE_PATH"
tar --zstd -cf "$ARCHIVE_PATH" \
  --exclude="$PROJECT_NAME/data" \
  --exclude="$PROJECT_NAME/data.tar.gz" \
  --exclude="$PROJECT_NAME/data.tar.gz.*" \
  --exclude="$PROJECT_NAME/.venv" \
  --exclude='*/__pycache__' \
  --exclude='*/__pycache__/*' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*/.pytest_cache' \
  --exclude='*/.mypy_cache' \
  --exclude='*/.ruff_cache' \
  --exclude='*/.ipynb_checkpoints' \
  -C "$PROJECT_PARENT" "$PROJECT_NAME"

(
  cd "$(dirname "$ARCHIVE_PATH")"
  sha256sum "$(basename "$ARCHIVE_PATH")" > "$(basename "$ARCHIVE_PATH").sha256"
)

echo "Verifying archive table of contents"
tar --zstd -tf "$ARCHIVE_PATH" >/dev/null

ls -lh "$ARCHIVE_PATH" "$ARCHIVE_PATH.sha256"
echo "Portable archive is ready."
