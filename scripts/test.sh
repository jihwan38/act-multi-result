#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec python -B -m _internal.test "$@"
