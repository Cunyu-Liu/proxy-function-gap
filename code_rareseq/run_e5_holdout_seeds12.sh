#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash run_e5_holdout_convergence.sh "$@"
