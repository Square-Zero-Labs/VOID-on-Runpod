#!/bin/bash
set -euo pipefail

pkill -f "python3 app.py" >/dev/null 2>&1 || true
pkill -f "python app.py" >/dev/null 2>&1 || true

export VOID_RESTART_ONLY=1
/opt/void_template/start-void.sh
