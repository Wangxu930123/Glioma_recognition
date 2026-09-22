#!/usr/bin/env sh
set -eu

exec python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1

