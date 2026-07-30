#!/bin/bash
cd "$(dirname "$0")"
set -a
source .env
set +a
./venv/bin/python hourly_notify.py
