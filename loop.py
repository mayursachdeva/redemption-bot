"""Run execute.run_once() on a lower-timeframe cadence. Ctrl+C to stop."""
from __future__ import annotations

import time
from datetime import datetime

from execute import run_once
from execute_futures import run_once_short
from leverage_journal import main as update_leverage_journal

INTERVAL_SECONDS = 5 * 60  # lower timeframe: matches the 1h-momentum/5m-Kronos signal granularity

if __name__ == "__main__":
    while True:
        print(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===")
        try:
            run_once()
        except Exception as e:  # ponytail: one bad cycle (API hiccup, etc.) shouldn't kill the loop
            print(f"cycle failed: {e}")
        try:
            run_once_short()
        except Exception as e:
            print(f"short cycle failed: {e}")
        try:
            update_leverage_journal()
        except Exception as e:
            print(f"leverage journal update failed: {e}")
        print(f"sleeping {INTERVAL_SECONDS // 60}min...")
        time.sleep(INTERVAL_SECONDS)
