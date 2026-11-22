"""Write data/sample, a seeded one-second tape so the page opens with no Databento file.

The walk is not a market recording. Replace it by putting your own ohlcv-1s
.dbn.zst in data/ and running `python data.py --rebuild`.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

OUT = ROOT / "data" / "sample"
START = datetime(2024, 6, 2, 22, 0, tzinfo=timezone.utc)  # 18:00 ET
SECONDS = 17 * 60 * 60
TICK = data.TICK


def main() -> None:
    rng = random.Random(7)
    price = 18800.0
    rec = np.empty(SECONDS, dtype=data.DTYPE)
    start = int(START.timestamp())
    for i in range(SECONDS):
        minute = i / 60
        drift = 0.02 * math.sin(minute / 90) + 0.008
        path = [price]
        for _ in range(4):
            path.append(path[-1] + rng.gauss(drift, 0.18))
        close = round(path[-1] / TICK) * TICK
        high = round(max(path) / TICK) * TICK
        low = round(min(path) / TICK) * TICK
        open_ = round(price / TICK) * TICK
        rec[i]["ts"] = start + i
        rec[i]["o"] = round(open_ / TICK)
        rec[i]["h"] = round(high / TICK)
        rec[i]["l"] = round(low / TICK)
        rec[i]["c"] = round(close / TICK)
        rec[i]["v"] = int(2 + abs(close - open_) * 40 + rng.random() * 8)
        price = close
    # 18:00 ET is 22:00 UTC. 09:30 ET is 13:30 UTC the next calendar day.
    rth = start + int(15.5 * 3600)
    day = {
        "date": "2024-06-03",
        "symbol": "NQM4",
        "i0": 0,
        "i1": SECONDS,
        "start_ts": start,
        "end_ts": start + SECONDS - 1,
        "roll": False,
        "open": float(rec["o"][0]) * TICK,
        "rth": float(rec["o"][int(15.5 * 3600)]) * TICK,
        "rth_ts": rth,
    }
    meta = {
        "tick": TICK,
        "point_value": data.POINT_VALUE,
        "rows": SECONDS,
        "first": start,
        "last": start + SECONDS - 1,
        "rolls": 0,
        "sessions": 1,
        "stats": {
            "raw_rows": SECONDS,
            "not_outright": 0,
            "bad_ohlc": 0,
            "off_tick": 0,
            "wide_second": 0,
            "closed_session": 0,
            "spike": 0,
            "not_front": 0,
            "duplicate": 0,
            "kept": SECONDS,
        },
        "days": [day],
        "built": "sample",
        "rule": (
            "Synthetic sample, so the page opens before a Databento file is added. "
            "Prices are a seeded random walk, not Nasdaq futures. "
            "Put one ohlcv-1s .dbn.zst in data/ and run python data.py --rebuild "
            "for a front-month tape."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    rec.tofile(OUT / "front_1s.bin")
    (OUT / "front_meta.json").write_text(json.dumps(meta))
    print(f"wrote {SECONDS:,} seconds to {OUT}")


if __name__ == "__main__":
    main()
