"""NQ one-second replay.

    python data.py

Serves the replay on http://127.0.0.1:8780. With no file of your own it
opens the sample in data/sample. Put one Databento ohlcv-1s .dbn.zst in
data/ and run `python data.py --rebuild` for a front-month tape.

The tape is one outright at a time, raw (not back-adjusted). The roll is a
sticky volume cross: after a finished session where the next quarter trades
more than the current one, the following session uses the next quarter and
does not flip back.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BIN_PATH = DATA / "front_1s.bin"
META_PATH = DATA / "front_meta.json"
ROLL_PATH = DATA / "roll_schedule.csv"
SAMPLE_BIN = DATA / "sample" / "front_1s.bin"
SAMPLE_META = DATA / "sample" / "front_meta.json"
PAGE = ROOT / "nq_tape.html"
JOURNAL = ROOT / "journal.json"
DRAWINGS = ROOT / "drawings.json"

from roll import MONTH, _resolve_year, globex_trade_date, roll_schedule  # noqa: E402

NY = ZoneInfo("America/New_York")
TICK = 0.25
POINT_VALUE = 20.0
PORT = 8780
DTYPE = np.dtype(
    [
        ("ts", "<i8"),
        ("o", "<i4"),
        ("h", "<i4"),
        ("l", "<i4"),
        ("c", "<i4"),
        ("v", "<u4"),
    ]
)
TIMEFRAMES = {
    "1s": 1,
    "2s": 2,
    "3s": 3,
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "2m": 120,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
}

TAPE: np.ndarray | None = None
META: dict | None = None


def _ticks(px: np.ndarray) -> np.ndarray:
    return np.rint(px / TICK).astype(np.int32)


def clean_chunk(df: pd.DataFrame, last_close: dict[str, float], stats: dict, tally: bool) -> pd.DataFrame:
    """Keep valid NQ quarterly outrights. Volume from these rows decides the roll."""
    if df.index.name == "ts_event" or "ts_event" not in getattr(df, "columns", []):
        df = df.reset_index()
    if tally:
        stats["raw_rows"] += int(len(df))
    sym = df["symbol"].astype(str)
    parts = sym.str.extract(r"^NQ([HMUZ])(\d{1,2})$")
    keep = parts[0].notna().to_numpy()
    if tally:
        stats["not_outright"] += int((~keep).sum())
    df = df.loc[keep, ["ts_event", "open", "high", "low", "close", "volume", "symbol"]].copy()
    if df.empty:
        return df
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    structure = (
        np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(v)
        & (o > 0) & (h > 0) & (l > 0) & (c > 0)
        & (v >= 1)
        & (h >= l) & (h >= o) & (h >= c) & (l <= o) & (l <= c)
        & (c > 5000) & (c < 80000)
    )
    if tally:
        stats["bad_ohlc"] += int((~structure).sum())
    df = df.iloc[np.flatnonzero(structure)].copy()
    if df.empty:
        return df
    px = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    off = (np.abs(px / TICK - np.rint(px / TICK)) > 1e-4).any(axis=1)
    wide = (df["high"].to_numpy(dtype=float) - df["low"].to_numpy(dtype=float)) > 80
    if tally:
        stats["off_tick"] += int(off.sum())
        stats["wide_second"] += int((~off & wide).sum())
    df = df.iloc[np.flatnonzero(~off & ~wide)].copy()
    if df.empty:
        return df

    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    local = df["ts_event"].dt.tz_convert(NY)
    wd = local.dt.weekday.to_numpy()
    mins = (local.dt.hour * 60 + local.dt.minute).to_numpy()
    closed = (
        ((wd == 4) & (mins >= 17 * 60))
        | (wd == 5)
        | ((wd == 6) & (mins < 18 * 60))
        | ((wd < 4) & (mins >= 17 * 60) & (mins < 18 * 60))
    )
    if tally:
        stats["closed_session"] += int(closed.sum())
    df = df.iloc[np.flatnonzero(~closed)].copy()
    if df.empty:
        return df

    prev = df.groupby("symbol", sort=False)["close"].shift()
    first = ~prev.notna()
    if first.any():
        seeded = df.loc[first, "symbol"].map(last_close)
        prev.loc[first] = seeded
    jump = (df["close"] - prev).abs()
    fence = np.maximum(60.0, prev.to_numpy(dtype=float) * 0.004)
    spike = prev.notna() & (jump.to_numpy() > fence)
    if tally:
        stats["spike"] += int(spike.sum())
    df = df.loc[~spike].copy()
    if df.empty:
        return df
    last = df.groupby("symbol", sort=False)["close"].last()
    last_close.update(last.to_dict())

    local = df["ts_event"].dt.tz_convert(NY)
    df["mins"] = local.dt.hour * 60 + local.dt.minute
    df["trade_date"] = globex_trade_date(df["ts_event"])
    years = [
        _resolve_year(yy, int(y))
        for yy, y in zip(
            df["symbol"].str.extract(r"^NQ[HMUZ](\d{1,2})$")[0].tolist(),
            df["ts_event"].dt.year.tolist(),
        )
    ]
    mcode = df["symbol"].str.extract(r"^NQ([HMUZ])")[0].map(MONTH)
    df["contract_month"] = np.asarray(years) * 12 + mcode.to_numpy()
    df["ts"] = (df["ts_event"].astype("int64") // 10**9).astype(np.int64)
    return df


def _volume_frame(vol: dict) -> pd.DataFrame:
    rows = [
        {
            "root": "NQ",
            "trade_date": pd.Timestamp(key[0]),
            "contract_month": key[1],
            "symbol": symbol,
            "volume": amount,
        }
        for key, (amount, symbol) in vol.items()
    ]
    return pd.DataFrame(rows)


def raw_file() -> Path:
    """The one Databento ohlcv-1s file sitting in data/."""
    files = sorted(path for path in DATA.glob("*.dbn.zst") if path.is_file())
    if len(files) != 1:
        raise SystemExit("Put one Databento ohlcv-1s .dbn.zst in data/, then run: python data.py --rebuild")
    return files[0]


def build() -> None:
    import databento as db

    raw = raw_file()
    print(f"Cleaning {raw.name}", flush=True)
    stats = {
        "raw_rows": 0,
        "not_outright": 0,
        "bad_ohlc": 0,
        "off_tick": 0,
        "wide_second": 0,
        "closed_session": 0,
        "spike": 0,
        "not_front": 0,
        "duplicate": 0,
        "kept": 0,
    }
    last_close: dict[str, float] = {}
    vol: dict[tuple, list] = {}
    t0 = time.time()
    store = db.DBNStore.from_file(raw)
    for i, df in enumerate(
        store.to_df(price_type="float", pretty_ts=True, map_symbols=True, count=750_000)
    ):
        part = clean_chunk(df, last_close, stats, tally=False)
        if part.empty:
            continue
        grouped = part.groupby(["trade_date", "contract_month", "symbol"], sort=False)["volume"].sum()
        for (day, month, symbol), amount in grouped.items():
            key = (pd.Timestamp(day).date().isoformat(), int(month))
            slot = vol.get(key)
            if slot is None:
                vol[key] = [float(amount), symbol]
            else:
                slot[0] += float(amount)
        print(
            f"  pass 1 chunk {i + 1}: {stats['raw_rows']:,} rows  {time.time() - t0:.0f}s",
            flush=True,
        )

    volumes = _volume_frame(vol)
    schedule = roll_schedule(volumes, "NQ")
    schedule.to_csv(ROLL_PATH, index=False)
    choice = {
        pd.Timestamp(r.trade_date).date().isoformat(): r.active_symbol
        for r in schedule.itertuples(index=False)
    }
    roll_days = {
        pd.Timestamp(r.trade_date).date().isoformat()
        for r in schedule.itertuples(index=False)
        if bool(r.is_roll)
    }
    print(f"Rolls: {len(roll_days)}  sessions: {len(choice)}", flush=True)

    last_close.clear()
    tmp = BIN_PATH.with_suffix(".bin.partial")
    days = []
    day = None
    last_ts = -1
    n = 0
    with tmp.open("wb") as out:
        store = db.DBNStore.from_file(raw)
        for i, df in enumerate(
            store.to_df(price_type="float", pretty_ts=True, map_symbols=True, count=750_000)
        ):
            part = clean_chunk(df, last_close, stats, tally=True)
            if part.empty:
                continue
            day_key = part["trade_date"].dt.strftime("%Y-%m-%d")
            active = day_key.map(choice)
            front = part["symbol"].to_numpy() == active.to_numpy()
            stats["not_front"] += int((~front).sum())
            part = part.iloc[np.flatnonzero(front)]
            if part.empty:
                continue
            ts = part["ts"].to_numpy()
            keep = np.ones(len(ts), dtype=bool)
            keep[0] = ts[0] != last_ts
            if len(ts) > 1:
                keep[1:] = ts[1:] != ts[:-1]
            stats["duplicate"] += int((~keep).sum())
            part = part.iloc[np.flatnonzero(keep)]
            if part.empty:
                continue
            rec = np.empty(len(part), dtype=DTYPE)
            rec["ts"] = part["ts"].to_numpy()
            rec["o"] = _ticks(part["open"].to_numpy(dtype=float))
            rec["h"] = _ticks(part["high"].to_numpy(dtype=float))
            rec["l"] = _ticks(part["low"].to_numpy(dtype=float))
            rec["c"] = _ticks(part["close"].to_numpy(dtype=float))
            rec["v"] = np.clip(part["volume"].to_numpy(dtype=np.int64), 0, np.iinfo(np.uint32).max).astype(np.uint32)
            mins = part["mins"].to_numpy(dtype=np.int32)
            symbols = part["symbol"].to_numpy()
            dates = part["trade_date"].dt.strftime("%Y-%m-%d").to_numpy()
            cuts = np.flatnonzero(dates[1:] != dates[:-1]) + 1
            bounds = np.concatenate(([0], cuts, [len(rec)]))
            for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
                d = str(dates[a])
                if day is None or day["date"] != d:
                    if day is not None:
                        day["i1"] = n + a
                        day["end_ts"] = int(last_ts)
                        days.append(day)
                    day = {
                        "date": d,
                        "symbol": str(symbols[a]),
                        "i0": n + a,
                        "i1": n + b,
                        "start_ts": int(rec["ts"][a]),
                        "end_ts": int(rec["ts"][b - 1]),
                        "roll": d in roll_days,
                        "open": float(rec["o"][a]) * TICK,
                        "rth": None,
                        "rth_ts": None,
                    }
                else:
                    day["i1"] = n + b
                    day["end_ts"] = int(rec["ts"][b - 1])
                if day["rth"] is None:
                    hit = np.flatnonzero((mins[a:b] >= 9 * 60 + 30) & (mins[a:b] < 16 * 60))
                    if hit.size:
                        k = a + int(hit[0])
                        day["rth"] = float(rec["o"][k]) * TICK
                        day["rth_ts"] = int(rec["ts"][k])
                last_ts = int(rec["ts"][b - 1])
            n += len(rec)
            rec.tofile(out)
            print(f"  pass 2 chunk {i + 1}: {n:,} front seconds  {time.time() - t0:.0f}s", flush=True)
    if day is not None:
        day["i1"] = n
        day["end_ts"] = int(last_ts)
        days.append(day)
    tmp.replace(BIN_PATH)
    stats["kept"] = n
    meta = {
        "tick": TICK,
        "point_value": POINT_VALUE,
        "rows": n,
        "first": int(days[0]["start_ts"]) if days else 0,
        "last": int(days[-1]["end_ts"]) if days else 0,
        "rolls": len(roll_days),
        "sessions": len(days),
        "stats": stats,
        "days": days,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rule": (
            "Front month only, prices unadjusted. Roll sticks after the first "
            "finished Globex session where the next quarter's volume is higher, "
            "and takes effect the next session. The 17:00–18:00 ET halt and the "
            "Friday 17:00–Sunday 18:00 ET close are removed. Seconds that jump "
            "more than 60 points (or 0.4%) from the previous print of that "
            "contract are dropped, as are one-second ranges wider than 80 points."
        ),
    }
    META_PATH.write_text(json.dumps(meta))
    print(
        f"Tape {n:,} seconds, {len(days)} sessions, {len(roll_days)} rolls → {BIN_PATH.name}",
        flush=True,
    )


def tape_paths() -> tuple[Path, Path]:
    """A built tape in data/, or the committed sample when there is none."""
    if BIN_PATH.exists() and META_PATH.exists():
        return BIN_PATH, META_PATH
    if SAMPLE_BIN.exists() and SAMPLE_META.exists():
        return SAMPLE_BIN, SAMPLE_META
    raise SystemExit("No tape. Add a .dbn.zst to data/ and run: python data.py --rebuild")


def load() -> None:
    global TAPE, META
    bin_path, meta_path = tape_paths()
    META = json.loads(meta_path.read_text())
    TAPE = np.memmap(bin_path, dtype=DTYPE, mode="r")
    if len(TAPE) != META["rows"]:
        raise SystemExit(f"{bin_path.name} has {len(TAPE)} rows, meta says {META['rows']}")


def _px(ticks: int) -> float:
    return ticks * TICK


def bars_ending(end_ts: int, tf: int, count: int) -> dict:
    assert TAPE is not None and META is not None
    ts = TAPE["ts"]
    i1 = int(np.searchsorted(ts, end_ts, side="right"))
    if i1 <= 0:
        return {"bars": [], "sessions": [], "cursor": end_ts}
    span = max(tf * count * 4, tf * 20)
    i0 = int(np.searchsorted(ts, int(ts[i1 - 1]) - span, side="left"))
    # Widen until there are enough buckets or we hit the start of the tape.
    for _ in range(8):
        buckets = (ts[i0:i1].astype(np.int64) // tf)
        n_buckets = 1 + int(np.unique(buckets).size) if i1 > i0 else 0
        if n_buckets >= count or i0 == 0:
            break
        span *= 3
        i0 = int(np.searchsorted(ts, int(ts[i1 - 1]) - span, side="left"))
    sl = TAPE[i0:i1]
    bucket = (sl["ts"].astype(np.int64) // tf) * tf
    # Reduce in order. The forming bucket includes only seconds <= end_ts.
    change = np.flatnonzero(bucket[1:] != bucket[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(sl)]))
    if len(starts) > count:
        starts = starts[-count:]
        ends = ends[-count:]
    bars = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        seg = sl[a:b]
        bars.append(
            {
                "time": int(bucket[a]),
                "open": _px(int(seg["o"][0])),
                "high": _px(int(seg["h"].max())),
                "low": _px(int(seg["l"].min())),
                "close": _px(int(seg["c"][-1])),
                "volume": int(seg["v"].astype(np.int64).sum()),
            }
        )
    first = bars[0]["time"] if bars else end_ts
    sessions = [
        d
        for d in META["days"]
        if d["end_ts"] >= first and d["start_ts"] <= end_ts
    ]
    return {"bars": bars, "sessions": sessions, "cursor": int(ts[i1 - 1])}


def step_cursor(cursor: int, tf: int, direction: int) -> int:
    """Move one candle. Forward always leaves the current bucket, even mid-bar."""
    assert TAPE is not None
    ts = TAPE["ts"]
    if direction > 0:
        nxt = (int(cursor) // tf) * tf + tf
        i = int(np.searchsorted(ts, nxt, side="left"))
        if i >= len(ts):
            return int(ts[-1])
        bucket = int(ts[i]) // tf * tf
        j = int(np.searchsorted(ts, bucket + tf, side="left")) - 1
        return int(ts[j])
    bucket = int(cursor) // tf * tf
    i = int(np.searchsorted(ts, bucket, side="left")) - 1
    if i < 0:
        return int(ts[0])
    return int(ts[i])


def cursor_info(cursor: int) -> dict:
    assert META is not None and TAPE is not None
    days = META["days"]
    hit = days[0]
    for d in days:
        if d["start_ts"] <= cursor:
            hit = d
        else:
            break
    i = int(np.searchsorted(TAPE["ts"], cursor, side="right")) - 1
    i = max(0, min(i, len(TAPE) - 1))
    return {
        "symbol": hit["symbol"],
        "date": hit["date"],
        "roll": bool(hit["roll"]),
        "open": hit["open"],
        "price": _px(int(TAPE["c"][i])),
        "cursor": int(TAPE["ts"][i]),
    }


def scan_exit(side: str, sl: float | None, tp: float | None, after: int, until: int) -> dict | None:
    """Walk one-second bars after the fill. The nearer side of the bar wins; a tie is the stop."""
    assert TAPE is not None
    if sl is None and tp is None:
        return None
    ts = TAPE["ts"]
    i0 = int(np.searchsorted(ts, after, side="right"))
    i1 = int(np.searchsorted(ts, until, side="right"))
    if i1 <= i0:
        return None
    slc = TAPE[i0:i1]
    o = slc["o"].astype(np.int32)
    h = slc["h"].astype(np.int32)
    l = slc["l"].astype(np.int32)
    sl_t = int(round(sl / TICK)) if sl is not None else None
    tp_t = int(round(tp / TICK)) if tp is not None else None
    long = side == "long"
    if long:
        sl_hit = (l <= sl_t) if sl_t is not None else np.zeros(len(slc), dtype=bool)
        tp_hit = (h >= tp_t) if tp_t is not None else np.zeros(len(slc), dtype=bool)
    else:
        sl_hit = (h >= sl_t) if sl_t is not None else np.zeros(len(slc), dtype=bool)
        tp_hit = (l <= tp_t) if tp_t is not None else np.zeros(len(slc), dtype=bool)
    either = sl_hit | tp_hit
    if not either.any():
        return None
    k = int(np.argmax(either))
    stop_first = bool(sl_hit[k]) and (
        not bool(tp_hit[k])
        or (long and abs(int(o[k]) - sl_t) <= abs(int(o[k]) - tp_t))
        or ((not long) and abs(int(o[k]) - sl_t) <= abs(int(o[k]) - tp_t))
    )
    if stop_first:
        if long:
            fill = int(o[k]) if int(o[k]) < sl_t else sl_t
            fill -= 1  # one tick of slippage
        else:
            fill = int(o[k]) if int(o[k]) > sl_t else sl_t
            fill += 1
        reason = "stop"
    else:
        if long:
            fill = int(o[k]) if int(o[k]) > tp_t else tp_t
        else:
            fill = int(o[k]) if int(o[k]) < tp_t else tp_t
        reason = "target"
    return {"time": int(slc["ts"][k]), "price": _px(fill), "reason": reason}


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n))

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if url.path == "/api/meta":
            assert META is not None
            self._json(
                {
                    "tick": META["tick"],
                    "point_value": META["point_value"],
                    "first": META["first"],
                    "last": META["last"],
                    "rolls": META["rolls"],
                    "sessions": META["sessions"],
                    "rows": META["rows"],
                    "built": META["built"],
                    "rule": META["rule"],
                    "stats": META["stats"],
                    "timeframes": list(TIMEFRAMES),
                    "days": [
                        {
                            "date": d["date"],
                            "symbol": d["symbol"],
                            "start_ts": d["start_ts"],
                            "roll": d["roll"],
                            "rth_ts": d.get("rth_ts"),
                        }
                        for d in META["days"]
                    ],
                }
            )
            return
        if url.path == "/api/bars":
            tf = TIMEFRAMES.get(q.get("tf", ["1m"])[0], 60)
            end = int(q.get("end", ["0"])[0])
            count = min(8000, max(10, int(q.get("count", ["400"])[0])))
            self._json(bars_ending(end, tf, count))
            return
        if url.path == "/api/step":
            tf = TIMEFRAMES.get(q.get("tf", ["1m"])[0], 60)
            cursor = int(q.get("cursor", ["0"])[0])
            direction = 1 if int(q.get("dir", ["1"])[0]) >= 0 else -1
            n = min(200, max(1, int(q.get("n", ["1"])[0])))
            nxt = cursor
            for _ in range(n):
                hopped = step_cursor(nxt, tf, direction)
                if hopped == nxt:
                    break
                nxt = hopped
            self._json({"cursor": nxt, **cursor_info(nxt)})
            return
        if url.path == "/api/info":
            cursor = int(q.get("cursor", ["0"])[0])
            self._json(cursor_info(cursor))
            return
        if url.path == "/api/journal":
            self._json(_read_json(JOURNAL, []))
            return
        if url.path == "/api/drawings":
            self._json(_read_json(DRAWINGS, []))
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/scan":
            b = self._body()
            hit = scan_exit(
                b.get("side", "long"),
                b.get("sl"),
                b.get("tp"),
                int(b.get("after", 0)),
                int(b.get("until", 0)),
            )
            self._json({"hit": hit})
            return
        if url.path == "/api/journal":
            _write_json(JOURNAL, self._body().get("trades", []))
            self._json({"ok": True})
            return
        if url.path == "/api/drawings":
            _write_json(DRAWINGS, self._body().get("drawings", []))
            self._json({"ok": True})
            return
        self._json({"error": "not found"}, 404)


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    has_front = BIN_PATH.exists() and META_PATH.exists()
    has_raw = any(DATA.glob("*.dbn.zst"))
    if rebuild or (has_raw and not has_front):
        build()
    load()
    assert META is not None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"http://127.0.0.1:{PORT} is already running")
    print(
        f"NQ tape  {META['rows']:,} seconds  {META['sessions']} sessions  {META['rolls']} rolls",
        flush=True,
    )
    print(f"http://127.0.0.1:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
