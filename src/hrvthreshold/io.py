"""Readers for RR-interval sources.

* plain text / CSV: one RR per line, or a delimited table with an RR column
  (auto-detected by a header containing "rr"; e.g. Polar Sensor Logger
  "Phone timestamp;RR-interval [ms]"). Seconds are auto-converted to ms.
* FIT (Garmin / COROS / Polar / Wahoo): beat-to-beat intervals live in
  ``hrv`` messages, field ``time`` = tuple of up to 5 RR values in seconds,
  with 65.535 (0xFFFF/1000) marking "invalid". Requires ``fitdecode``.
"""

from __future__ import annotations

import csv
import io as _io
from pathlib import Path

import numpy as np

FIT_INVALID_S = 65.535


def _to_ms(rr: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr, float)
    rr = rr[np.isfinite(rr)]
    if rr.size and np.nanmedian(rr) < 10:  # looks like seconds
        rr = rr * 1000.0
    return rr


def read_rr_text(path: str | Path, column: str | None = None) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: empty")
    first = lines[0]
    delim = max([";", ",", "\t"], key=first.count)
    has_header = any(c.isalpha() for c in first)
    if not has_header and first.count(delim) == 0:
        return _to_ms(np.array([float(x) for x in lines]))
    reader = csv.reader(_io.StringIO("\n".join(lines)), delimiter=delim)
    rows = list(reader)
    if has_header:
        header = [h.strip().lower() for h in rows[0]]
        if column is not None:
            idx = header.index(column.lower())
        else:
            cands = [i for i, h in enumerate(header) if "rr" in h or "ibi" in h]
            if not cands:
                raise ValueError(f"{path}: no RR/IBI column in header {header}")
            idx = cands[0]
        rows = rows[1:]
    else:
        idx = 0 if column is None else int(column)
    vals = []
    for r in rows:
        if idx < len(r) and r[idx].strip():
            # Polar Sensor Logger may pack several RRs in one cell separated by spaces
            vals.extend(float(v) for v in r[idx].replace(",", " ").split())
    return _to_ms(np.array(vals))


def read_fit(path: str | Path) -> np.ndarray:
    try:
        import fitdecode
    except ImportError as e:  # pragma: no cover
        raise ImportError("FIT support needs `pip install fitdecode`") from e
    rr: list[float] = []
    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "hrv":
                if not frame.has_field("time"):
                    continue
                vals = frame.get_value("time")
                if not isinstance(vals, (tuple, list)):
                    vals = (vals,)
                rr.extend(v for v in vals if v is not None and v < FIT_INVALID_S - 1e-6)
    if not rr:
        raise ValueError(f"{path}: no HRV (beat interval) messages found. "
                         "Enable 'Log HRV' / 'HRV recording' on the watch and use a chest strap.")
    return np.asarray(rr) * 1000.0


def read_rr(path: str | Path, column: str | None = None) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".fit":
        return read_fit(p)
    return read_rr_text(p, column)
