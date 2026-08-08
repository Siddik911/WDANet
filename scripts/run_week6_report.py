#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W4 = ROOT / "outputs/week4_cross_subject"
W5 = ROOT / "outputs/week5_cross_dataset"
OUT = ROOT / "outputs/week6_report"
OUT.mkdir(parents=True, exist_ok=True)


def _collect(path: Path):
    vals = []
    for fp in sorted(path.glob("*.json")):
        obj = json.loads(fp.read_text())
        vals.append(obj["final"]["L_total"])
    return vals


w4 = _collect(W4)
w5 = _collect(W5)

report = {
    "week4": {
        "n": len(w4),
        "mean_L_total": statistics.mean(w4) if w4 else None,
        "std_L_total": statistics.pstdev(w4) if len(w4) > 1 else None,
    },
    "week5": {
        "n": len(w5),
        "mean_L_total": statistics.mean(w5) if w5 else None,
        "std_L_total": statistics.pstdev(w5) if len(w5) > 1 else None,
    },
}

(OUT / "summary.json").write_text(json.dumps(report, indent=2))

md = ["# WDANet Week 6 Report", "", "## Summary"]
for k in ["week4", "week5"]:
    md.append(f"- **{k}**: n={report[k]['n']}, mean_L_total={report[k]['mean_L_total']}, std_L_total={report[k]['std_L_total']}")
(OUT / "report.md").write_text("\n".join(md) + "\n")
print(f"Wrote report to {OUT}")
