"""Print a per-task table (completion, precision, force MAE, time, drops) for an evaluate.sh output folder."""
import glob
import json
import os
import sys

out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    d = json.load(open(f))
    task = list(d["per_task"]["nominal"])[0]
    m = d["per_task"]["nominal"][task]
    drops = sum(1 for e in d["episodes"] if e["metrics"].get("dropped"))
    rows.append((task, m.get("completion") or 0.0, m.get("precision") or 0.0, m.get("force_mae_N") or 0.0,
                 m.get("time_s") or 0.0, drops))
print(f'{"task":20s} {"compl":>6} {"prec":>6} {"fMAE":>6} {"time":>6} drops')
for t, c, p, f, s, d in rows:
    print(f"{t:20s} {c:6.3f} {p:6.3f} {f:6.3f} {s:6.2f} {d}")
if rows:
    print(f"mean completion {sum(r[1] for r in rows) / len(rows):.3f}")
