"""E2/E3 (placement evidence and speed): each case in a fresh process; the parent polls
the ANE driver power state at about 20 Hz and keeps the polls inside the child's loop
window. Gaps between cases let the ANE power down. Battery guard before every case."""
import json, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

PY, HERE = sys.executable, Path(__file__).parent
SECONDS = 3.0
CASES = [  # label, package or 'mlx', units, batch, hidden
    ("idle (child loads nothing)", "idle", "-", 1, 0),
    ("router b1 CPU_ONLY", "router_b1", "cpu", 1, 2560),
    ("router b1 CPU_AND_NE", "router_b1", "ane", 1, 2560),
    ("router b1 CPU_AND_GPU", "router_b1", "gpu", 1, 2560),
    ("router b256 CPU_ONLY", "router_b256", "cpu", 256, 2560),
    ("router b256 CPU_AND_NE", "router_b256", "ane", 256, 2560),
    ("router b256 CPU_AND_GPU", "router_b256", "gpu", 256, 2560),
    ("matmul 512x2048x2048 CPU_ONLY", "matmul_b512_h2048", "cpu", 512, 2048),
    ("matmul 512x2048x2048 CPU_AND_NE", "matmul_b512_h2048", "ane", 512, 2048),
    ("matmul 512x2048x2048 CPU_AND_GPU", "matmul_b512_h2048", "gpu", 512, 2048),
    ("mlx fp16 matmul 512x2048x2048", "mlx", "-", 512, 2048),
]
for name, kind, batch, hidden in (("matmul_b512_h2048", "matmul", 512, 2048),):
    if not (C.OUT / f"{name}.mlpackage").exists():
        C.build_model(name, kind, batch, hidden=hidden)

polls, stop = [], threading.Event()


def poller():
    while not stop.is_set():
        polls.append((time.time(), C.ane_power_state()))


threading.Thread(target=poller, daemon=True).start()
rows = []
for label, pkg, units, batch, hidden in CASES:
    battery = C.guard()
    waited = time.time()
    while time.time() - waited < 60:                   # wait until the ANE has powered down
        recent = [s for t, s in polls if t > time.time() - 2.0]
        if len(recent) > 10 and not any(recent):
            break
        time.sleep(0.5)
    settled = [s for t, s in polls if t > time.time() - 1.5]
    waited = time.time() - waited
    if pkg == "idle":
        time.sleep(SECONDS); window = [s for t, s in polls if t > time.time() - SECONDS]
        rows.append({"case": label, "ane_up_fraction": sum(window) / max(1, len(window)),
                     "settled_before_up_fraction": sum(settled) / max(1, len(settled)), "wait_s": waited, "battery": battery})
        print(rows[-1], flush=True); continue
    target = pkg if pkg == "mlx" else str(C.OUT / f"{pkg}.mlpackage")
    out = subprocess.run([PY, str(HERE / "e2_child.py"), target, units, str(SECONDS), str(batch),
                          str(hidden)], capture_output=True, text=True, timeout=300)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        rows.append({"case": label, "error": out.stderr[-400:]}); print(rows[-1]); continue
    row = json.loads(line[7:])
    window = [s for t, s in polls if row["loop_start"] <= t <= row["loop_end"] and s is not None]
    row.update(case=label, ane_up_fraction=(sum(window) / len(window)) if window else None,
               polls_in_loop=len(window), ane_up_before=sum(settled) / max(1, len(settled)), wait_s=waited, battery=battery)
    rows.append(row)
    print(f"{label:36s} p50 {row['p50_ms']:8.3f} ms  p95 {row['p95_ms']:8.3f}  "
          f"ANE up {row['ane_up_fraction']}  polls {len(window)}  wait {waited:.0f}s  "
          f"neural MB after load {row['neural_after_load'] / 1e6:.2f}  "
          f"RSS +{(row['rss_after_load'] - row['rss_before']) / 1e6:.0f} MB", flush=True)
stop.set()
(C.OUT / "e2_verify.json").write_text(json.dumps(rows, indent=1))
