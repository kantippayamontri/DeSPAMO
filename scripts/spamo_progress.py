"""Read-only progress view for tools/spamo_repro runs.

Usage: python scripts/spamo_progress.py [RUN_ROOT]
"""

import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT = "/home/kan/datasets/despamo/clip-s2-own-features-spamo-ckpt-recipe-200ep-seed0-v1"


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def alive(pid) -> bool:
    return bool(pid) and Path(f"/proc/{pid}").exists()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    approval = read(root / "approval.json") or {}
    parent, child, state = (
        read(root / "parent.json"),
        read(root / "child.json"),
        read(root / "state.json"),
    )
    max_epochs = (
        200
        if "200 epochs" in approval.get("scope", "")
        else (500 if "500" in approval.get("scope", "") else 40)
    )
    total = max_epochs * 444
    now = time.time()
    print(f"run     {root.name}")
    if parent:
        hours = (now - parent["started"]) / 3600
        status = "ALIVE" if alive(parent["pid"]) else "STOPPED"
        print(f"parent  pid {parent['pid']} {status}  elapsed {hours:.1f} h")
    if child:
        if "exit_code" in child:
            cstat = f"exited {child['exit_code']}"
        else:
            cstat = "running" if alive(child["pid"]) else "DEAD (no exit record)"
        mins = (now - child.get("started", now)) / 60
        print(f"worker  {child['stage']:<5} epoch {child['epoch']}  {cstat}  {mins:.0f} min")
    log = root / "train-log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    updates = [r for r in rows if "optimizer_update" in r]
    if updates:
        u = updates[-1]
        n = u["optimizer_update"]
        ce = "-" if u["CE"] is None else f"{u['CE']:.3f}"
        done = max((r["epoch"] for r in rows if "published" in r), default=0)
        print(
            f"train   epoch {done}/{max_epochs} done  update {n}/{total} ({100 * n / total:.1f}%)"
            f"  lr {u['lr']:.2e}  CE {ce}  VT {u['VT']:.3f}"
        )
    devs = (
        sorted((root / "dev").glob("dev-epoch-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
        if (root / "dev").exists()
        else []
    )
    if devs:
        print("\nepoch   BLEU1   BLEU4  ROUGE-L")
        best = max(devs, key=lambda p: read(p)["metrics"]["bleu4"])
        for p in devs[-15:]:
            d = read(p)
            m = d["metrics"]
            mark = "  <- best" if p == best else ""
            row = f"{d['epoch']:5d}  {m['bleu1']:6.2f}  {m['bleu4']:6.2f}  {m['rougeL_f1']:.4f}"
            print(row + mark)
        if len(devs) > 15:
            print(f"  ({len(devs) - 15} earlier checks hidden)")
    if state and state.get("history"):
        hist = [h["bleu4"] for h in state["history"]]
        since = len(hist) - 1 - max(range(len(hist)), key=hist.__getitem__)
        print(
            f"\nbest    epoch {state['best']['epoch']} dev BLEU4 {state['best']['bleu4']:.3f}"
            f"  checks since best {since}/50  status {state['status']}"
            f"{'  ' + state['stop_reason'] if state.get('stop_reason') else ''}"
        )
    for name in ("manual-stop.json", "operator-stop.json"):
        if (root / name).exists():
            print(f"STOP    {name}: {read(root / name)}")
    tests = sorted((root / "test").glob("test-epoch-*.json")) if (root / "test").exists() else []
    for p in tests:
        d = read(p)
        print(f"test    epoch {d['epoch']} BLEU4 {d['metrics']['bleu4']:.2f}  ({d['clips']} clips)")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"gpu     {gpu}")


if __name__ == "__main__":
    main()
