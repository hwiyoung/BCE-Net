#!/usr/bin/env python3
"""Show the current status of a BCE-Net training output directory."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="/home/work/models/BCE-Net/map-ortho-robust-v2-20260720",
    )
    parser.add_argument("--tail", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    pid_path = output_dir / "train.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    running = process_is_running(pid) if pid is not None else False
    metrics_path = output_dir / "metrics.jsonl"
    records = read_jsonl(metrics_path)
    record = records[-1] if records else None
    config_path = output_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )

    summary: dict[str, object] = {
        "output_dir": str(output_dir),
        "pid": pid,
        "running": running,
        "checkpoint_last": (output_dir / "checkpoint-last.pth").exists(),
        "checkpoint_best": (output_dir / "checkpoint-best.pth").exists(),
    }
    if record is not None:
        summary.update(
            {
                "completed_epoch": int(record["epoch"]) + 1,
                "total_epochs": config.get("epochs"),
                "lr": record["lr"],
                "train_loss": record["train"]["losses"]["loss"],
                "val_loss": record["val"]["losses"]["loss"],
                "val_change_f1": record["val"]["metrics"]["change"]["f1"],
                "val_macro_change_f1": record.get(
                    "val_macro_change_f1",
                    (
                        record["val"]["metrics"]["omission"]["f1"]
                        + record["val"]["metrics"]["excess"]["f1"]
                    )
                    / 2.0,
                ),
                "val_omission_f1": record["val"]["metrics"]["omission"]["f1"],
                "val_excess_f1": record["val"]["metrics"]["excess"]["f1"],
                "omission_prediction_rate": record["val"]["metrics"][
                    "omission"
                ].get("prediction_rate"),
                "excess_prediction_rate": record["val"]["metrics"]["excess"].get(
                    "prediction_rate"
                ),
            }
        )
        scores = [
            item.get(
                "selection_score",
                item["val"]["metrics"]["change"]["f1"],
            )
            for item in records
        ]
        summary["best_selection_score"] = max(scores)
        total_epochs = config.get("epochs")
        if total_epochs and running:
            average_seconds = sum(
                float(item.get("seconds", 0.0)) for item in records[-5:]
            ) / min(5, len(records))
            remaining = max(0, int(total_epochs) - int(record["epoch"]) - 1)
            summary["estimated_remaining"] = str(
                timedelta(seconds=round(average_seconds * remaining))
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    log_lines = tail(output_dir / "train.log", max(0, args.tail))
    if log_lines:
        print("\nRecent log:")
        print("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
