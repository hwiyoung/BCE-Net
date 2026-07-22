#!/usr/bin/env python3
"""Compare frozen test metrics across spatial-overlap diagnostic subsets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/map_ortho_manifest.csv")
    parser.add_argument("--per-sample-metrics", required=True)
    parser.add_argument("--overlap-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rect(row: Any, offset: int, size: int) -> tuple[int, int, int, int]:
    x0 = int(row.xoff) + offset
    y0 = int(row.yoff) + offset
    return x0, y0, x0 + size, y0 + size


def intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(
        a[1], b[1]
    )


def affected_samples(
    frame: pd.DataFrame,
    split_a: str,
    split_b: str,
    *,
    offset_a: int,
    size_a: int,
    offset_b: int,
    size_b: int,
) -> set[str]:
    rows_a = list(frame[frame["split"] == split_a].itertuples(index=False))
    rows_b = list(frame[frame["split"] == split_b].itertuples(index=False))
    return {
        str(row_b.sample_id)
        for row_b in rows_b
        if any(
            intersects(
                rect(row_a, offset_a, size_a),
                rect(row_b, offset_b, size_b),
            )
            for row_a in rows_a
        )
    }


def aggregate(
    sample_ids: set[str], rows_by_id: dict[str, dict[str, str]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(sample_ids)}
    for class_name in ("omission", "excess", "combined"):
        tp = sum(int(rows_by_id[sample_id][f"{class_name}_tp"]) for sample_id in sample_ids)
        fp = sum(int(rows_by_id[sample_id][f"{class_name}_fp"]) for sample_id in sample_ids)
        fn = sum(int(rows_by_id[sample_id][f"{class_name}_fn"]) for sample_id in sample_ids)
        total = sum(
            int(rows_by_id[sample_id][f"{class_name}_total_pixels"])
            for sample_id in sample_ids
        )
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        result[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": tp / max(1, tp + fp + fn),
            "prediction_rate": (tp + fp) / max(1, total),
            "target_rate": (tp + fn) / max(1, total),
        }
    result["macro_omission_excess_f1"] = (
        result["omission"]["f1"] + result["excess"]["f1"]
    ) / 2.0
    return result


def report(results: dict[str, Any]) -> str:
    rows = []
    for name in (
        "all100",
        "source_overlap24",
        "source_nonoverlap76",
        "possible_eval_overlap8",
        "no_possible_eval_overlap92",
    ):
        item = results[name]
        rows.append(
            f"| {name} | {item['samples']} | "
            f"{item['omission']['f1']:.6f} | {item['excess']['f1']:.6f} | "
            f"{item['macro_omission_excess_f1']:.6f} | "
            f"{item['combined']['f1']:.6f} | "
            f"{item['combined']['target_rate']:.6f} |"
        )
    return """# Frozen test spatial-overlap subset metrics

이 표는 동일한 frozen prediction과 threshold 0.5를 overlap 기준으로 사후
분할해 재집계한 진단 결과다. 모델이나 threshold 선택에는 사용하지 않았다.

| subset | samples | omission F1 | excess F1 | macro F1 | combined F1 | combined target rate |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

`source_overlap24`는 test의 1024 source가 train 또는 validation source와
겹치는 경우다. `possible_eval_overlap8`은 train crop이 도달할 수 있는
union 또는 validation 중앙 crop이 test 중앙 평가 crop과 겹치는 경우다.
실제 epoch별 train jitter origin은 기록되지 않아 train 쪽은 잠재 중복이다.

중첩 subset의 F1이 더 높지만 combined target rate와 공간/클래스 구성이
동시에 다르므로 차이를 leakage의 인과 효과로 해석할 수 없다. 다만 현재
all-100 metric이 strict spatial generalization 근거로 안전하지 않다는
판정은 바뀌지 않는다. 정확한 영향 측정에는 buffered split v2에서 baseline을
다시 학습해야 한다.
"""


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    metrics_path = Path(args.per_sample_metrics).resolve()
    pairs_path = Path(args.overlap_pairs).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (manifest, metrics_path, pairs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    json_path = output_dir / "subset_metrics.json"
    report_path = output_dir / "SUBSET_METRICS.md"
    if json_path.exists() or report_path.exists():
        raise FileExistsError("Subset metric outputs already exist")

    frame = pd.read_csv(manifest)
    metric_rows = read_csv(metrics_path)
    rows_by_id = {row["sample_id"]: row for row in metric_rows}
    test_ids = set(rows_by_id)
    overlap_pairs = read_csv(pairs_path)
    source_overlap = {
        row["sample_b"]
        for row in overlap_pairs
        if row["split_b"] == "test" and row["split_a"] in {"train", "val"}
    }
    possible_train_eval = affected_samples(
        frame,
        "train",
        "test",
        offset_a=128,
        size_a=768,
        offset_b=256,
        size_b=512,
    )
    direct_val_eval = affected_samples(
        frame,
        "val",
        "test",
        offset_a=256,
        size_a=512,
        offset_b=256,
        size_b=512,
    )
    possible_eval = possible_train_eval | direct_val_eval
    if len(test_ids) != 100 or len(source_overlap) != 24 or len(possible_eval) != 8:
        raise ValueError(
            "Unexpected subset sizes: "
            f"test={len(test_ids)}, source={len(source_overlap)}, "
            f"eval={len(possible_eval)}"
        )
    results = {
        "all100": aggregate(test_ids, rows_by_id),
        "source_overlap24": aggregate(source_overlap, rows_by_id),
        "source_nonoverlap76": aggregate(test_ids - source_overlap, rows_by_id),
        "possible_eval_overlap8": aggregate(possible_eval, rows_by_id),
        "no_possible_eval_overlap92": aggregate(test_ids - possible_eval, rows_by_id),
        "definitions": {
            "source_overlap_test_ids": sorted(source_overlap),
            "possible_train_eval_overlap_test_ids": sorted(possible_train_eval),
            "direct_val_eval_overlap_test_ids": sorted(direct_val_eval),
        },
        "causal_interpretation": False,
    }
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report(results), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
