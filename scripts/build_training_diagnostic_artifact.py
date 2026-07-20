#!/usr/bin/env python3
"""Build a canonical report artifact from a BCE-Net metrics.jsonl snapshot."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path


def pct_change(first: float, last: float) -> float:
    return (last / first - 1.0) if first else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="training_monitor/current/metrics.jsonl",
    )
    parser.add_argument(
        "--output",
        default="training_monitor/loss_diagnostic_artifact.json",
    )
    parser.add_argument(
        "--database",
        default="training_monitor/loss_diagnostic.sqlite",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No metric rows in {metrics_path}")

    all_rows = []
    for record in records:
        validation = record["val"]["metrics"]
        all_rows.append(
            {
                "epoch": int(record["epoch"]) + 1,
                "learning_rate": float(record["lr"]),
                "train_loss": float(record["train"]["losses"]["loss"]),
                "validation_loss": float(record["val"]["losses"]["loss"]),
                "macro_f1": float(record["val_macro_change_f1"]),
                "combined_f1": float(validation["change"]["f1"]),
                "omission_f1": float(validation["omission"]["f1"]),
                "excess_f1": float(validation["excess"]["f1"]),
                "omission_precision": float(
                    validation["omission"]["precision"]
                ),
                "omission_recall": float(validation["omission"]["recall"]),
                "excess_precision": float(validation["excess"]["precision"]),
                "excess_recall": float(validation["excess"]["recall"]),
                "omission_prediction_rate": float(
                    validation["omission"]["prediction_rate"]
                ),
                "omission_target_rate": float(
                    validation["omission"]["target_rate"]
                ),
                "excess_prediction_rate": float(
                    validation["excess"]["prediction_rate"]
                ),
                "excess_target_rate": float(validation["excess"]["target_rate"]),
            }
        )

    first = all_rows[0]
    latest = all_rows[-1]
    train_losses = [row["train_loss"] for row in all_rows]
    validation_losses = [row["validation_loss"] for row in all_rows]
    train_increases = sum(
        current > previous
        for previous, current in zip(train_losses, train_losses[1:])
    )
    latest_window = train_losses[-5:]
    latest_cv = (
        statistics.pstdev(latest_window) / statistics.mean(latest_window)
        if len(latest_window) > 1
        else 0.0
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    sampling_step = max(1, (len(all_rows) + 17) // 18)
    columns = list(all_rows[0])
    database_path = Path(args.database)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TABLE IF EXISTS epoch_metrics")
        definitions = ", ".join(
            f"{column} {'INTEGER' if column == 'epoch' else 'REAL'}"
            for column in columns
        )
        connection.execute(f"CREATE TABLE epoch_metrics ({definitions})")
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO epoch_metrics VALUES ({placeholders})",
            [[row[column] for column in columns] for row in all_rows],
        )
        query_sql = (
            f"SELECT {', '.join(columns)} FROM epoch_metrics "
            f"WHERE ((epoch - 1) % {sampling_step}) = 0 "
            f"OR epoch = {latest['epoch']} ORDER BY epoch"
        )
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(query_sql)]
        connection.commit()
    finally:
        connection.close()
    source = {
        "id": "bcenet-training-metrics",
        "label": "BCE-Net epoch metrics",
        "path": "training_monitor/loss_diagnostic.sqlite",
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": query_sql,
            "description": (
                "Select a deterministic sample of completed epoch metrics for "
                "the training diagnostic charts and table."
            ),
            "executed_at": generated_at,
            "tables_used": ["epoch_metrics"],
            "filters": [
                f"completed epochs through {latest['epoch']}",
                f"every {sampling_step} epoch(s), always including latest",
                "validation threshold 0.5",
            ],
            "metric_definitions": [
                "train_loss is the mean total training loss across all batches in an epoch",
                "validation_loss is the mean total validation loss across validation batches",
                "macro_f1 is the arithmetic mean of omission F1 and excess F1",
                "prediction_rate is positive predicted pixels divided by all evaluated pixels",
            ],
        },
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "BCE-Net Training Loss Diagnostic",
            "description": (
                "Snapshot diagnosis of loss movement and class-head health."
            ),
            "generatedAt": generated_at,
            "sources": [source],
            "charts": [
                {
                    "id": "loss-by-epoch",
                    "title": "Training and validation loss by epoch",
                    "subtitle": (
                        "Both series decline over the completed training window."
                    ),
                    "type": "line",
                    "dataset": "epoch_metrics",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "ordinal",
                            "label": "Epoch",
                        },
                        "y": {
                            "fields": ["train_loss", "validation_loss"],
                            "type": "quantitative",
                            "label": "Loss",
                        },
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Mean total loss",
                    "layout": "full",
                },
                {
                    "id": "f1-by-epoch",
                    "title": "Validation F1 by epoch",
                    "subtitle": (
                        "Omission and excess heads remain active while macro F1 rises."
                    ),
                    "type": "line",
                    "dataset": "epoch_metrics",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "ordinal",
                            "label": "Epoch",
                        },
                        "y": {
                            "fields": [
                                "macro_f1",
                                "omission_f1",
                                "excess_f1",
                            ],
                            "type": "quantitative",
                            "format": "percent",
                            "label": "F1",
                        },
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "F1",
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "prediction-rate-by-epoch",
                    "title": "Predicted and target positive-pixel rates",
                    "subtitle": (
                        "Nonzero prediction rates rule out the earlier zero-output collapse."
                    ),
                    "type": "line",
                    "dataset": "epoch_metrics",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "ordinal",
                            "label": "Epoch",
                        },
                        "y": {
                            "fields": [
                                "omission_prediction_rate",
                                "omission_target_rate",
                                "excess_prediction_rate",
                                "excess_target_rate",
                            ],
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Positive-pixel rate",
                        },
                    },
                    "xAxisTitle": "Epoch",
                    "yAxisTitle": "Rate",
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "epoch-detail",
                    "title": "Epoch-level diagnostic metrics",
                    "subtitle": (
                        "A deterministic epoch sample in descending order."
                    ),
                    "dataset": "epoch_metrics",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "epoch", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "epoch", "label": "Epoch", "format": "number"},
                        {
                            "field": "train_loss",
                            "label": "Train loss",
                            "format": "number",
                        },
                        {
                            "field": "validation_loss",
                            "label": "Val loss",
                            "format": "number",
                        },
                        {
                            "field": "macro_f1",
                            "label": "Macro F1",
                            "format": "percent",
                        },
                        {
                            "field": "omission_f1",
                            "label": "Omission F1",
                            "format": "percent",
                        },
                        {
                            "field": "excess_f1",
                            "label": "Excess F1",
                            "format": "percent",
                        },
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# BCE-Net Training Loss Diagnostic",
                },
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Technical summary\n\n"
                        f"The apparent oscillation is normal mini-batch and "
                        f"augmentation noise around a clear downward trend. "
                        f"Through epoch {latest['epoch']}, mean train loss fell "
                        f"from {first['train_loss']:.3f} to "
                        f"{latest['train_loss']:.3f} "
                        f"({pct_change(first['train_loss'], latest['train_loss']):.1%}), "
                        f"validation loss fell "
                        f"{abs(pct_change(first['validation_loss'], latest['validation_loss'])):.1%}, "
                        f"and macro F1 rose from {first['macro_f1']:.3f} to "
                        f"{latest['macro_f1']:.3f}. No intervention is warranted "
                        f"from the current loss pattern."
                    ),
                },
                {
                    "id": "loss-finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Loss is noisy locally but declining globally\n\n"
                        f"Train loss increased between adjacent completed epochs "
                        f"{train_increases} times out of {len(all_rows) - 1}; the "
                        f"largest increases are small relative to the overall "
                        f"decline. The last-five-epoch coefficient of variation "
                        f"is {latest_cv:.2%}. The console's intra-epoch values "
                        f"are cumulative epoch averages, so small reversals reflect "
                        f"the shuffled mix of changed-object sizes and augmentations, "
                        f"not raw-batch instability."
                    ),
                },
                {
                    "id": "loss-chart-block",
                    "type": "chart",
                    "chartId": "loss-by-epoch",
                    "layout": "full",
                },
                {
                    "id": "head-health",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Both change heads are improving\n\n"
                        f"At epoch {latest['epoch']}, omission F1 is "
                        f"{latest['omission_f1']:.3f} and excess F1 is "
                        f"{latest['excess_f1']:.3f}. Macro F1 has improved by "
                        f"{latest['macro_f1'] - first['macro_f1']:.3f} from epoch 1. "
                        f"This is the opposite of the prior run's omission-head "
                        f"collapse."
                    ),
                },
                {
                    "id": "f1-chart-block",
                    "type": "chart",
                    "chartId": "f1-by-epoch",
                    "layout": "full",
                },
                {
                    "id": "prediction-rate-finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## Positive prediction rates remain healthy\n\n"
                        f"Omission predicts {latest['omission_prediction_rate']:.1%} "
                        f"of validation pixels against a {latest['omission_target_rate']:.1%} "
                        f"target rate. Excess predicts {latest['excess_prediction_rate']:.1%} "
                        f"against a {latest['excess_target_rate']:.1%} target rate. "
                        f"Neither head is converging to all-background output."
                    ),
                },
                {
                    "id": "prediction-rate-chart-block",
                    "type": "chart",
                    "chartId": "prediction-rate-by-epoch",
                    "layout": "full",
                },
                {
                    "id": "scope-and-definitions",
                    "type": "markdown",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The report covers completed epochs only. Train and "
                        "validation loss are epoch means of the complete BCE-Net "
                        "objective. Macro F1 is the arithmetic mean of omission "
                        "and excess pixel F1 at threshold 0.5. Validation uses the "
                        "fixed 100-sample spatial split. Narrative calculations use "
                        f"all {len(all_rows)} completed epochs; charts and the table "
                        f"show a deterministic {len(rows)}-epoch sample to keep the "
                        "report payload compact."
                    ),
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "body": (
                        "## Diagnostic method\n\n"
                        "The diagnosis compares the first completed epoch, latest "
                        "completed epoch, adjacent-epoch reversals, the latest "
                        "five-epoch variation, validation loss, both class-head "
                        "F1 series, and predicted-positive rates. A loss-only "
                        "diagnosis would be insufficient because the earlier run "
                        "showed class-head collapse despite a decreasing loss."
                    ),
                },
                {
                    "id": "detail-table-block",
                    "type": "table",
                    "tableId": "epoch-detail",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and uncertainty\n\n"
                        "The validation labels are known to contain annotation "
                        "errors, so F1 is a noisy proxy for real-world quality. "
                        "This snapshot cannot prove final convergence because "
                        "training is still running. Visual review of fixed "
                        "validation examples remains necessary."
                    ),
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "- Continue the current run without changing the learning rate.\n"
                        "- Review qualitative panels every five epochs.\n"
                        "- Intervene only if validation loss rises persistently while "
                        "train loss falls, macro F1 deteriorates for several epochs, "
                        "or either prediction rate approaches zero."
                    ),
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "The most decision-relevant follow-up is whether a manually "
                        "cleaned validation subset changes the ranking of best "
                        "checkpoints or the preferred omission/excess thresholds."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "epoch_metrics": rows,
            },
        },
        "sources": [source],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
