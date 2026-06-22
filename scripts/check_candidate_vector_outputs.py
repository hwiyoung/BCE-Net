#!/usr/bin/env python3
"""Check BCE-Net candidate vector GeoPackage schema and geometry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd


LAYERS = ["building_missing_candidates", "building_excess_candidates"]
BASE_COLUMNS = [
    "candidate_id",
    "candidate_type",
    "model_output",
    "confidence_mean",
    "confidence_p90",
    "confidence_max",
    "area_m2",
    "source_tile",
    "review_status",
    "review_comment",
    "is_synthetic",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpkg", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--require-non-empty", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpkg = Path(args.gpkg).resolve()
    summary_json = Path(args.summary_json).resolve()
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    layer_reports: dict[str, Any] = {}

    if not gpkg.exists():
        failures.append({"error": "missing GeoPackage", "path": str(gpkg)})
        layers = []
    else:
        layers = list(fiona.listlayers(gpkg))

    if not summary_json.exists():
        failures.append({"error": "missing summary JSON", "path": str(summary_json)})

    total = 0
    for layer in LAYERS:
        if layer not in layers:
            failures.append({"layer": layer, "error": "missing layer"})
            continue
        gdf = gpd.read_file(gpkg, layer=layer)
        total += len(gdf)
        required = BASE_COLUMNS + (["old_building_id"] if layer == "building_excess_candidates" else [])
        missing = [column for column in required if column not in gdf.columns]
        invalid_types = sorted(set(gdf["candidate_type"].dropna().astype(str)) - {"MISSING", "EXCESS"}) if "candidate_type" in gdf else []
        bad_review = sorted(set(gdf["review_status"].dropna().astype(str)) - {"UNREVIEWED"}) if "review_status" in gdf else []
        valid_geometry = bool(gdf.geometry.is_valid.all()) if len(gdf) else True
        has_crs = gdf.crs is not None
        if missing:
            failures.append({"layer": layer, "error": "missing required columns", "columns": missing})
        if invalid_types:
            failures.append({"layer": layer, "error": "invalid candidate_type", "values": invalid_types})
        if bad_review:
            failures.append({"layer": layer, "error": "invalid review_status", "values": bad_review})
        if not valid_geometry:
            failures.append({"layer": layer, "error": "invalid geometry"})
        if not has_crs:
            failures.append({"layer": layer, "error": "missing CRS"})
        layer_reports[layer] = {
            "feature_count": int(len(gdf)),
            "crs": gdf.crs.to_string() if gdf.crs else None,
            "geometry_valid": valid_geometry,
            "required_columns_present": not missing,
            "candidate_types": sorted(set(gdf["candidate_type"].dropna().astype(str))) if "candidate_type" in gdf else [],
            "review_status_values": sorted(set(gdf["review_status"].dropna().astype(str))) if "review_status" in gdf else [],
            "columns": list(gdf.columns),
        }

    if args.require_non_empty and total == 0:
        failures.append({"error": "require-non-empty was true but total candidate count is 0"})

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "blocked",
        "gpkg": str(gpkg),
        "summary_json": str(summary_json),
        "layers": layers,
        "layer_reports": layer_reports,
        "total_candidate_count": total,
        "require_non_empty": args.require_non_empty,
        "failures": failures,
        "results_are_confirmed_errors": False,
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote candidate vector output check JSON: {out_json}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
