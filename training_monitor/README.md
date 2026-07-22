# Current BCE-Net training

`current` is a read-only symbolic link to the active model output directory.
VS Code Explorer can open the following files without adding
`/home/work/models` as a workspace root:

- `current/training-curves.png`
- `current/qualitative/`
- `current/train.log`
- `current/metrics.jsonl`
- `loss_diagnostic_report.html` (snapshot report for the current loss diagnosis)

The link reflects newly created files automatically. If VS Code does not
refresh the directory, use **Explorer: Refresh** from the Command Palette.
For the HTML report, use **Open with Default Browser** or an installed
HTML-preview extension.

Independent test outputs use a separate `test-*` link so that `current` keeps
pointing to the frozen training run. Each test link is read-only in normal use
and points to a distinct output directory outside the repository.

Two-stage pilot audit packages also use `test-*` links. Their `README.md`
describes how to review GT first, reveal predictions second, and return the
completed CSV without modifying source labels.

`test-split-overlap-audit-20260722` contains the later file/group/pixel overlap
audit. It shows that the v1 manifest is file-disjoint but not strictly
spatial-disjoint, so its report must be read before treating test metrics as
spatial generalization evidence.

`test-audit-full100-v1-20260722` is the canonical combined audit package. Use
its `full100_review.csv`; it contains the pilot P IDs and remaining R IDs in a
single portable package.

Spatial split v2 and its retrained robust baseline use separate links:

- `spatial-v2-audit-20260722`: cross-split source/crop overlap audit (all zero)
- `robust-spatial-v2-20260722`: 100-epoch training, best epoch 46
- `test-spatial-v2-center512-best-e46-20260722`: frozen center-512 test result
- `test-audit-full100-spatial-v2-20260722`: v2 test 100장 전체 정성 검수,
  정량표, lossless source crop, dropdown 기반 `review_form.html`, 로컬 갤러리

The v2 test macro omission/excess F1 is `0.722276`. This is the robust
baseline control for later Formula (7) comparisons, not an exact paper
reproduction result.

Run the status summary from the repository root:

```bash
./scripts/run_in_env.sh python scripts/show_bcenet_training_status.py
```
