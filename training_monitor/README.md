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

Run the status summary from the repository root:

```bash
./scripts/run_in_env.sh python scripts/show_bcenet_training_status.py
```
