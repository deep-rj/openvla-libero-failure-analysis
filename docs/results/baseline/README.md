# Baseline: LIBERO-10, seed 0, 250 episodes

`openvla/openvla-7b-finetuned-libero-10`, tasks 0–9 × init states 0–24, logger and video on.
Run on RTX 4090 pods at harness `6ee2214` (seven shards, merged locally); raw traces are not in
git. Interpretation is in `docs/block1_findings.md`.

| File | Produced by | Code |
|---|---|---|
| `manifest.csv` | the runner, one row per episode | `6ee2214` |
| `run_config.json`, `merged_from.json` | `scripts/merge_runs.py` over the seven shards | `6ee2214` |
| `summary.md`, `summary.json` | `scripts/summarize_run.py outputs/baseline` | `6ee2214` |
| `pilot_report.md`, `pilot_report.json` | `scripts/pilot_report.py outputs/baseline --stage-a <stage_a_report.json>` | `5952203` |
| `instrument_checks.md`, `instrument_checks.json` | `scripts/instrument_checks.py outputs/baseline` | `5952203` |

The pilot report's anomalies section comes from the Stage A run's `stage_a_report.json`, not
from these 250 episodes; T3 and T5 over all 250 are in `instrument_checks`.
