"""Combines run directories written by sharded pods into one run directory."""

import itertools
import json
import shutil
from pathlib import Path

from harness import manifest, summary

# Shards legitimately differ in what they ran and where they wrote it.
_SHARD_SPECIFIC = {"out_dir", "tasks", "init_states", "allow_dirty"}


class MergeError(ValueError):
    pass


def _read_shard(path) -> dict:
    path = Path(path)
    config_path = path / "run_config.json"
    if not config_path.exists():
        raise MergeError(f"{path} has no run_config.json")
    record = json.loads(config_path.read_text())
    return {
        "dir": path,
        "config": record["config"],
        "environment": record.get("environment", {}),
        "rows": manifest.read_rows(path / "manifest.csv"),
    }


def _shared_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if k not in _SHARD_SPECIFIC}


def _check_compatible(shards: list[dict], allow_mixed_commits: bool) -> None:
    first = _shared_config(shards[0]["config"])
    for shard in shards[1:]:
        other = _shared_config(shard["config"])
        differing = sorted(k for k in first.keys() | other.keys() if first.get(k) != other.get(k))
        if differing:
            raise MergeError(f"{shard['dir']} differs from {shards[0]['dir']} in: {differing}")
    shas = {row["git_sha"] for shard in shards for row in shard["rows"]}
    if len(shas) > 1 and not allow_mixed_commits:
        raise MergeError(f"episodes come from different commits: {sorted(shas)}")


def _episode_files(shard: dict, row: dict) -> list[tuple[Path, str]]:
    """(source, path relative to the run directory) for every file an episode owns."""
    eid = row["episode_id"]
    files = [(row["episode_file"], True), (f"actions/{eid}.npy", True)]
    if row["video_file"]:
        files.append((row["video_file"], True))
    files.append((f"parity/{eid}.json", False))
    return [
        (shard["dir"] / rel, rel)
        for rel, required in files
        if required or (shard["dir"] / rel).exists()
    ]


def _check_complete(shards: list[dict]) -> None:
    for shard in shards:
        for row in shard["rows"]:
            for source, rel in _episode_files(shard, row):
                if not source.exists():
                    raise MergeError(f"{shard['dir']}: {rel} is missing")


def merge_runs(
    shard_dirs,
    out_dir,
    *,
    expected_tasks=None,
    expected_init_states=None,
    allow_mixed_commits: bool = False,
) -> dict:
    """Merges shards into `out_dir`; sources are only read. Validates before writing anything."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise MergeError(f"{out} is not empty")
    shards = [_read_shard(d) for d in shard_dirs]
    if not shards:
        raise MergeError("no shards given")

    _check_compatible(shards, allow_mixed_commits)
    ids = [row["episode_id"] for shard in shards for row in shard["rows"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise MergeError(f"duplicate episodes across shards: {duplicates}")
    _check_complete(shards)

    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (row for shard in shards for row in shard["rows"]),
        key=lambda r: (r["task_id"], r["init_state_idx"], r["episode_id"]),
    )
    for shard in shards:
        for row in shard["rows"]:
            for source, rel in _episode_files(shard, row):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, out / rel)
    for row in rows:
        manifest.append_row(out / "manifest.csv", row)

    config = _shared_config(shards[0]["config"])
    config["tasks"] = sorted({r["task_id"] for r in rows})
    config["init_states"] = sorted({r["init_state_idx"] for r in rows})
    (out / "run_config.json").write_text(
        json.dumps({"config": config, "environment": {"merged_shards": len(shards)}}, indent=2)
    )
    provenance = [
        {
            "dir": str(shard["dir"]),
            "config": shard["config"],
            "environment": shard["environment"],
            "episode_ids": sorted(row["episode_id"] for row in shard["rows"]),
        }
        for shard in shards
    ]
    (out / "merged_from.json").write_text(json.dumps({"shards": provenance}, indent=2))

    present = {(r["task_id"], r["init_state_idx"]) for r in rows}
    expected = (
        set(itertools.product(expected_tasks, expected_init_states))
        if expected_tasks is not None and expected_init_states is not None
        else set()
    )
    return {
        "n_episodes": len(rows),
        "missing": sorted(expected - present),
        "out_dir": str(out),
        "summary": summary.write_outputs(out),
    }
