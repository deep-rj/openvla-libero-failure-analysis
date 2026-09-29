"""Runs a set of (task, init state) episodes and writes traces, videos and a manifest."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from harness import manifest, metadata, schema, video
from harness import policy as policy_lib
from harness.rollout import MAX_STEPS_BY_SUITE, EpisodeSpec, run_episode
from harness.world import scene_layout

REPO_DIR = Path(__file__).resolve().parent.parent
_SIBLINGS = {"openvla": "openvla", "LIBERO": "LIBERO"}


@dataclass
class RunConfig:
    out_dir: Path
    tasks: list[int]
    init_states: list[int]
    suite: str = "libero_10"
    seed: int = 0
    policy: str = "stub"
    mode: str = "logged"
    max_steps: int | None = None
    video: bool = True
    allow_dirty: bool = False
    camera_size: int = 256
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    unnorm_key: str = "libero_10"
    center_crop: bool = True


def default_policy_factory(cfg: RunConfig):
    """Returns (policy, provenance) for `cfg.policy`."""
    if cfg.policy == "stub":
        return policy_lib.StubPolicy(seed=cfg.seed), {}
    if cfg.policy == "openvla":
        from harness.openvla_policy import OpenVLAPolicy, PolicyConfig

        loaded = OpenVLAPolicy.load(
            PolicyConfig(
                pretrained_checkpoint=cfg.checkpoint,
                unnorm_key=cfg.unnorm_key,
                center_crop=cfg.center_crop,
            )
        )
        return loaded, loaded.provenance()
    raise ValueError(f"unknown policy {cfg.policy!r}")


def build_env(suite, task_id: int, camera_size: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=camera_size, camera_widths=camera_size
    )
    env.seed(seed)
    return env, task


def _config_json(cfg: RunConfig, environment: dict) -> str:
    config = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    return json.dumps({"config": config, "environment": environment}, indent=2)


class ActionRecorder:
    """Env proxy that records the actions its `step` actually receives."""

    def __init__(self, env):
        self._env = env
        self.sent: list[np.ndarray] = []

    def step(self, action):
        self.sent.append(np.array(action, dtype=np.float64))
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)


def run(cfg: RunConfig, policy_factory=None) -> list[dict]:
    """Runs every episode not already in the manifest; returns the new manifest rows."""
    from libero.libero import benchmark

    siblings = {name: REPO_DIR.parent / dirname for name, dirname in _SIBLINGS.items()}
    environment = metadata.environment_info(REPO_DIR, siblings)
    if cfg.policy != "stub" and not cfg.allow_dirty and environment["git"]["harness"]["dirty"]:
        raise RuntimeError("harness working tree is dirty; commit first or pass allow_dirty")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(_config_json(cfg, environment))
    manifest_path = out / "manifest.csv"
    done = manifest.completed_episode_ids(manifest_path)

    active_policy, provenance = (policy_factory or default_policy_factory)(cfg)
    suite = benchmark.get_benchmark_dict()[cfg.suite]()
    max_steps = cfg.max_steps or MAX_STEPS_BY_SUITE[cfg.suite]
    logged = cfg.mode in ("logged", "parity")
    rows = []
    for task_id in cfg.tasks:
        specs = [EpisodeSpec(cfg.suite, task_id, i, cfg.seed) for i in cfg.init_states]
        todo = [s for s in specs if s.episode_id not in done]
        if not todo:
            continue
        env, task = build_env(suite, task_id, cfg.camera_size, cfg.seed)
        recorder = ActionRecorder(env)
        try:
            init_states = suite.get_task_init_states(task_id)
            for spec in todo:
                recorder.sent.clear()
                result = run_episode(
                    recorder,
                    active_policy,
                    task.language,
                    init_states[spec.init_state_idx],
                    max_steps=max_steps,
                    mode=cfg.mode,
                    record_frames=cfg.video and logged,
                )
                row = _write_outputs(
                    out, cfg, spec, task.language, recorder, result, active_policy, provenance,
                    max_steps, environment, np.stack(recorder.sent),
                )  # fmt: skip
                manifest.append_row(manifest_path, row)
                rows.append(row)
                print(
                    f"{spec.episode_id}: success={result.success} steps={result.num_policy_steps}"
                )
        finally:
            env.close()
    return rows


def _write_outputs(
    out, cfg, spec, task_label, env, result, active_policy, provenance, max_steps, environment, sent
):
    episode_file = video_file = ""
    (out / "actions").mkdir(exist_ok=True)
    np.save(out / "actions" / f"{spec.episode_id}.npy", sent)
    if cfg.mode == "parity":
        (out / "parity").mkdir(exist_ok=True)
        (out / "parity" / f"{spec.episode_id}.json").write_text(
            json.dumps({"mismatched_steps": result.parity_mismatches})
        )
    if cfg.mode != "reference":
        meta = metadata.episode_metadata(
            spec=spec,
            task_label=task_label,
            layout=scene_layout(env),
            result=result,
            mjcf_xml=env.sim.model.get_xml(),
            provenance=provenance,
            action_stats=active_policy.action_stats,
            max_steps=max_steps,
            checkpoint=cfg.checkpoint,
            unnorm_key=cfg.unnorm_key,
            center_crop=cfg.center_crop,
            environment=environment,
        )
        episode_file = f"episodes/{spec.episode_id}.h5"
        (out / "episodes").mkdir(exist_ok=True)
        schema.write_episode(out / episode_file, result.steps, meta)
        if cfg.video:
            video_file = f"videos/{spec.episode_id}.mp4"
            (out / "videos").mkdir(exist_ok=True)
            video.write_video(
                out / video_file, result.agent_frames, result.wrist_frames, result.is_wait_steps
            )

    return {
        "episode_id": spec.episode_id,
        "task_suite": spec.suite,
        "task_id": spec.task_id,
        "task_instruction": task_label,
        "init_state_idx": spec.init_state_idx,
        "seed": spec.seed,
        "success": result.success,
        "num_steps": result.num_policy_steps,
        "terminated_by": result.terminated_by,
        "episode_file": episode_file,
        "video_file": video_file,
        "git_sha": environment["git"]["harness"]["sha"],
        "seconds_per_step": result.wall_s / max(result.num_policy_steps, 1),
    }
