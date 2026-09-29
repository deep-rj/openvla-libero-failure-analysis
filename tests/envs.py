from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def build_env(task_id, suite_name="libero_10", camera_size=64):
    """Returns (env, obs, task label, init state 0) for a freshly reset LIBERO task."""
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size
    )
    env.reset()
    init_state = suite.get_task_init_states(task_id)[0]
    obs = env.set_init_state(init_state)
    return env, obs, task.language, init_state
