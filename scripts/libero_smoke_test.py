from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
