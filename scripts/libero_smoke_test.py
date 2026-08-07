# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# %%
def run_single_action(env, visualize=False, flip=True):
    env.reset()
    env.set_init_state(init_states[0])
    rng = np.random.default_rng()
    random_action = rng.uniform(-1.0, 1.0, 7)
    print(f"Sampled action: {random_action}")
    obs, reward, done, info = env.step(random_action)

    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        if flip:
            # Flip the images upside-down. It seems MuJoCo's off-screen renderer has this behaviour
            axes[0].imshow(obs["agentview_image"][::-1])
            axes[0].set_title("agentview")
            axes[1].imshow(obs["robot0_eye_in_hand_image"][::-1])
            axes[1].set_title("wrist cam")
        else:
            axes[0].imshow(obs["agentview_image"])
            axes[0].set_title("agentview")
            axes[1].imshow(obs["robot0_eye_in_hand_image"])
            axes[1].set_title("wrist cam")
        plt.show()


def run_random_sim(env, visualize=False, flip=True):
    agent_frames = []
    wrist_frames = []
    env.reset()
    obs = env.set_init_state(init_states[0])
    rng = np.random.default_rng()
    # Run simulation for 30 steps
    for _ in range(30):
        action = rng.uniform(env.env.action_spec[0], env.env.action_spec[1])
        obs, reward, done, info = env.step(action)
        agent_img = obs["agentview_image"]
        wrist_img = obs["robot0_eye_in_hand_image"]
        if flip:
            # Flip the images upside-down. It seems MuJoCo's off-screen renderer has this behaviour
            agent_img = agent_img[::-1]
            wrist_img = wrist_img[::-1]
        agent_frames.append(agent_img)
        wrist_frames.append(wrist_img)

    if visualize:
        fig, axes = plt.subplots(2, 6, figsize=(15, 6))
        for i in range(6):
            axes[0, i].imshow(agent_frames[i * 5])  # Sample every 5th frame
            axes[0, i].axis("off")
            axes[1, i].imshow(wrist_frames[i * 5])
            axes[1, i].axis("off")
        axes[0, 0].set_title("agentview", loc="left")
        axes[1, 0].set_title("wrist cam", loc="left")
        plt.show()


# %%
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"
task_suite = benchmark_dict[task_suite_name]()

task_id = 0
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)

# %%
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128,
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()

init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

# %%
print("Running a single random action")
run_single_action(env)

# %%
print("Running a sequence of random actions")
run_random_sim(env)

# %%
env.close()
print("Smoke test completed successfully!")
