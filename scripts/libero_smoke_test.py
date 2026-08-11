# %%
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

REPO_DIR = os.environ["REPO_DIR"]


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


# %%
def run_random_sim(env, visualize=False, flip=True, save_path=None):
    agent_frames, wrist_frames = [], []
    actions, rewards, dones = [], [], []

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
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    if save_path is not None:
        with h5py.File(save_path, "w") as f:
            ep = f.create_group("episode_0")
            ep.create_dataset(
                "agentview_images", data=np.array(agent_frames), compression="gzip"
            )
            ep.create_dataset(
                "wrist_images", data=np.array(wrist_frames), compression="gzip"
            )
            ep.create_dataset("actions", data=np.array(actions))
            ep.create_dataset("rewards", data=np.array(rewards))
            ep.create_dataset("dones", data=np.array(dones))
            ep.attrs["success"] = bool(info.get("success", False))
            ep.attrs["num_steps"] = len(actions)

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

    # Run validations
    n_steps = len(actions)
    assert len(agent_frames) == n_steps
    assert len(wrist_frames) == n_steps
    assert len(rewards) == n_steps
    assert len(dones) == n_steps

    expected_shape = agent_frames[0].shape
    for img in agent_frames + wrist_frames:
        assert img.shape == expected_shape
        assert img.dtype == np.uint8

    assert np.all(np.isfinite(rewards)), "non-finite reward encountered"

    with h5py.File(save_path, "r") as f:
        ep = f["episode_0"]
        assert ep["actions"].shape == (n_steps, env.env.action_dim)
        assert ep.attrs["num_steps"] == n_steps


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
simulation_results_filepath = os.path.join(
    REPO_DIR, "outputs", "libero_smoke_test_results.h5"
)
run_random_sim(env, save_path=simulation_results_filepath)

# %%
env.close()
print("Smoke test completed successfully!")
