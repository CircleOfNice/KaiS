# Test to verify functionality of the model in a test environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sb3_contrib
from new_gym_with_next_state_fixed import CustomEnv
import numpy as np
import os
from env_run import get_all_task_kubernetes
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)


def pprint_obs(obs:np.array, precision=2):

    obs = np.round(obs, 2)
    num_nodes = (obs.shape[0] - 2) // 4

    print()
    print(f"Task: CpuReq: {obs[-2]} - MemReq: {obs[-1]}")
    for i in range(num_nodes):
        print(f"Node {i}: FreeCpu: {obs[i*4]} - FreeMem: {obs[i*4+1]} - CpuAlloc: {obs[i*4+2]} - MemAlloc: {obs[i*4+3]}")


def pprint_pred(model, obs):

    mask = np.ones(MAX_NODE_CAPACITY)
    node_idx, _ = model.predict(obs, deterministic=True, action_masks=mask)

    obs_torch = model.policy.obs_to_tensor(obs)[0]
    dis = model.policy.get_distribution(obs_torch)
    probs = dis.distribution.probs[0]

    for i in range(len(probs)):
        print(f"Node {i} proba: {np.round(probs[i].item(), 2)}")

    print(f"Model Prediction: Node {node_idx}")

    return node_idx


if __name__ == "__main__":

    num_test_runs = 1

    MODEL_PATH = r"models/PPO2/final_ppo_model.zip"
    ENV_PATH = r"models/PPO2/final_env.zip"

    path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
    result_list,_ = get_all_task_kubernetes(path)

    # https://stable-baselines3.readthedocs.io/en/v0.11.1/guide/examples.html#pybullet-normalizing-input-features
    # Link above shows example of how to load a model with a vecnormalize wrapper
    env = CustomEnv(4, 3, result_list, normalize_obs=True, init_random=False)
    model = sb3_contrib.MaskablePPO.load(MODEL_PATH)


    # model = sb3_contrib.MaskablePPO.load(MODEL_PATH)
    # self.model:sb3_contrib.MaskablePPO = sb3_contrib.MaskablePPO.load(MODEL_SAVE_PATH)
    MAX_NODE_CAPACITY = model.action_space.n
    action_dist = np.zeros(MAX_NODE_CAPACITY)

    print(model.policy)
    for _ in range(num_test_runs):
        obs = env.reset()

        # env.master.debug_init_node_list()
        # obs = env.master.get_observation_space()

        pprint_obs(obs)
        # obs = vec_env.normalize_obs(obs)
        pred = pprint_pred(model, obs)
        action_dist[pred] += 1


    print(f"Action distribution: {action_dist}")
    print(f"Action distribution normalized: {action_dist / np.sum(action_dist)}")

    done = False

    action_list = []
    remaining_cpu_list = []
    remaining_mem_list = []

    print("Testing on empty cluster")
    episode_length = 0
    while not done:
        action_mask = np.ones(MAX_NODE_CAPACITY) # TODO action mask set to all ones currently
        action,_ = model.predict(obs, deterministic=True, action_masks=action_mask)
        action = action.item()
        obs, reward, done, info = env.step(action)

        episode_length += 1
        action_list.append(action)
        remaining_cpu_list.append(obs[::4][:-1])
        remaining_mem_list.append(obs[1::4][:-1])

    label_list = ["Node_" + str(i) for i in range(1, MAX_NODE_CAPACITY+1)]
    print(f"Episode length: {episode_length}")
    print(f"Cluster max achieved? {env.master.max_capacity_count}")
    plt.figure()
    plt.plot(remaining_cpu_list, label=label_list)
    plt.title("Remaining CPU")
    plt.legend()
    plt.savefig("remaining_cpu.png")

    plt.figure()
    plt.plot(remaining_mem_list, label=label_list)
    plt.title("Remaining Memory")
    plt.legend()
    plt.savefig("remaining_mem.png")

    action_dist, counts = np.unique(action_list, return_counts=True)
    print(f"Action distribution: {counts}")



    





