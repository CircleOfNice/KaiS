# Test to verify functionality of the model in a test environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sb3_contrib
from new_gym_with_next_state_fixed import CustomEnv
import numpy as np
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

    num_test_runs = 10_000

    MODEL_PATH = r"models/PPO2/final_ppo_model.zip"
    ENV_PATH = r"models/PPO2/final_env.zip"

    env = DummyVecEnv([lambda: CustomEnv(4, 3, [[]])])
    # vec_env = VecNormalize.load(ENV_PATH, env)
    # vec_env.training = False

    # model = sb3_contrib.MaskablePPO.load(MODEL_PATH, env=env)
    # model.set_env(vec_env)

    model = sb3_contrib.MaskablePPO.load(MODEL_PATH)
    # self.model:sb3_contrib.MaskablePPO = sb3_contrib.MaskablePPO.load(MODEL_SAVE_PATH)
    MAX_NODE_CAPACITY = model.action_space.n
    action_dist = np.zeros(MAX_NODE_CAPACITY)

    print(model.policy)
    for _ in range(num_test_runs):
        obs = env.reset()[0]
        pprint_obs(obs)
        pred = pprint_pred(model, obs)
        action_dist[pred] += 1


    print(f"Action distribution: {action_dist}")
    print(f"Action distribution normalized: {action_dist / np.sum(action_dist)}")


    





