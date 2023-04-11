import gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_run import get_all_task_kubernetes
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

#from new_gym_patching import CustomEnv
from gym_env_patching_and_repeatable import CustomEnv


def mask_fn(env:CustomEnv) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.ordered_valid_action_mask()
    # return env.valid_action_mask()

path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)
total_nodes = 4
masked_nodes = 3

custom_env = CustomEnv(total_nodes, masked_nodes, result_list, True)   # Initialize env
custom_env = ActionMasker(custom_env, mask_fn)  # Wrap to enable masking
# custom_env = Monitor(custom_env, filename="monitor.log")
check_env(custom_env, warn=False)

# Use evaluation environment to calculate mean reward and find best model
eval_env = CustomEnv(total_nodes, masked_nodes, result_list, True)   # Initialize env
eval_env = ActionMasker(custom_env, mask_fn)  # Wrap to enable masking
eval_env = Monitor(eval_env)
eval_callback = EvalCallback(eval_env, best_model_save_path="best_model", log_path="logs",
                              eval_freq=10_000, deterministic=True, render=False, n_eval_episodes=1, verbose=False)

Episode_length = len(result_list[0])#[:5])
Episodes = 300 #5000
# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
total_reward_list = []

# policy_kwargs = dict(net_arch=[32, 64, 128, 256, 512, 1024])
# policy_kwargs = dict(net_arch=[32, 32, 32])
policy_kwargs = None
model = MaskablePPO(MaskableActorCriticPolicy, custom_env, verbose=0, tensorboard_log="tensorboard_logs", policy_kwargs = policy_kwargs)#, verbose=True)

# Simple one shot training 
custom_env.set_train_param(True)
model.learn(total_timesteps=(Episode_length-1)*Episodes, progress_bar=True, callback=eval_callback)
print(' len(customenv.reward_list) : ', sum(custom_env.reward_list ), len(custom_env.reward_list))
sum_reward = sum(custom_env.reward_list )/ len(custom_env.reward_list)
model.save(os.path.join('models','PPO2', str(sum_reward)))



# max_reward_model = 0
# for epi in tqdm(range(Episodes)):
#     customenv.set_train_param(True)
#     model.learn(total_timesteps=(Episode_length-1))
#     print('done PPO2 learn')
#     print(' len(customenv.reward_list) : ', sum(customenv.reward_list ), len(customenv.reward_list))
#     sum_reward = sum(customenv.reward_list )/ len(customenv.reward_list)
#     print(f'Episodes : {epi} average reward till episode {sum_reward}')
#     if sum_reward>max_reward_model:
#         model.save(os.path.join('models','PPO2', str(sum_reward)+'_'+str(total_nodes) +'_node_'+ str(masked_nodes)+'_mask.zip'))
#         max_reward_model = sum_reward
#     total_reward_list.append(sum_reward)
#     plt.plot(total_reward_list)
#     plt.title('Reward PPO2 Over time')
#     plt.ylabel('Reward Accumulated ')
#     plt.xlabel('Episodes')
#     #plt.show()
#     plt.savefig('current_progress2.png',  dpi=300)
   
# print('total_reward_list : ', total_reward_list)
# model.save("PPO2_Final_" +'_'+str(total_nodes) +'_node_'+ str(masked_nodes)+'_mask.zip')
# plt.plot(total_reward_list)
# plt.title('Final Reward PPO2 Over time')
# plt.ylabel('Reward Accumulated ')
# plt.xlabel('Episodes')
# plt.show()
