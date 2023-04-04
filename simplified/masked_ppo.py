import gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_run import get_all_task_kubernetes
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
#from new_gym_patching import CustomEnv
from gym_env_patching_and_repeatable import CustomEnv
def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)
custom_env = CustomEnv(10, 4, result_list, True)   # Initialize env
customenv = ActionMasker(custom_env, mask_fn)  # Wrap to enable masking
Episode_length = len(result_list[0])#[:5])
Episodes = 10#5000
# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
total_reward_list = []

model = MaskablePPO(MaskableActorCriticPolicy, customenv, verbose=0, tensorboard_log="tensorboard_logs")#, verbose=True)



# Simple one shot training 
'''
customenv.set_train_param(True)
model.learn(total_timesteps=(Episode_length-1)*Episodes)
print('done PPO2 learn')
print(' len(customenv.reward_list) : ', sum(customenv.reward_list ), len(customenv.reward_list))
sum_reward = sum(customenv.reward_list )/ len(customenv.reward_list)
print(f'Episodes : {Episodes} reward for episode {sum_reward}')
print('done PPO2 learn throughput: ', sum_reward/ (Episode_length))
model.save(os.path.join('models','PPO2', str(sum_reward)))
max_reward_model = sum_reward
total_reward_list.append(sum_reward)
'''


max_reward_model = 0
for epi in tqdm(range(Episodes)):
    customenv.set_train_param(True)
    model.learn(total_timesteps=(Episode_length-1))
    print('done PPO2 learn')
    print(' len(customenv.reward_list) : ', sum(customenv.reward_list ), len(customenv.reward_list))
    sum_reward = sum(customenv.reward_list )/ len(customenv.reward_list)
    print(f'Episodes : {epi} reward for episode {sum_reward}')
    print('done PPO2 learn throughput: ', sum_reward/ (Episode_length))
    if sum_reward>max_reward_model:
        model.save(os.path.join('models','PPO2', str(sum_reward)))
        max_reward_model = sum_reward
    total_reward_list.append(sum_reward)
   
print('total_reward_list : ', total_reward_list)
model.save("PPO2")
plt.plot(total_reward_list)
plt.title('Reward PPO2 Over time')
plt.ylabel('Reward Accumulated ')
plt.xlabel('Episodes')
plt.show()

plt.plot([x / (2*Episode_length) for x in total_reward_list])
plt.title('Throughput PPO2 Over time')
plt.ylabel('Throughput ')
plt.xlabel('Episodes')
plt.show()
