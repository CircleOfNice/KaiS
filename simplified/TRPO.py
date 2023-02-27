import os
from tqdm import tqdm
import tensorflow as tf
#from over_simplified_env_check import *
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.trpo_mpi import TRPO
from new_env import CustomEnv
from env_run import get_all_task_kubernetes
from collections import Counter

#env = DummyVecEnv([lambda: CustomEnv(4)])
path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)

env = CustomEnv(4, result_list, True) 

Episodes = 5
Episode_length = len(result_list[0][:-1])
print(Episode_length)
total_reward_list = []
policy_kwargs = dict(act_fun=tf.nn.sigmoid, net_arch=[256, 128, 64, 32])
model =  TRPO(MlpPolicy, env, verbose=1,policy_kwargs=policy_kwargs,)

for epi in tqdm(range(Episodes)):
    env.set_train_param(True)
    model.learn(total_timesteps=Episode_length-1)
    print('done dqn learn')
    print(f'Episode : {epi} reward for episode {sum(env.reward_list )}')
    total_reward_list.append(sum(env.reward_list ))
    env.reward_list=[]

print('total_reward_list : ', total_reward_list)