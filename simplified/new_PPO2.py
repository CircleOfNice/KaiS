import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
#from over_simplified_env_check import *
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy
#from stable_baselines.common.policies.ActorCriticPolicy import MlpPolicy
from stable_baselines import DQN
from stable_baselines import PPO2
#from new_env import CustomEnv
#from new_env_with_step_count_in_state import CustomEnv
from Gym_Env_Random_Tasks_With_Labels import CustomEnv
from env_run import get_all_task_kubernetes
from collections import Counter

#env = DummyVecEnv([lambda: CustomEnv(4)])
path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)

env = CustomEnv(4, result_list, True) 

Episodes = 10000
Episode_length = len(result_list[0])#[:1000])

print(Episode_length)
total_reward_list = []
#policy_kwargs = dict(act_fun=tf.nn.sigmoid, net_arch=[ 32])
model = PPO2("MlpPolicy", env)
''',exploration_fraction = 0.1,
train_freq=500, batch_size =32, double_q = True,
exploration_initial_eps=1,  
gamma=0.99,exploration_final_eps=0.05,  prioritized_replay=True,
learning_rate=0.001,  buffer_size=5000000)
'''
max_reward_model = 0
for epi in tqdm(range(Episodes)):
    env.set_train_param(True)
    model.learn(total_timesteps=Episode_length-1)
    
    sum_reward = sum(env.reward_list )
    print('done PPO2 learn Episode_length : ', sum_reward/ (Episode_length*2))
    print(f'Episode : {epi} reward for episode {sum_reward}')
    if sum_reward>max_reward_model:
        model.save(os.path.join('models','PPO2', str(sum_reward)))
        max_reward_model = sum_reward
    total_reward_list.append(sum_reward)
    env.reward_list=[]
    env.reset()



print('total_reward_list : ', total_reward_list)
model.save("PPO2")
plt.plot(total_reward_list)
plt.title('Reward PPO2 Over time')
plt.ylabel('Reward Accumulated')
plt.xlabel('Episodes')
plt.show()