import os
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
#from over_simplified_env_check import *
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
#from new_env_with_step_count_in_state import CustomEnv
#from new_env import CustomEnv
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
policy_kwargs = dict(act_fun=tf.nn.sigmoid, net_arch=[32])
#model = A2C(MlpPolicy, env, verbose=1,n_steps = 100,policy_kwargs=policy_kwargs,
#            learning_rate=0.001)
model = A2C(MlpPolicy, env,n_steps =int(Episode_length/2),
            learning_rate=0.001)

for epi in tqdm(range(Episodes)):
    env.set_train_param(True)
    model.learn(total_timesteps=Episode_length-1)
    print('done a2c learn episode')
    print(f'Episode : {epi} reward for episode {sum(env.reward_list )}')
    total_reward_list.append(sum(env.reward_list ))
    env.reward_list=[]
    print('new reward list for environment : ', env.reward_list)
    env.reset()

print('total_reward_list : ', total_reward_list)
model.save("a2c")
plt.plot(total_reward_list)
plt.title('Reward A2C Over time')
plt.ylabel('Reward Accumulated')
plt.xlabel('Episodes')
plt.show()