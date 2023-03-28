import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch 
from stable_baselines3 import PPO

from new_gym_patching import CustomEnv
from env_run import get_all_task_kubernetes
from collections import Counter

path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)

env = CustomEnv(4, result_list, True) 

Episodes = 20#5000
Episode_length = len(result_list[0])#[:1000])

print(Episode_length)
total_reward_list = []


policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[64, 128, 256, 512, 1024])

model = PPO("MlpPolicy", env)#, policy_kwargs = policy_kwargs)
print(model.policy)
#a=B
max_reward_model = 0
for epi in tqdm(range(Episodes)):
    env.set_train_param(True)
    model.learn(total_timesteps=Episode_length-1)
    print('done PPO2 learn')
    sum_reward = sum(env.reward_list )
    print(f'Episode : {epi} reward for episode {sum_reward}')
    print('done PPO2 learn throughput: ', sum_reward/ (Episode_length*30))
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
plt.ylabel('Reward Accumulated ')
plt.xlabel('Episodes')
plt.show()

plt.plot([x / (2*Episode_length) for x in total_reward_list])
plt.title('Throughput PPO2 Over time')
plt.ylabel('Throughput ')
plt.xlabel('Episodes')
plt.show()