import os
from tqdm import tqdm
from components.Task import Task
from over_simplified_env_check import CustomEnv
from env_run import get_all_task_kubernetes
path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')


result_list,_ = get_all_task_kubernetes(path)

[type_list, start_time_list, end_time_list, cpu_list, mem_list] =result_list
print(len(start_time_list))

env = CustomEnv(4)    
#check_env(env)

Episodes = 10
total_reward_list = []
for i in range(Episodes):
    total_reward = 0
    for i in range(len(start_time_list)):
        env.reset()
        task = [type_list[i], start_time_list[i], end_time_list[i], cpu_list[i], mem_list[i]]
        env.update_incoming_task(task)
        action = env.get_random_action()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
    total_reward_list.append(total_reward)
print('total_reward_list : ', total_reward_list)
print('Average total_reward_list : ', sum(total_reward_list)/len(total_reward_list))
