import os
from tqdm import tqdm
#from over_simplified_env_check import *
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
from over_simplified_env_check import CustomEnv
from env_run import get_all_task_kubernetes
from collections import Counter
path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
def get_rewards_kubernetes_data(path, model, env):
    
    result_list,_ = get_all_task_kubernetes(path)

    [type_list, start_time_list, end_time_list, cpu_list, mem_list] =result_list
    print('env.train : ', env.train)
    
    total_reward = 0
    action_list = []
    for i in range(len(start_time_list)):
        task = [type_list[i], start_time_list[i], end_time_list[i], cpu_list[i], mem_list[i]]
        env.update_incoming_task(task)
        observation = env.master.get_master_observation_space()
        
        action, _states = model.predict(observation)
        action_list.append(action)
        #if i%50==0:
        #    print('observation : ' , observation)
        #    print('Kubernetese action_produced : ', action )
        #if i == 2000:
        #    break
        #print(action)
        #action = env.get_random_action()
        observation, reward, done, info = env.step(action)
        total_reward += reward
    print('set(action_list) : ', set(action_list))   
    values, counts = Counter(action_list).keys(), Counter(action_list).values()
    #print('Counter : ', Counter)
    print('values, counts', values, counts) 
    print('final observation : ', observation)
    return total_reward
    
#env = DummyVecEnv([lambda: CustomEnv(4)])
env = CustomEnv(4, train=True)
Episodes = 10
Training_cycles = 1
Episode_length = 10000000
total_reward_list = []

model = DQN(MlpPolicy, env, verbose=1,exploration_fraction = 0.1,
            train_freq=10, batch_size =10000, double_q = False,
            exploration_initial_eps=1.0,  
            gamma=0.99,exploration_final_eps=0.2,  prioritized_replay=True,
            learning_rate=1,  buffer_size=5000000)

avg_total_reward_kuberenetes  = get_rewards_kubernetes_data(path, model, env)
print('avg_total_reward_kuberenetes for episode Before : ', avg_total_reward_kuberenetes)

for epi in tqdm(range(Episodes)):
    env.set_train_param(True)
    model.learn(total_timesteps=Episode_length)
    #print('done dqn learn')
    #print(f'Episode : {epi} reward for episode {sum(env.reward_list )}')
    env.reward_list = []
env.set_train_param(False)
avg_total_reward_kuberenetes  = get_rewards_kubernetes_data(path, model, env)
print('avg_total_reward_kuberenetes for episode  : ', avg_total_reward_kuberenetes)


#print('Evaluating kubernetes rewards : ')
#env.set_train_param(False)
#avg_total_reward_kuberenetes  = get_rewards_kubernetes_data(path, model, env,)
#print('avg_total_reward_kuberenetes Before Training  : ', avg_total_reward_kuberenetes)
'''
for episode in range(Episodes):
    print()
    print()
    print()
    print('episode : ', episode)
    for cycle in range(Training_cycles):
        env.set_train_param(True)
        print(f'Episode : {episode} Training Cycle : {cycle}')
        model.learn(total_timesteps=Episode_length)
        print('model.gamma: ', model.gamma)
        print('model.gamma: ', model.gamma)
    print('model learning done, know evaluating on Kubenetes data')
    env.set_train_param(False)
    avg_total_reward_kuberenetes  = get_rewards_kubernetes_data(path, model, env)
    print('avg_total_reward_kuberenetes for episode  : ', avg_total_reward_kuberenetes)
print('done dqn learn')

'''


'''

for i in range(Episodes):
    total_reward = 0
    for episode_step in range(Episode_length):
        observation = env.reset()
        
        task = env.generate_random_task()
        env.update_incoming_task(task)
        
        action = env.get_random_action()
        observation, rewards, done, info = env.step(action)
        total_reward += rewards
        
    total_reward_list.append(total_reward)
print('total_reward_list : ', total_reward_list)
print('Average total_reward_list : ', sum(total_reward_list)/len(total_reward_list))

'''