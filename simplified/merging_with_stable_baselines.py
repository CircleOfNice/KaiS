from over_simplified_env_check import *

#from stable_baselines3.common.policies import MlpPolicy
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3 import A2C


#from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines3 import A2C, DQN

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
env = DummyVecEnv([lambda: CustomEnv(4)])
#env = CustomEnv(4)    
#check_env(env)

Episodes = 1000
Episode_length = 100
total_reward_list = []

model = DQN(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=100000)
print('done dqn learn')
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