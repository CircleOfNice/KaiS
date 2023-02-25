import gym 
from gym import spaces
from components.Master import Master
#from components.Node import Node
from stable_baselines3.common.env_checker import check_env

from newspace import NewSpace
import numpy as np
from gym.spaces.box import Box
from gym.spaces.space import Space
from gym.spaces.tuple import Tuple
from gym.spaces.multi_discrete import MultiDiscrete
from gym.wrappers import FlattenObservation
class Node:
    """[This class serves as framework for definition of Edge Node with properties such as 
    task queue, service_list, cpu processing and  memory]
    """
    def __init__(self, cpu:float, mem:float):
        self.cpu = cpu
        self.mem = mem
    
    def get_current_state_data(self):
        state = [self.cpu, self.mem,]
        return state
    def update_state(self, cpu_val, mem_val):
        self.cpu = self.cpu + cpu_val
        self.mem = self.mem + mem_val

class Master:
    """[This class serves as framework for definition of Master Node with properties such as 
    task queue, cpu processing, memory, done and undone tasks, Kind of tasks done and undone. all task index]
    """
    def __init__(self, number_of_nodes):
        self.number_of_nodes = number_of_nodes
        
        self.max_available_cpu= 4000
        self.max_available_memory = 2
        self.current_incoming_task = [0, 0, 0, 0, 0]
        self.req_cpu = 100
        self.req_mem = 1
        self.set_node_list()
        self.action_space = len(self.node_list)
        self.observation_space_dims = self.get_master_observation_space().shape
        
    def set_node_list(self):
        cpu_params= []
        mem_params= []
        # To be triggered Every Time Step:#
        for _ in range(self.number_of_nodes):
            cpu_params.append(np.random.randint(0, self.max_available_cpu))
            mem_params.append(np.random.uniform(0, self.max_available_memory))
            #print(cpu_params, mem_params)
        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i]) for i in range(self.number_of_nodes)]
    
    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu = task[3]
        self.req_mem = task[4]
        
    def get_master_observation_space(self):
        master_observation_space = [] 
        for i, node in  enumerate(self.node_list):
            master_observation_space.append(node.get_current_state_data())
        master_observation_space = np.vstack(master_observation_space)
        return master_observation_space
    
    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    def execute_action(self, action):
        reward = -1
        node = self.node_list[action]
        if node.cpu>=self.req_cpu and node.mem >= self.req_mem:
          #print('Before State : ', node.get_current_state_data())
          #print('self.req_cpu, self.req_mem : ', self.req_cpu, self.req_mem)
          node.update_state(cpu_val = - self.req_cpu, mem_val= - self.req_mem)
          #print('After State : ', node.get_current_state_data())
          reward =  1    
        #print('reward : ', reward)
        return reward 

    def reset(self):
      self.set_node_list()
'''
number_of_nodes = 4  
master = Master(number_of_nodes)

print('reset : ', [(node.cpu, node.mem) for node in master.node_list])
print('Action Space ', master.action_space)
print('master.max_available_cores ', master.max_available_cpu)
print('master.max_available_memory ', master.max_available_memory)

for i in range(10):
  req_cpu = np.random.randint(0, master.max_available_cpu)
  req_mem = np.random.uniform(0, master.max_available_memory)
  task = [0,0,0, req_cpu, req_mem]
  #print('req_cpu, req_mem : ', req_cpu, req_mem)
  master.set_incoming_task(task)
  master.set_node_list()
  observation = master.get_master_observation_space()
  #print('Before observation : ', observation)
  action = master.get_random_action()
  print('get_random_action ', action)
  reward = master.execute_action(action)
  print('Reward Generated : ', reward)
  #print('After observation : ', observation)
  
a=b
'''

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, number_of_nodes ):
    super(CustomEnv, self).__init__()
    self.number_of_nodes=number_of_nodes
    self.master = Master(number_of_nodes)
    self.reset()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(self.master.action_space)
    # Example for using image as input:
    #self.observation_space = NewSpace(shape=self.master.observation_space_dims)
    self.observation_space = Box(low=0.0, high=4000.0, shape=self.master.observation_space_dims, dtype=np.float32)
    #self.observation_space = MultiDiscrete(nvec= )
    #print('self.action_space , self.observation_space : ', self.action_space , self.observation_space)
    
  def reset(self):
    self.master.reset()
    observation = self.master.get_master_observation_space()
    #print('reset observation : ', observation)
    return observation  # reward, done, info can't be included
  
  def get_random_action(self):
      return self.master.get_random_action()
  def generate_random_task(self):
    req_cpu = np.random.randint(0, self.master.max_available_cpu)
    req_mem = np.random.uniform(0, self.master.max_available_memory)
    task = [0,0,0, req_cpu, req_mem]
    return task

  def step(self, action):
    reward = self.master.execute_action(action)
    observation = self.master.get_master_observation_space()
    #print('observation : ', observation[:,0], observation[:,2])
    done =False
    info = {}
    # Only done during training inference on reall data requires it to be removed
    
    self.reset()    
    task = self.generate_random_task()
    self.update_incoming_task(task)
    
    
    return observation, reward, done, info
    
  def update_incoming_task(self, task):
    self.master.set_incoming_task(task)
    
  def render(self, mode='human'):
    pass
  def close (self):
    self.master.reset()

'''
env = CustomEnv(4)    
#check_env(env)

Episodes = 1000
Episode_length = 100
total_reward_list = []
for i in range(Episodes):
    total_reward = 0
    for episode_step in range(Episode_length):
        env.reset()
        task = env.generate_random_task()
        env.update_incoming_task(task)
        action = env.get_random_action()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
    total_reward_list.append(total_reward)
print('total_reward_list : ', total_reward_list)
print('Average total_reward_list : ', sum(total_reward_list)/len(total_reward_list))

'''