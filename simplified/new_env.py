import gym 
from gym import spaces
from components.Master import Master
#from components.Node import Node
import numpy as np
from gym.spaces.box import Box
from random import shuffle
class Node:
    """[This class serves as framework for definition of Edge Node with properties such as 
    task queue, service_list, cpu processing and  memory]
    """
    def __init__(self, cpu:float, mem:float):
        self.cpu = cpu
        self.mem = mem
    
    def get_current_state_data(self):
        state = [self.cpu, self.mem]
        return state
    def update_state(self, cpu_val, mem_val):
        self.cpu = self.cpu + cpu_val
        self.mem = self.mem + mem_val

class Master:
    """[This class serves as framework for definition of Master Node with properties such as 
    task queue, cpu processing, memory, done and undone tasks, Kind of tasks done and undone. all task index]
    """
    def __init__(self, number_of_nodes, data, train = True):
        self.number_of_nodes = number_of_nodes
        
        self.max_available_cpu= 4000
        self.max_available_memory = 2
        self.current_incoming_task = [0, 0, 0, 0, 0]
        self.req_cpu = 100
        self.req_mem = 0.1
        self.train = train
        self.set_node_list()
        self.action_space = len(self.node_list)
        self.observation_space_dims = self.get_master_observation_space().shape
        self.data = data
    def set_node_list(self):
        cpu_params= []
        mem_params= []
        check = False
        # To be triggered Every Time Step:#
        for _ in range(self.number_of_nodes):
            #cpu_params.append(np.random.randint(0, self.current_incoming_task[3]))
            #mem_params.append(np.random.uniform(0, self.current_incoming_task[4]))
        
            cpu_params.append(self.max_available_cpu)
            mem_params.append(self.max_available_memory)
        #shuffle(cpu_params)
        #shuffle(mem_params)
        '''
          if self.train:
            cpu_params.append(np.random.randint(0, self.max_available_cpu))
            mem_params.append(np.random.uniform(0, self.max_available_memory))
          elif self.train == False:
            #print('elif self.train == False : ', self.train == False)
            cpu_params.append(self.max_available_cpu)
            mem_params.append(self.max_available_memory)
        '''
          #print(cpu_params, mem_params)
        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i]) for i in range(self.number_of_nodes)]
    
    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu = task[3]
        self.req_mem = task[4]
        
    def get_master_observation_space(self, normalised = True):
        master_observation_space = [] 
        for i, node in  enumerate(self.node_list):
            state_normalisation = node.get_current_state_data()
            if normalised:
                master_observation_space.append((state_normalisation[0]/self.max_available_cpu, state_normalisation[1]/ self.max_available_memory))
            else:
                master_observation_space.append((state_normalisation[0], state_normalisation[1]))
        master_observation_space = np.vstack(master_observation_space)
        return master_observation_space
    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    def execute_action(self, action):
        reward = 0 
        node = self.node_list[action]
        
        #if self.train == True:
        #  print('phase train: ', self.train)
        #  print('Action Requested', action )
        #  print('Before State : ', node.get_current_state_data())
        #  print('self.req_cpu, self.req_mem : ', self.req_cpu, self.req_mem)
        if node.cpu>=self.req_cpu and node.mem >= self.req_mem:
          
          node.update_state(cpu_val = - self.req_cpu, mem_val= - self.req_mem)
          
          reward =  1  
        #if self.train == True:
        #  print('After State : ', node.get_current_state_data())
        #  print('reward : ', reward)
        #  print()
        #  print()
        #if reward == 1 and self.train:
        #  a=b
        return reward 

    def reset(self):
      self.set_node_list()

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, number_of_nodes, data, train ):
    super(CustomEnv, self).__init__()
    self.number_of_nodes=number_of_nodes
    self.master = Master(number_of_nodes, data, train)
    self.reset()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(self.master.action_space)
    # Example for using image as input:
    #self.observation_space = NewSpace(shape=self.master.observation_space_dims)
    self.observation_space = Box(low=0.0, high=4000.0, shape=self.master.observation_space_dims, dtype=np.float32)
    self.train = train
    self.reward_list = []
    self.step_counter = 0
    task = self.generate_new_task()
    self.update_incoming_task(task) 
    self.data_len = len(self.master.data[0][:])
  def set_train_param(self, param):
    self.train = param
    self.master.train = param
    self.master.set_node_list()
    
  def reset(self):
    self.master.reset()
    observation = self.master.get_master_observation_space()
    self.step_counter = 0
    task = self.generate_new_task()
    self.update_incoming_task(task) 
    #print('reset observation : ', observation)
    return observation  # reward, done, info can't be included
  
  def get_random_action(self):
      return self.master.get_random_action()
  def generate_random_task(self):
    req_cpu = np.random.randint(0, self.master.max_available_cpu/4)
    req_mem = np.random.uniform(0, self.master.max_available_memory/8)
    task = [0,0,0, req_cpu, req_mem]
    return task
  def generate_new_task(self):
    data = self.master.data
    #print(len(data), self.step_counter)
    task = [data[0][self.step_counter], data[1][self.step_counter], data[2][self.step_counter], data[3][self.step_counter], data[4][self.step_counter]]
    return task
  def step(self, action):
    
    #if self.step_counter%250==0:
    #    self.master.set_node_list()
    reward = self.master.execute_action(action)
    
    
    observation = self.master.get_master_observation_space()
    #if self.train:
    #  print('action training : ', action )
    #  print('observation : ', observation)
    #  print('reward training : ',  reward)
    #print('observation : ', observation[:,0], observation[:,2])
    done =False
    info = {}
    # Only done during training inference on reall data requires it to be removed
    reward = reward*self.step_counter/ self.data_len#- self.step_counter*self.data_len
    self.reward_list.append(reward)  
    
    self.step_counter = self.step_counter + 1
    if self.step_counter == self.data_len:
        self.step_counter = 0
        task = self.generate_new_task()
        self.update_incoming_task(task) 
    
    return observation, reward, done, info
    
  def update_incoming_task(self, task):
    self.master.set_incoming_task(task)
    
  def render(self, mode='human'):
    pass
  def close (self):
    self.master.reset()

import os
from env_run import get_all_task_kubernetes
path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
    
result_list,_ = get_all_task_kubernetes(path)

env = CustomEnv(4, result_list, True)    
#check_env(env)


'''
Episodes = 10
Episode_length = len(result_list[0][:-1])
print(Episode_length)

total_reward_list = []
for i in range(Episodes):
    total_reward = 0
    env.reset()
    for episode_step in range(Episode_length):
        action = env.get_random_action()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
    total_reward_list.append(total_reward)
observation = env.master.get_master_observation_space(False)
print('observation : ', observation)
print('total_reward_list : ', total_reward_list)
print('Average total_reward_list : ', sum(total_reward_list)/len(total_reward_list))

'''