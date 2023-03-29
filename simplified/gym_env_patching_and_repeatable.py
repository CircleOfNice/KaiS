import gym 
from gym import spaces
from components.Master import Master
from tqdm import tqdm
#from components.Node import Node
import numpy as np
from gym.spaces.box import Box
from random import shuffle, choice, sample
import random
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
        self.max_available_cpu_choices = [4000, 2000, 1000, 500]
        self.max_available_cpu= random.choice(self.max_available_cpu_choices)
        
        self.max_available_mem_choices = [2, 1, 0.5]

        self.mask_list = [1 for i in range(self.number_of_nodes)]
        self.max_available_memory = random.choice(self.max_available_mem_choices)
        self.current_incoming_task = [0, 0, 0, 0, 0]
        self.req_cpu = 100
        self.req_mem = 0.1
        self.train = train
        self.set_node_list()
        self.action_space = len(self.node_list)
        self.observation_space_dims = self.get_master_observation_space().shape
        self.data = data
        self.action_value = 0
    def set_node_list(self):
        
        cpu_params= []
        mem_params= []
        
        fail_params_choice_list = [[[0, self.req_cpu],[0, self.req_mem]],
                                [[self.req_cpu, self.max_available_cpu],[0, self.req_mem]],
                                [[0, self.req_cpu],[self.req_mem, self.max_available_memory],
                                ]]
                                
        
        fail_params_choice_list = [[[0, self.max_available_cpu],[0, self.max_available_memory]],]
        selection_list = [i for i in range(self.number_of_nodes)]
        actionable_node = np.random.choice(selection_list)
        
        # To be triggered Every Time Step:#
        for _ in range(self.number_of_nodes):
          selected_fail_param = choice(fail_params_choice_list)
          cpu_params.append(np.random.randint(selected_fail_param[0][0],selected_fail_param[0][1]))
          mem_params.append(np.random.uniform(selected_fail_param[1][0],selected_fail_param[1][1]))

        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i]) for i in range(self.number_of_nodes)]
        self.action_value = actionable_node
        
    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu = task[3]
        self.req_mem = task[4]
        
    def get_master_observation_space(self, normalised = True):
        master_observation_space = [] 
        for i, node in  enumerate(self.node_list):
            state_normalisation = node.get_current_state_data()
            if normalised:
                master_observation_space.append((state_normalisation[0], self.max_available_cpu, state_normalisation[1], self.max_available_memory))
            else:
                master_observation_space.append((state_normalisation[0], self.max_available_cpu, state_normalisation[1], self.max_available_memory))
        
        master_observation_space = np.vstack(master_observation_space)
        return master_observation_space
    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    def execute_action(self, action):
      
        node = self.node_list[action]
        cpu_list = []
        mem_list = []
        for i , node in enumerate(self.node_list):
            if self.mask_list[i]==0:
                cpu_list.append(0)
                mem_list.append(0)
            else:
                cpu_list.append(node.cpu)
                mem_list.append(node.mem)
        cpu_index_max = max(range(len(cpu_list)), key=cpu_list.__getitem__)  
        mem_index_max = max(range(len(mem_list)), key=mem_list.__getitem__)  
        
        cpu_reward = (node.cpu/ self.req_cpu)#/self.max_available_cpu)#((node.cpu- self.req_cpu)/self.max_available_cpu)
        mem_reward = (node.mem/ self.req_mem)#((node.mem- self.req_mem)/self.max_available_memory)
        
        #print()
        #print('action : ', action)
        #print('Before cpu_reward, mem_reward : ', cpu_reward, mem_reward)
        if cpu_reward>=1 and mem_reward>=1:
          if cpu_reward<mem_reward:
              if action ==cpu_index_max:
                #cpu_reward = mem_reward
                cpu_reward = cpu_reward + 10
                
          elif mem_reward<cpu_reward:
              if action ==mem_index_max:
                #mem_reward = cpu_reward
                mem_reward = mem_reward + 10

        elif cpu_reward<1 and  mem_reward<1:
          cpu_reward = -abs(cpu_reward)
          mem_reward = -abs(mem_reward)
        else:
          cpu_reward = -10
          mem_reward = -10
        #print('After cpu_reward, mem_reward : ', cpu_reward, mem_reward)  
        #print('cpu_reward, mem_reward : ', cpu_reward, mem_reward)
        reward = cpu_reward  + mem_reward
        '''
        if mem_reward>0:
          mem_reward = 1
          
        else:
          mem_reward = -1
        
        reward = cpu_reward  + mem_reward
        '''
        if reward >0:
          if action ==mem_index_max:
              reward = 10 + reward

          if action ==cpu_index_max:
              reward = 10 + reward
        
        if node.cpu>=self.req_cpu and node.mem >= self.req_mem:
          
          node.update_state(cpu_val = - self.req_cpu, mem_val= - self.req_mem)

        return reward 

    def reset(self):
      self.set_node_list()

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, number_of_nodes, mask_nodes, data, train ):
    super(CustomEnv, self).__init__()
    self.number_of_nodes=number_of_nodes
    self.master = Master(number_of_nodes, data, train)
    self.mask_nodes = mask_nodes
    
    self.reset()
    self.action_space = spaces.Discrete(self.master.action_space)

    self.observation_space = Box(low=0.0, high=4000.0, shape=self.master.observation_space_dims, dtype=np.float32)
    self.train = train
    self.reward_list = []
    self.step_counter = 0
    self.number_of_masked_nodes = choice([i for i in range(int(self.number_of_nodes/2))])
    task = self.generate_new_task()
    self.update_incoming_task(task) 
    self.data_len = len(self.master.data[0][:])
  def set_train_param(self, param):
    self.train = param
    self.master.train = param
    self.master.set_node_list()
    
  def valid_action_mask(self):
      actions = [i for i in range(self.number_of_nodes)]
      mask = sample(actions,  self.mask_nodes)
      self.master.mask_list = [1 if i in mask else 0 for i in range(self.number_of_nodes)]
      return self.master.mask_list
  def reset(self):
    self.master.reset()
    observation = self.master.get_master_observation_space()
    self.step_counter = 0
    task = self.generate_new_task()
    self.update_incoming_task(task) 
    return observation 
  
  def get_random_action(self):
      return self.master.get_random_action()
    
  '''
  def generate_random_task(self):
    req_cpu = np.random.randint(0, self.master.max_available_cpu/4)
    req_mem = np.random.uniform(0, self.master.max_available_memory/8)
    task = [0,0,0, req_cpu, req_mem]
    return task
  '''
  def generate_new_task(self):
    data = self.master.data
    task = [data[0][self.step_counter], data[1][self.step_counter], data[2][self.step_counter], data[3][self.step_counter], data[4][self.step_counter]]
    return task
  def step(self, action):
    optimal_action = self.master.action_value
    reward = self.master.execute_action(action)

    done =False
    info = {}
    # Only done during training inference on reall data requires it to be removed
    #reward = reward#*self.step_counter/ self.data_len#- self.step_counter*self.data_len
    self.reward_list.append(reward)  
    self.step_counter = self.step_counter + 1
    self.master.reset()
    if self.step_counter == self.data_len -1:
        self.step_counter = 0

    observation_ = self.master.get_master_observation_space()
    return observation_, reward, done, info
    
  def update_incoming_task(self, task):
    self.master.set_incoming_task(task)
    
  def render(self, mode='human'):
    pass
  def close (self):
    self.master.reset()

