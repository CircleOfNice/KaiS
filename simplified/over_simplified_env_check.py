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
        state = [self.cpu, self.mem]
        return state
    def update_state(self, cpu_val, mem_val):
        self.cpu = self.cpu + cpu_val
        self.mem = self.mem + mem_val

class Master:
    """[This class serves as framework for definition of Master Node with properties such as 
    task queue, cpu processing, memory, done and undone tasks, Kind of tasks done and undone. all task index]
    """
    def __init__(self, number_of_nodes, train = True):
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
        
    def set_node_list(self):
        cpu_params= []
        mem_params= []
        check = False
        # To be triggered Every Time Step:#
        for _ in range(self.number_of_nodes):
          '''
          condition_no = [0, 1, 2, 3, 4]
          choice = np.random.choice(condition_no)
          if choice == 0:
            req_cpu = np.random.randint(0, self.max_available_cpu)
            req_mem = np.random.uniform(0, self.max_available_memory)
          elif choice == 1:
            req_cpu = np.random.randint(self.req_cpu, self.max_available_cpu)
            req_mem = np.random.uniform(self.req_mem, self.max_available_memory)
          elif choice == 2:
            req_cpu = np.random.randint(self.req_cpu, self.max_available_cpu)
            req_mem = np.random.uniform(0, self.req_mem)
          elif choice == 3:
            if self.req_cpu ==0:
              req_cpu = self.req_cpu
            else:
              req_cpu = np.random.randint(0, self.req_cpu)
            req_mem = np.random.uniform(self.req_mem, self.max_available_memory)
          elif choice == 4:
            if self.req_cpu ==0:
              req_cpu = self.req_cpu
            else:
              req_cpu = np.random.randint(0, self.req_cpu)
            req_mem = np.random.uniform(0, self.req_mem)
          #print('updating node list')
          cpu_params.append(req_cpu)#np.random.randint(0, req_cpu))
          mem_params.append(req_mem)#np.random.uniform(0, req_mem))
          '''
          if self.train:
            cpu_params.append(np.random.randint(0, self.max_available_cpu))
            mem_params.append(np.random.uniform(0, self.max_available_memory))
            '''
            #if np.random.uniform(0, 1)>0.5:
            if np.random.uniform(0, 1)<0.5:
              cpu_params.append(np.random.randint(0, self.max_available_cpu))
              mem_params.append(np.random.uniform(0, self.max_available_memory))
            else:
              cpu_params.append(self.max_available_cpu)
              mem_params.append(self.max_available_memory)
            #else:
            #    cpu_params.append(self.max_available_cpu)
            #    mem_params.append(self.max_available_memory)
            '''
          elif self.train == False:
            #print('elif self.train == False : ', self.train == False)
            cpu_params.append(self.max_available_cpu)
            mem_params.append(self.max_available_memory)
          #print(cpu_params, mem_params)
        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i]) for i in range(self.number_of_nodes)]
    
    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu = task[3]
        self.req_mem = task[4]
        
    def get_master_observation_space(self):
        master_observation_space = [] 
        for i, node in  enumerate(self.node_list):
            state_normalisation = node.get_current_state_data()
            master_observation_space.append((state_normalisation[0]/self.max_available_cpu, state_normalisation[1]/ self.max_available_memory))
        master_observation_space = np.vstack(master_observation_space)
        return master_observation_space
    
    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    def execute_action(self, action):
        reward = -1
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

  def __init__(self, number_of_nodes, train ):
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
    self.train = train
    self.reward_list = []
    
  def set_train_param(self, param):
    self.train = param
    self.master.train = param
    self.master.set_node_list()
    
  def reset(self):
    self.master.reset()
    observation = self.master.get_master_observation_space()
    #print('reset observation : ', observation)
    return observation  # reward, done, info can't be included
  
  def get_random_action(self):
      return self.master.get_random_action()
  def generate_random_task(self):
    req_cpu = np.random.randint(0, self.master.max_available_cpu/4)
    req_mem = np.random.uniform(0, self.master.max_available_memory/8)
    task = [0,0,0, req_cpu, req_mem]
    return task

  def step(self, action):
    
    reward = self.master.execute_action(action)
    
    self.reward_list.append(reward)  
    observation = self.master.get_master_observation_space()
    #if self.train:
    #  print('action training : ', action )
    #  print('observation : ', observation)
    #  print('reward training : ',  reward)
    #print('observation : ', observation[:,0], observation[:,2])
    done =False
    info = {}
    # Only done during training inference on reall data requires it to be removed
    
    if self.train:
      #print('inside train : ')
      task = self.generate_random_task()
      self.update_incoming_task(task)
      self.reset()    
    
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
        action = env.get_random_action()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
    total_reward_list.append(total_reward)
print('total_reward_list : ', total_reward_list)
print('Average total_reward_list : ', sum(total_reward_list)/len(total_reward_list))

'''