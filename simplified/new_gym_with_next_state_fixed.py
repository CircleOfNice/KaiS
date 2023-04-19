import gym 
from gym import spaces
# from components.Master import Master
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
    def __init__(self, cpu:float, mem:float, max_cpu, max_mem):
        self.cpu = cpu
        self.mem = mem
        self.max_cpu = max_cpu
        self.max_mem = max_mem
    def __str__(self):
        return f"(cpu : {self.cpu}, mem: {self.mem})"
    def get_current_state_data(self):
        state = [self.cpu, self.mem, self.max_cpu, self.max_mem]
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

        self.action_distribution = np.zeros(number_of_nodes)

    
    def __str__(self):
        new_str ='\n'
        #new_str = [new_str + str(x) for x in self.node_list ]
        
        for x in self.node_list:
            new_str+=str(x)
            new_str+='\n'
        #new_str = new_str[-1]
        #print(new_str[-1])
        #print(*(x for x in self.node_list))
        #print('new_str : ', new_str)
        #a=b
        return new_str

    def set_node_list(self):
        """ This method is used to set the cpu and memory values of all nodes to a random value within the allowed interval """
        cpu_params= []
        mem_params= []
        max_cpu_params = [ random.choice(self.max_available_cpu_choices) for i in range(self.number_of_nodes)]
        max_mem_params = [ random.choice(self.max_available_mem_choices) for i in range(self.number_of_nodes)]

        for i in range(self.number_of_nodes):
            #cpu_params.append(np.random.randint(0,max_cpu_params[i]))
            #mem_params.append(np.random.uniform(0,max_mem_params[i]))
            cpu_params.append(max_cpu_params[i])
            mem_params.append(max_mem_params[i])
        try:
            #print('self.master.req_cpu, self.master.req_mem : ', self.req_cpu, self.req_mem)
            #print('Before self.node_list : ')
            
            print(self)
            #print(*(x for x in self.node_list), sep='\n')
            #print('done')
        except:
            pass
        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i], max_cpu=max_cpu_params[i], max_mem=max_mem_params[i]) for i in range(self.number_of_nodes)]
        #print('After self.node_list :', )
        #print(self)
        #
        # print(*(x for x in self.node_list), sep='\n')
        #print('done')
    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu = task[3]
        self.req_mem = task[4]
        

    def get_master_observation_space(self) -> np.array:
        """This method is used to get the observation for the model. The observation consists out of the values for available cpu and memory
        for each node with the shape of (#Node_num, 2). At the end we append the request cpu and memory for the given task, resulting in the shape 
        of (#Node_num+1, 2) for the observation space

        Returns:
            np.array: The observation with node and task information
        """
        master_observation_space = [] 

        # Get information of all nodes
        for _, node in  enumerate(self.node_list):
            state_normalisation = node.get_current_state_data()
            master_observation_space.append((state_normalisation[0], state_normalisation[1]))# , state_normalisation[2], state_normalisation[3]))

        # Get information of the task
        master_observation_space.append((self.current_incoming_task[3], self.current_incoming_task[4]))#, 0,0))

        master_observation_space = np.hstack(master_observation_space)
        return master_observation_space
    

    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    

    def execute_action(self, action:int) -> float:
        """This method is used to determine the reward for a given scheduling decision (action).
        Currently the reward is based on the amount of free resources. The more free resources (cpu and/or memory) the chose node has,
        the higher the reward for the model.

        This should result in an agent that favors a somewhat load-balanced approach

        Args:
            action (int): The number of the node which to schedule the given task to.

        Returns:
            float: The reward for the agent
        """
        node_choice = self.node_list[action]
        self.action_distribution[action] += 1
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
    
        '''
        if node_choice.cpu == 0 or node_choice.mem == 0:
            reward = -2
        else:
            cpu_pct = self.req_cpu / node_choice.cpu
            mem_pct = self.req_mem / node_choice.mem 
            reward = - max(cpu_pct, mem_pct)
        '''
        # The more space the node has left, the higher the reward
        cpu_reward = (node_choice.cpu / self.req_cpu)
        mem_reward = (node_choice.mem / self.req_mem)

        # reward = 0

        # if cpu_reward > 1 and mem_reward > 1:
        #     return 1
        
        '''
        if cpu_reward>=1 and mem_reward>=1:
            reward = 1
        else: 
            reward = -1 
        '''
        # Proportional Approach
        
        if cpu_reward>=1 and mem_reward>=1:
            cpu_reward = cpu_reward
            mem_reward = mem_reward
            '''
            if cpu_reward<mem_reward:
                if action ==cpu_index_max:
                    cpu_reward = cpu_reward + 10
                
            elif mem_reward<cpu_reward:
                if action ==mem_index_max:
                    mem_reward = mem_reward + 10
            '''
            
        #elif cpu_reward<1 and  mem_reward<1:
        #    cpu_reward = -abs(cpu_reward)
        #    mem_reward = -abs(mem_reward)
        else:
            cpu_reward = -cpu_reward
            mem_reward = -mem_reward
            
        reward = cpu_reward  + mem_reward
        #print('reward :  ', reward)
        #reward = min(cpu_reward, mem_reward)

        if node_choice.cpu>=self.req_cpu and node_choice.mem >= self.req_mem:
            node_choice.update_state(cpu_val = - self.req_cpu, mem_val= - self.req_mem)
        #print(reward)
        return reward 

    def reset_master(self):
        self.set_node_list()


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, number_of_nodes, mask_nodes, data, train ):
        super(CustomEnv, self).__init__()
        self.number_of_nodes=number_of_nodes
        self.master = Master(number_of_nodes, data, train)
        #print('master _ print : ', self.master)
        #a=b
        self.mask_nodes = mask_nodes
        self.step_counter = 0
        self.reset()
        self.action_space = spaces.Discrete(self.master.action_space)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.master.observation_space_dims, dtype=np.float64)
        self.train = train
        self.reward_list = []
        
        self.number_of_masked_nodes = choice([i for i in range(int(self.number_of_nodes/2))])
        task = self.generate_new_task()
        self.update_incoming_task(task) 
        self.data_len = len(self.master.data[0][:])


    def set_train_param(self, param):
        self.train = param
        self.master.train = param
        #self.master.set_node_list()


    def valid_action_mask(self):
        actions = [i for i in range(self.number_of_nodes)]
        #mask = sample(actions,  self.mask_nodes)
        mask = sample(actions,  np.random.randint(0,self.mask_nodes))
        #print('len(mask) : ', len(mask))
        self.master.mask_list = [1 if i in mask else 0 for i in range(self.number_of_nodes)]
        self.master.mask_list = self.all_valid_action_mask()
        return self.master.mask_list

    def all_valid_action_mask(self):
        """Returns an action mask of all ones for debugging purposes"""
        valid_mask = np.ones(self.number_of_nodes)
        return valid_mask

    def ordered_valid_action_mask(self) -> np.array:
        """This method masks nodes like it would happen in the kubernetes cluster, meaning the masked nodes get removed from the end of a list,
        resulting in a masked array that looks something like this:

        [1, 1, 1, 1, 1, 1, 0, 0, 0]

        where only the last values of the arrays are those nodes that are not available.

        Currently we assume that there is always at least one node available.

        Returns:
            np.array: The boolean mask of available nodes
        """
        valid_mask = np.ones(self.number_of_nodes)
        masked_node_num = np.random.randint(low=0, high=self.number_of_nodes)
        if masked_node_num:
            valid_mask[-masked_node_num:] = 0
        self.master.mask_list = valid_mask
        return valid_mask


    def reset(self):
        """This method is used to reset the state of the environment to a random new one, so each node gets randomly
        new cpu and memory values assigned and also updates the task information

        Returns:
            np.array: The observation including information about the nodes and the task.
        """
        #print('starting reset')
        
        #print('Before  Reset self.master.get_master_observation_space() : ', self.master.get_master_observation_space())
        #task = self.generate_new_task()
        #self.update_incoming_task(task) 
        self.master.reset_master()
        #print('reset done')
        #print()
        #print()
        #print('After Reset self.master.get_master_observation_space() : ', self.master.get_master_observation_space())
        return self.master.get_master_observation_space()

    def get_random_action(self):
        return self.master.get_random_action()
        

    def generate_random_task(self):
        req_cpu = np.random.randint(0, self.master.max_available_cpu/4)
        req_mem = np.random.uniform(0, self.master.max_available_memory/8)
        task = [0,0,0, req_cpu, req_mem]
        return task


    def generate_new_task(self):
        data = self.master.data
        task = [data[0][self.step_counter], data[1][self.step_counter], data[2][self.step_counter], data[3][self.step_counter], data[4][self.step_counter]]
        return task

    def get_done_status(self, observation):
        done = True
        for i in range(int(len(observation)/2)-1):
            cpu_i = 2*i
            mem_i = 2*i+1 
            if observation[cpu_i]>=self.master.req_cpu and observation[mem_i]>=self.master.req_mem:
                #print('done False : ', False)
                return False
        return done
    def step(self, action:int):
        """ One step in the environment. Takes the action determined by the agent and calculates the reward.
        The Episode is considered done once we analyzed all datapoints.

        Args:
            action (int): Number of the node the task should be scheduled to.

        Returns:
            Tuple: observation, reward, done, info
        """
        #print('Before action : ', action, self.master.get_master_observation_space(), self.master.req_cpu, self.master.req_mem)
        reward = self.master.execute_action(action)

        #done = False
        info = {}
        self.reward_list.append(reward)  
        self.step_counter = self.step_counter + 1
        task = self.generate_new_task()
        self.update_incoming_task(task) 
        observation_ = self.master.get_master_observation_space()
        #print('After action : ', action, observation_, self.master.req_cpu, self.master.req_mem)
        done = self.get_done_status(observation_)
        
        #self.step_counter +=1 
        if self.step_counter == self.data_len -1:
            print('set self.step_counter reset at step count : ', self.step_counter)
            self.step_counter = 0
            print('After set self.step_counter to : ', self.step_counter)
            done = True
            
            #done = True # done needs to be set to True at some point for the EvalCallback to finish
        #a=b
        
        if done ==True:
            print('Done true ? : ', done,self.step_counter)
            print('self.master.req_cpu, self.master.req_mem : ', self.master.req_cpu, self.master.req_mem)
            print('observation_ : ', observation_)
            #a=b
        
        #if self.step_counter ==50:
        #    a=b
        return observation_, reward, done, info
        

    def update_incoming_task(self, task):
        self.master.set_incoming_task(task)
        

    def render(self, mode='human'):
        pass


    def close (self):
        self.master.reset_master()

