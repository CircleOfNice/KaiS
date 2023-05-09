



import gym 
from gym import spaces
# from components.Master import Master
from tqdm import tqdm
#from components.Node import Node
import numpy as np
from gym.spaces.box import Box
from random import shuffle, choice, sample
import random
from scipy.stats import entropy 
from scipy.stats import variation , pearsonr
class Node:
    """[This class serves as framework for definition of Edge Node with properties such as 
    task queue, service_list, cpu processing and  memory]
    """
    def __init__(self, cpu:float, mem:float, max_cpu:float, max_mem:float):
        """Init method for a node
        Args:
            cpu (float): Current free CPU space for a node. Measured in millicores like kubernetes
            mem (float): Current free Memory space for a node. Measured in Gigabytes.
            max_cpu (_type_): Maximum CPU capacity for a node
            max_mem (_type_): Maximum Memory capacity for a node
        """
        self.cpu = cpu
        self.mem = mem
        self.max_cpu = max_cpu
        self.max_mem = max_mem


    def __str__(self):
        return f"(cpu : {self.cpu}, mem: {self.mem})"
    

    def get_current_state_data(self):
        state = [self.cpu, self.mem, self.max_cpu, self.max_mem]
        return state
    

    def update_state(self, cpu_val:float, mem_val:float):
        """Method to update the state of a node. 
        Args:
            cpu_val (float): Adds this value to the free CPU space of this Node.
            mem_val (float): Adds this value to the free memory space of this Node.
        """
        self.cpu = self.cpu + cpu_val
        self.mem = self.mem + mem_val


class Master:
    """[This class serves as framework for definition of Master Node with properties such as 
    task queue, cpu processing, memory, done and undone tasks, Kind of tasks done and undone. all task index]
    """
    def __init__(self, number_of_nodes:int, task_data:list, normalize_obs=False):
        self.number_of_nodes = number_of_nodes
        self.max_available_cpu_choices = [1000, 2000, 4000, 8000]
        self.max_available_mem_choices = [1, 2, 4, 8, 16, 33.4916444]

        self.max_cpu_value = max(self.max_available_cpu_choices)
        self.max_mem_value = max(self.max_available_mem_choices)
        self.normalize_obs = normalize_obs

        self.mask_list = [1 for _ in range(self.number_of_nodes)]
        
        self.current_incoming_task = [0, 0, 0, 0, 0]
        self.req_cpu_current_task = 0
        self.req_mem_current_task = 0

        self.init_node_list()
        self.action_space = len(self.node_list)
        self.observation_space_dims = self.get_master_observation_space().shape
        self.task_data = task_data
        self.action_value = 0

        # Array to count the number of scheduling decisions for each node
        self.action_distribution = np.zeros(number_of_nodes)
        # Counter for how many times the cluster was completely full
        self.max_capacity_count = 0 
        # Counter for how often the environment was reset
        self.invalid_decision_counter = 0
    
    def __str__(self):
        new_str ='\n'

        for x in self.node_list:
            new_str+=str(x)
            new_str+='\n'

        return new_str


    def init_node_list(self, init_random:bool=True):
        """This method is used to set the cpu and memory values of all nodes to a random value within the allowed interval
        Args:
            init_random (bool, optional): If True, initializes the cluster with random CPU and Memory usages. If False
            set current cpu and mem usage for each node to 0. Defaults to False.
        """
        cpu_params= []
        mem_params= []
        max_cpu_params = [random.choice(self.max_available_cpu_choices) for i in range(self.number_of_nodes)]
        max_mem_params = [random.choice(self.max_available_mem_choices) for i in range(self.number_of_nodes)]

        for i in range(self.number_of_nodes):
            if init_random:
                cpu_params.append(np.random.randint(0,max_cpu_params[i]))
                mem_params.append(np.random.uniform(0,max_mem_params[i]))
            else:
                cpu_params.append(max_cpu_params[i])
                mem_params.append(max_mem_params[i])


        self.node_list = [Node(cpu = cpu_params[i], mem = mem_params[i], max_cpu=max_cpu_params[i], max_mem=max_mem_params[i]) for i in range(self.number_of_nodes)]


    def check_remaining_node_space(self) -> bool:
        """Method to check wether the cluster has still space to execute the next incoming task
        Also respects the masking caused by maskedPPO.
        Returns:
            bool: Returns true if at least one node has enough cpu and memory capacity to execute the task. False otherwise
        """
        for idx, node in enumerate(self.node_list):
            if self.mask_list[idx]:
                if node.cpu >= self.req_cpu_current_task and node.mem >= self.req_mem_current_task:
                    return True
        
        return False
    

    def check_node_usage(self) -> bool:
        """Method to check wether there are nodes who got negative available cpu or memory space.
        If yes, this means the node got a task which it would not be able to process
        Returns:
            bool: Returns False if there are some Nodes who have negative available space. Returns True otherwise.
        """
        for node in self.node_list:
            if node.cpu < 0:
                return False
            if node.mem < 0:
                return False
        
        return True
    

    def set_incoming_task(self, task):
        self.current_incoming_task = task
        self.req_cpu_current_task = task[3]
        self.req_mem_current_task = task[4]
        

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
            master_observation_space.append((state_normalisation[0], state_normalisation[1], state_normalisation[2], state_normalisation[3]))# , state_normalisation[2], state_normalisation[3]))

        # Get information of the task
        master_observation_space.append((self.current_incoming_task[3], self.current_incoming_task[4]))#, 0,0))

        master_observation_space = np.hstack(master_observation_space)

        # This normalization relies on the fact that every 2nd entry in the array describes some cpu value, and every other value
        # describes a memory value
        if self.normalize_obs:
            master_observation_space[::2] = master_observation_space[::2] / self.max_cpu_value
            master_observation_space[1::2] = master_observation_space[1::2] / self.max_mem_value

        return master_observation_space
    

    def get_random_action(self):
        action = np.random.choice(self.action_space, 1)
        return action[0]
    

    def find_low_resource_nodes(self):
        """Method to find the index of the node with lowest cpu usage and lowest memory usage, returns as tuple.
        Respects the mask, does not consider if the current task can currently be executed on said node"""
        max_cpu = -1
        max_cpu_index = -1

        max_mem = -1
        max_mem_index = -1

        for idx, node in enumerate(self.node_list):
            if self.mask_list[idx]:
                if node.cpu > max_cpu:
                    max_cpu = node.cpu
                    max_cpu_index = idx

                if node.mem > max_mem:
                    max_mem = node.mem
                    max_mem_index = idx

        return max_cpu_index, max_mem_index
    
    def get_utilisation_ratios(self):
        cpu_utilisation = []
        mem_utilisation = []
        for i , node in enumerate(self.node_list):
            # If node is masked, set available cpu and memory space to 0
            cpu_utilisation.append(node.cpu/node.max_cpu)
            mem_utilisation.append(node.mem/node.max_mem)
        return cpu_utilisation, mem_utilisation
    
    def get_std_deviations(self, cpu_utilisation, mem_utilisation):
        std_cpu = np.std(cpu_utilisation, ddof=1)
        std_mem = np.std(mem_utilisation, ddof=1)
        return std_cpu, std_mem
    
    def get_entropy(self, cpu_utilisation, mem_utilisation):
        entropy_cpu = entropy(cpu_utilisation)
        entropy_mem = entropy(mem_utilisation)
        return entropy_cpu, entropy_mem
    
    
    def get_coefficient_of_variation(self, cpu_utilisation, mem_utilisation):
        coeff_cpu = variation(cpu_utilisation)
        coeff_mem = variation(mem_utilisation)
        return coeff_cpu, coeff_mem
    
    def get_standard_deviation_reward(self, cpu_utilisation, mem_utilisation):
        std_cpu, std_mem = self.get_std_deviations(cpu_utilisation, mem_utilisation)
        
        std_cpu_reward = 1/(1+std_cpu)
        std_mem_reward = 1/(1+std_mem)
        return std_cpu_reward + std_mem_reward
    
    def get_entropy_reward(self, cpu_utilisation, mem_utilisation, action):
        #entropy_cpu, entropy_mem= self.get_entropy(cpu_utilisation, mem_utilisation)
        #entropy_cpu, entropy_mem= self.get_entropy(cpu_utilisation, mem_utilisation)
        
        #entropy_std =  np.std([entropy_cpu, entropy_mem], ddof=1)
        
        #entropy_reward = 1/(1+entropy_std)
        
        calculated_relative_entropies = []
        calculated_relative_rewards = []
        for i in range(len(cpu_utilisation)):
            ent = entropy([cpu_utilisation[i], mem_utilisation[i]])
            #print(ent, 1/(1+ent))
            calculated_relative_entropies.append(ent)
            calculated_relative_rewards.append(1/(1+ent))
            
        #print(calculated_relative_entropies)
        #print('calculated_relative_rewards : ', calculated_relative_rewards)
        
        return calculated_relative_rewards[action]
    
    
    def reward_chat_gpt_sd_entropy(self, cpu_utilisation, mem_utilisation):
        w1, w2, w3, w4 = [0.20, 0.20, 0.20, 0.20]
        entropy_cpu, entropy_mem = self.get_entropy(cpu_utilisation, mem_utilisation)
        std_cpu, std_mem = self.get_std_deviations(cpu_utilisation, mem_utilisation)
        coeff_cpu, coeff_mem = self.get_coefficient_of_variation( cpu_utilisation, mem_utilisation)
        
        corr, pval = pearsonr(cpu_utilisation, mem_utilisation) 
        #print(corr, pval)
        #print('entropy_cpu, entropy_mem : ', entropy_cpu, entropy_mem)
        #print('std_cpu, std_mem : ', std_cpu, std_mem)
        
        Load_Balance_Score = w1*entropy_cpu + w2*entropy_mem + w3*std_cpu + w4*std_mem - (1-w1-w2-w3-w4)*coeff_cpu*coeff_mem*(1-abs(corr))
        
        #print('Load_Balance_Score : ', Load_Balance_Score)
        return Load_Balance_Score
        
        
        
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

        # This calculation needs to happen, _before_ the nodes are updated
        reward = 0
        max_cpu_index, max_mem_index = self.find_low_resource_nodes()

        if action == max_cpu_index:
            reward += 1
        if action == max_mem_index:
            reward += 1

        node_choice.update_state(cpu_val = - self.req_cpu_current_task, mem_val= - self.req_mem_current_task)

        # if not selfmax_capa.check_remaining_node_space():
        #     self.city_count += 1
        
        
        # cpu_utilisation, mem_utilisation = self.get_utilisation_ratios()
        #std_cpu = np.std(cpu_utilisation, ddof=1)
        #std_mem = np.std(mem_utilisation, ddof=1)
        
        # std_mem = np.std([std_cpu, std_mem], ddof=1)
        # reward = np.exp(1/(1 + np.exp(-(std_cpu+ std_mem))))

        #reward = np.exp(-1/(1 + np.exp(-(std_cpu + std_mem))))
        #std_reward = self.get_standard_deviation_reward(cpu_utilisation, mem_utilisation)
        #entropy_reward = self.get_entropy_reward( cpu_utilisation, mem_utilisation, action)
        #reward = entropy_reward + std_reward
        # reward = self.reward_chat_gpt_sd_entropy( cpu_utilisation, mem_utilisation)
        return reward 

    def reset_master(self):
        self.init_node_list()


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, number_of_nodes:int, mask_nodes:int, data:list, normalize_obs:bool):
        super(CustomEnv, self).__init__()
        self.number_of_nodes=number_of_nodes
        self.master = Master(number_of_nodes, data, normalize_obs=normalize_obs)
        self.mask_nodes = mask_nodes
        self.step_counter = 0
        self.reset()
        self.action_space = spaces.Discrete(self.master.action_space)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.master.observation_space_dims, dtype=np.float64)
        self.reward_list = []
        
        self.number_of_masked_nodes = choice([i for i in range(int(self.number_of_nodes/2))])
        task = self.generate_task()
        self.update_incoming_task(task) 
        self.data_len = len(self.master.task_data[0][:])


    def get_action_mask(self):
        """Wrapper to call the action mask"""
        return self.ordered_valid_action_mask()
    
    def valid_action_mask(self):
        actions = [i for i in range(self.number_of_nodes)]
        mask = sample(actions,  np.random.randint(0,self.mask_nodes))
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
        self.master.init_node_list()
        
        return self.master.get_master_observation_space()

    def get_random_action(self):
        return self.master.get_random_action()
        

    def generate_task(self):
        """ Simple Wrapper for task generation """
        return self.sample_task_from_kubernetes_data_set()
        # return self.generate_random_task()
    

    def generate_random_task(self):
        """ Method that returns a tasks with random required resources.
        As a result there might be no node, which is able to handle the given task"""
        req_cpu = np.random.randint(4, max(self.master.max_available_cpu_choices) // 10 )
        req_mem = np.random.uniform(0.001, max(self.master.max_available_mem_choices) / 10)
        task = [0,0,0, req_cpu, req_mem]
        return task

    
    def sample_task_from_kubernetes_data_set(self):
        """Samples task from real Kuberenetes choices"""
        data = self.master.task_data
        data_len = len(data[0])
        rand_count =  np.random.randint(0, data_len)
        task = [data[0][rand_count], data[1][rand_count], data[2][rand_count], data[3][rand_count], data[4][rand_count]]
        return task


    def generate_new_task(self):
        data = self.master.task_data
        task = [data[0][self.step_counter], data[1][self.step_counter], data[2][self.step_counter], data[3][self.step_counter], data[4][self.step_counter]]
        return task


    def generate_new_simple_task(self):
        """ Method that returns always the same simple task for debugging purposes"""
        task = [0, 0, 0, 1, 0.01]
        return task
    

    def get_done_status(self, observation):
        """ Method to get boolean as whether the episode has ended"""
        done = False

        node_valid_flag = self.master.check_node_usage()

        if not node_valid_flag:
            self.master.invalid_decision_counter += 1
            done = True

            if not self.master.check_remaining_node_space():
                self.master.max_capacity_count += 1
            
        return done
    

    def step(self, action:int):
        """ One step in the environment. Takes the action determined by the agent and calculates the reward.
        The Episode is considered done once we analyzed all datapoints.
        Args:
            action (int): Number of the node the task should be scheduled to.
        Returns:
            Tuple: observation, reward, done, info
        """
        
        reward = self.master.execute_action(action)

        info = {}
        
        self.step_counter = self.step_counter + 1
        task = self.generate_task()
        self.update_incoming_task(task) 
        observation_ = self.master.get_master_observation_space()
       
        done = self.get_done_status(observation_)
        
        if self.step_counter == self.data_len -1:
            self.step_counter = 0
            # done = True
        
        if done:
            reward = -1

        self.reward_list.append(reward)      
        return observation_, reward, done, info
        

    def update_incoming_task(self, task):
        """ Method to set the current task"""
        self.master.set_incoming_task(task)
        

    def render(self, mode='human'):
        pass


    def close (self):
        self.master.reset_master()