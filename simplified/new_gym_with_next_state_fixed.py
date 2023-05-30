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
from scipy.stats import truncnorm


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
    def __init__(self, number_of_nodes:int, task_data:list, normalize_obs=False, init_random=True, init_uniform=False):
        """_summary_

        Args:
            number_of_nodes (int): Number of nodes in the cluster
            task_data (list): List of tasks to generate sample tasks from
            normalize_obs (bool, optional): Wether to normalize observations. Defaults to False.
            init_random (bool, optional): Wether to give nodes random initial values. Defaults to True.
            init_uniform (bool, optional): Wether to give all nodes in the cluster same resources. Defaults to False.
        """
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

        self.init_random = init_random
        self.init_uniform = init_uniform

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
        
        
        self.avg_mem_utilisation_ratios = []
        self.avg_cpu_utilisation_ratios = []
        self.avg_std_cpu = []
        self.avg_std_mem = []
        self.avg_ent_cpu = []
        self.avg_ent_mem = []
        self.avg_coeff_cpu = []
        self.avg_coeff_mem = []
        self.avg_rel_entropy_per_node = []
        
        
    def __str__(self):
        new_str ='\n'

        for x in self.node_list:
            new_str+=str(x)
            new_str+='\n'

        return new_str


    def init_node_list(self):
        """This method is used to set the cpu and memory values of all nodes to a random value within the allowed interval
        Args:
            init_random (bool, optional): If True, initializes the cluster with random CPU and Memory usages. If False
            set current cpu and mem usage for each node to 0. Defaults to False.
        """
        cpu_params= []
        mem_params= []

        if self.init_uniform:
            # Gives all nodes same max cpu and memory resources
            cpu_size = random.choice(self.max_available_cpu_choices)
            mem_size = random.choice(self.max_available_mem_choices)
            max_cpu_params = [cpu_size for _ in range(self.number_of_nodes)]
            max_mem_params = [mem_size for _ in range(self.number_of_nodes)]
        else:
            # Gives all nodes random max cpu and memory resources
            max_cpu_params = [random.choice(self.max_available_cpu_choices) for i in range(self.number_of_nodes)]
            max_mem_params = [random.choice(self.max_available_mem_choices) for i in range(self.number_of_nodes)]

        for i in range(self.number_of_nodes):
            if self.init_random:
                # Initialize nodes with random cpu and memory usage
                cpu_params.append(np.random.randint(0,max_cpu_params[i]))
                mem_params.append(np.random.uniform(0,max_mem_params[i]))
            else:
                # Initialize empty nodes
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
        """Method to set appropriate tasks as requirements

        Args:
            task (list): list containing the data for next incoming data
        """
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
        for i, node in  enumerate(self.node_list):
            
            state_normalisation = node.get_current_state_data()
            #master_observation_space.append((state_normalisation[0], state_normalisation[1], state_normalisation[2], state_normalisation[3]))# , state_normalisation[2], state_normalisation[3]))
            #TODO check where Exactly where masking of output nodes is happening
            mask_info = self.mask_list[i]
            master_observation_space.append((state_normalisation[0]*mask_info, state_normalisation[1]*mask_info, state_normalisation[2], state_normalisation[3]))
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
        """Method to get random action out of action spaces

        Returns:
            action: action as an integer
        """
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
    
    def get_utilisation_ratios(self, masking =True):
        """Method to calculate Utilisation Ratios of CPU and Memory

        Returns:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio
        """
        cpu_utilisation = []
        mem_utilisation = []
        for i , node in enumerate(self.node_list):
            # If node is masked, set available cpu and memory space to 0
            if masking:
                mask_info = self.mask_list[i]
                #print("i, mask_info : ", i, mask_info)
                if mask_info==1:
                    cpu_utilisation.append(node.cpu/node.max_cpu)
                    mem_utilisation.append(node.mem/node.max_mem)
                    
            else:
                cpu_utilisation.append(node.cpu/node.max_cpu)
                mem_utilisation.append(node.mem/node.max_mem)
        return cpu_utilisation, mem_utilisation
    

    def get_normalized_utilization_ratios(self):
        """Method to return the cpu utilization ratios for each resource (CPU, Memory), divided by the sum of the respective
        resource.

        Returns:
            Tuple: cpu_utilisation_array, memory_utilisation_array
        """
        node_num = len(self.node_list)
        cpu_utilisation_arr = np.zeros(node_num)
        mem_utilisation_arr = np.zeros(node_num)
        for idx , node in enumerate(self.node_list):
            #TODO Check it here for state update
            if self.mask_list==1:
                cpu_utilisation_arr[idx] = node.cpu
                mem_utilisation_arr[idx] = node.mem

        cpu_utilisation_arr = cpu_utilisation_arr / np.sum(cpu_utilisation_arr)
        mem_utilisation_arr = mem_utilisation_arr / np.sum(mem_utilisation_arr)
        return cpu_utilisation_arr, mem_utilisation_arr

    def get_std_deviations(self, cpu_utilisation, mem_utilisation):
        """Method to calculate standard deviations across CPU and Memory Utilisation

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            std_cpu: Standard Deviation across CPU utilsation ratios
            std_mem: Standard Deviation across Memory utilsation ratios
        """
        std_cpu = np.std(cpu_utilisation, ddof=1)
        std_mem = np.std(mem_utilisation, ddof=1)
        return std_cpu, std_mem
    
    def get_entropy(self, cpu_utilisation, mem_utilisation):
        """Method to calculate entropy across CPU and Memory Utilisation

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            entropy_cpu: Entropy across CPU utilsation ratios
            entropy_mem: Entropy across Memory utilsation ratios
        """
        entropy_cpu = entropy(cpu_utilisation)
        entropy_mem = entropy(mem_utilisation)
        return entropy_cpu, entropy_mem
    
    def get_relative_avg_entropy_per_node(self, cpu_utilisation, mem_utilisation):
        
        """Method to calculate average entropy across each node for its CPU and Memory Utilisation

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            relative_avg_entropy_per_node: Relative Average Entropy per Node
        """
        
        list_of_entropy = []
        for i in range(len(cpu_utilisation)):
            list_of_entropy.append(entropy([cpu_utilisation[i], mem_utilisation[i]]))
        relative_avg_entropy_per_node = sum(list_of_entropy)/len(list_of_entropy)
        return relative_avg_entropy_per_node
    
    def get_coefficient_of_variation(self, cpu_utilisation, mem_utilisation):
        """Method to calculate coefficient of variation across CPU and Memory Utilisation ratios

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            coeff_cpu: Coefficient of variation  across CPU utilsation ratios
            coeff_mem: Coefficient of variation  across Memory utilsation ratios
        """
        coeff_cpu = variation(cpu_utilisation)
        coeff_mem = variation(mem_utilisation)
        return coeff_cpu, coeff_mem
    
    def get_coefficient_of_variation_reward(self, cpu_utilisation, mem_utilisation):
        """Method to calculate reward based on coefficient of variation of CPU Utilisation and Memory Utilisation ratio

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            coff_reward: calculated coefficient of variation based reward
        """
        
        coeff_cpu, coeff_mem= self.get_coefficient_of_variation(cpu_utilisation, mem_utilisation)
        
        #coeff_cpu_reward = 1/(1+coeff_cpu)
        #coeff_mem_reward = 1/(1+coeff_mem)
        
        coeff_cpu_reward =  (np.exp(np.exp((1/(1+coeff_cpu))-0.5)-1)-1) / (np.exp(np.exp((1/(1))-0.5)-1)-1 )
        coeff_mem_reward = (np.exp(np.exp((1/(1+coeff_mem))-0.5)-1)-1) / (np.exp(np.exp((1/(1))-0.5)-1)-1 )
        coff_reward = coeff_cpu_reward + coeff_mem_reward
        return coff_reward
    
    
    def get_standard_deviation_reward(self, cpu_utilisation, mem_utilisation):
        """Method to calculate reward based on standard deviation of CPU Utilisation and Memory Utilisation ratio

        Args:
            cpu_utilisation: CPU Utilisation Ratioe
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            std_reward: calculated standard deviation based reward
        """
        
        std_cpu, std_mem = self.get_std_deviations(cpu_utilisation, mem_utilisation)
        
        #std_cpu_reward = np.exp((1/(1+std_cpu))-0.5)-1
        #std_mem_reward = np.exp((1/(1+std_mem))-0.5)-1
        
        #std_cpu_reward = np.power(10, (1/(1+std_cpu))-0.5)-1
        #std_mem_reward = np.power(10, (1/(1+std_mem))-0.5)-1
        
        #std_cpu_reward = np.power(10, np.power(10, (1/(1+std_cpu))-0.5)-1)-1
        #std_mem_reward = np.power(10, np.power(10, (1/(1+std_mem))-0.5)-1)-1
        #TODO reward variations needed to be tried
        
        std_cpu_reward =  (np.exp(np.exp((1/(1+std_cpu))-0.5)-1)-1) / (np.exp(np.exp((1/(1))-0.5)-1)-1 )
        std_mem_reward = (np.exp(np.exp((1/(1+std_mem))-0.5)-1)-1) / (np.exp(np.exp((1/(1))-0.5)-1)-1 )
        std_reward = std_cpu_reward + std_mem_reward
        
        #scaling_factor = (np.power(power, np.power(power, (1/(1))-0.5)-1)-1)
        #std_reward = (std_cpu_reward/scaling_factor) + (std_mem_reward / scaling_factor)
        #std_reward = min(std_cpu_reward, std_mem_reward)
        #std_reward = np.std([std_cpu_reward , std_mem_reward], ddof=1)
        #std_reward = np.power(10, (1/(1+std_reward))-0.5)-1
        return std_reward
    
    def get_entropy_reward(self, cpu_utilisation, mem_utilisation):
        """Method to calculate reward based on Entropy of CPU Utilisation and Memory Utilisation ratio

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            entropy_reward: calculated Entropy based reward
        """
        calculated_relative_rewards = []
        for i in range(len(cpu_utilisation)):
            ent = entropy([cpu_utilisation[i], mem_utilisation[i]])
            normalised_ent_reward = (np.exp(np.exp((1/(1+ent))-0.5)-1)-1) / (np.exp(np.exp((1/(1))-0.5)-1)-1 )
            calculated_relative_rewards.append(1/(1+normalised_ent_reward))
            
            #calculated_relative_rewards.append(1/(1+ent))
        entropy_reward = sum(calculated_relative_rewards)/len(calculated_relative_rewards)    
        return entropy_reward
    
    
    def reward_chat_gpt_sd_entropy(self, cpu_utilisation, mem_utilisation):
        """Method to calculate suggested chatgpt reward 

        Args:
            cpu_utilisation: CPU Utilisation Ratio
            mem_utilisation: Memory Utilisation Ratio

        Returns:
            Load_Balance_Score: calculated suggested chatgpt reward
        """
        calculated_relative_rewards = []
        for i in range(len(cpu_utilisation)):
            ent = entropy([cpu_utilisation[i], mem_utilisation[i]])
            calculated_relative_rewards.append(1/(1+ent))
        entropy_reward = sum(calculated_relative_rewards)/len(calculated_relative_rewards)  
        w1, w2, w3, w4 = [0.20, 0.20, 0.20, 0.20]
        entropy_cpu, entropy_mem = self.get_entropy(cpu_utilisation, mem_utilisation)
        std_cpu, std_mem = self.get_std_deviations(cpu_utilisation, mem_utilisation)
        coeff_cpu, coeff_mem = self.get_coefficient_of_variation( cpu_utilisation, mem_utilisation)
        
        corr, pval = pearsonr(cpu_utilisation, mem_utilisation) 
        Load_Balance_Score = w1*entropy_cpu + w2*entropy_mem + w3*std_cpu + w4*std_mem - (1-w1-w2-w3-w4)*coeff_cpu*coeff_mem*(1-abs(corr))
        
        return Load_Balance_Score
    

    def log_statistical_info(self):  
        """
        Method to log statistical information regarding the usage of CPU and Memory.
        """      
        cpu_utilisation, mem_utilisation = self.get_utilisation_ratios(masking=False)
        entropy_cpu, entropy_mem = self.get_entropy(cpu_utilisation, mem_utilisation)
        std_cpu, std_mem= self.get_std_deviations(cpu_utilisation, mem_utilisation)
        coeff_cpu, coeff_mem = self.get_coefficient_of_variation(cpu_utilisation, mem_utilisation)
        
        avg_rel_entropy_per_node = self.get_relative_avg_entropy_per_node(cpu_utilisation, mem_utilisation)
        
        
        avg_mem_utilisation_ratio = sum(mem_utilisation)/len(mem_utilisation)
        avg_cpu_utilisation_ratio = sum(cpu_utilisation)/len(cpu_utilisation)
        
        self.avg_mem_utilisation_ratios.append(avg_mem_utilisation_ratio)
        self.avg_cpu_utilisation_ratios.append(avg_cpu_utilisation_ratio)
        
        self.avg_std_cpu.append(std_cpu)
        self.avg_std_mem.append(std_mem)
        
        self.avg_coeff_cpu.append(coeff_cpu)
        self.avg_coeff_mem.append(coeff_mem)
        
        self.avg_ent_cpu.append(entropy_cpu)
        self.avg_ent_mem.append(entropy_mem)
        
        self.avg_rel_entropy_per_node.append(avg_rel_entropy_per_node)

        
        
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
        # reward = 0
        # max_cpu_index, max_mem_index = self.find_low_resource_nodes()

        # if action == max_cpu_index:
        #     reward += 1
        # if action == max_mem_index:
        #     reward += 1

        # cpu_reward = node_choice.cpu / max(self.max_available_cpu_choices)
        # mem_reward = node_choice.mem / max(self.max_available_mem_choices)
        # reward = (cpu_reward + mem_reward) * 2

        node_choice.update_state(cpu_val = - self.req_cpu_current_task, mem_val= - self.req_mem_current_task)

        # if not selfmax_capa.check_remaining_node_space():
        #     self.city_count += 1
        
        
        cpu_utilisation, mem_utilisation = self.get_utilisation_ratios()
        std_reward = self.get_standard_deviation_reward(cpu_utilisation, mem_utilisation)
        reward_list = [std_reward]
        reward = sum(reward_list)/len(reward_list)#std_reward  + entropy_reward  + coeff_reward

        # cpu_utilisation, mem_utilisation = self.get_normalized_utilization_ratios()
        #std_cpu = np.std(cpu_utilisation, ddof=1)
        #std_mem = np.std(mem_utilisation, ddof=1)
        
        # std_mem = np.std([std_cpu, std_mem], ddof=1)
        # reward = np.exp(1/(1 + np.exp(-(std_cpu+ std_mem))))

        #reward = np.exp(-1/(1 + np.exp(-(std_cpu + std_mem))))e

        # entropy_reward = self.get_entropy_reward( cpu_utilisation, mem_utilisation)
        # coeff_reward = self.get_coefficient_of_variation_reward( cpu_utilisation, mem_utilisation)

        #reward = self.reward_chat_gpt_sd_entropy( cpu_utilisation, mem_utilisation)
        return reward 

    def reset_master(self):
        self.init_node_list()


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, number_of_nodes:int, mask_nodes:int, data:list, normalize_obs:bool, init_random:bool=True, init_uniform:bool=False,
                 no_masking_prob=1):
        """Custom environment representing a kubernetes cluster with multiple nodes

        Args:
            number_of_nodes (int): Number of nodes in the cluster
            mask_nodes (int): Maximum number of masked nodes (used to simulate node-outage)
            data (list): List of tasks to potentially sample from
            normalize_obs (bool): 
            init_random (bool, optional): Gives nodes random starting cpu and memory usage. Defaults to True.
            init_uniform(bool, optional): Initializes max cpu and memory values of all nodes uniform
            masking_prob(bool, optional): Probability of applying no masking at the start of an episode, leaving all nodes available (from 0 to 1)
        """
        super(CustomEnv, self).__init__()
        self.number_of_nodes = number_of_nodes
        self.master = Master(number_of_nodes, data, normalize_obs=normalize_obs, init_random=init_random, init_uniform=init_uniform)
        self.mask_nodes = mask_nodes
        self.step_counter = 0
        # self.reset()
        self.action_space = spaces.Discrete(self.master.action_space)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.master.observation_space_dims, dtype=np.float64)
        self.reward_list = []
        
        self.number_of_masked_nodes = choice([i for i in range(int(self.number_of_nodes/2))])
        task = self.generate_task()
        self.update_incoming_task(task) 
        self.data_len = len(self.master.task_data[0][:])
        self.no_masking_prob = no_masking_prob
        self.initial_standard_mask(no_mask_prob=self.no_masking_prob)
        #print("Done initialising")

    def get_action_mask(self):
        """Wrapper to call the action mask"""
        return self.ordered_valid_action_mask()
    
    def valid_action_mask(self):
        actions = [i for i in range(self.number_of_nodes)]
        mask = sample(actions,  np.random.randint(2,self.mask_nodes)) #chaning masked nodes for stability purposes
        #print("mask : ", mask)
        #TODO add this minimum masking information to environment as a variable
        self.master.mask_list = [1 if i in mask else 0 for i in range(self.number_of_nodes)]
        #self.master.mask_list = self.all_valid_action_mask()
        #print("self.master.mask_list : ", self.master.mask_list)
        return self.master.mask_list

    
    def initial_standard_mask(self, no_mask_prob=1):
        
        valid_mask = np.ones(self.number_of_nodes)
        test_condition = random.random()
        if no_mask_prob < test_condition:
            masked_node_num = np.random.randint(low=0, high=self.mask_nodes)
            
            if masked_node_num:
                valid_mask[-masked_node_num:] = 0
        self.master.mask_list = valid_mask

    def all_valid_action_mask(self):
        """Returns an action mask of all ones for debugging purposes"""
        valid_mask = np.ones(self.number_of_nodes)
        return valid_mask

    def repeatable_ordered_valid_action_mask(self) -> np.array:
        """This method masks nodes like it would happen in the kubernetes cluster, meaning the masked nodes get removed from the end of a list,
        resulting in a masked array that looks something like this:
        [1, 1, 1, 1, 1, 1, 0, 0, 0]
        where only the last values of the arrays are those nodes that are not available.
        Currently we assume that there is always at least one node available.
        Returns:
            np.array: The boolean mask of available nodes
        """
        #print(self.master.mask_list)
        return self.master.mask_list
    
    

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
        self.initial_standard_mask(no_mask_prob=self.no_masking_prob)
        self.master.reset_master()
        
        return self.master.get_master_observation_space()

    def get_random_action(self):
        """Method to sample a random action

        Returns:
            action: integer value for the action
        """
        return self.master.get_random_action()
        

    def generate_task(self):
        """ Simple Wrapper for task generation """
        # if random.random() > 0.5:
        #     task = self.sample_task_from_kubernetes_data_set()
        # else:
        task = self.generate_random_task()
        return task
    
    

    def get_truncated_normal(self, mean=0, sd=1, low=0, upper=10):
        """Generates a scipy sample for sampling purposes

        Args:
            mean (int, optional): mean for the sampler object. Defaults to 0.
            sd (int, optional): standard deviation for the sampler object. Defaults to 1.
            low (int, optional): lowest possible value that can be sampled. Defaults to 0.
            upper (int, optional): highest possible value that can be sampled. Defaults to 10.

        Returns:
            Scipy sammpler: Scipy sampler object 
        """
        return truncnorm(
            (low - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)

    def get_truncated_norm_cpu_mem_data(self, mean_cpu, std_cpu, mean_mem, std_mem, max_factor = 7, datapoints =1):
        
        """Generates a sample for cpu and memory value resembling the tasks in the kubernetes data
        Args:
            mean_cpu (int, optional): mean cpu for the sampler object. Defaults to 0.
            std_cpu (int, optional): standard deviation for cpu for the sampler object. Defaults to 0.
            mean_mem (int, optional): mean mem for the sampler object. Defaults to 0.
            std_mem (int, optional): standard deviation for mem for the sampler object. Defaults to 0.
            max_factor (int, optional): number of standard deviations to account for the maximum value. Defaults to 10.
            
        Returns:
            cpu_samples: sampled cpu data
            mem_samples: sampled mem data
        """
        cpu_sampler = self.get_truncated_normal(mean_cpu, std_cpu, low = 0, upper = mean_cpu + max_factor * std_cpu)
        mem_sampler = self.get_truncated_normal(mean_mem, std_mem, low = 0, upper = mean_mem + max_factor * std_mem)
        cpu_samples = cpu_sampler.rvs(datapoints)
        mem_samples = mem_sampler.rvs(datapoints)
        return cpu_samples, mem_samples
        

    def generate_random_task(self, uniform = True):
        """ Method that returns a tasks with random required resources.
        As a result there might be no node, which is able to handle the given task"""
        
        if uniform:
            req_cpu = np.random.randint(4, max(self.master.max_available_cpu_choices) // 10 )
            req_mem = np.random.uniform(0.001, max(self.master.max_available_mem_choices) / 10)
        else: 
            mean_cpu = 38.31644473313847
            mean_mem = 0.04445434537338971
            
            std_cpu = 14.824416377146973
            std_mem =  0.017403803968774088
            
            max_factor = 7 # number of standard deviations to consider for upper limit of data 
            req_cpu, req_mem = self.get_truncated_norm_cpu_mem_data(self, mean_cpu, std_cpu, mean_mem, std_mem, max_factor = max_factor, datapoints =100)
            req_cpu, req_mem = req_cpu[0], req_mem[0] 
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
        
        """ Method that returns always the same simple task for debugging purposes"""
        data = self.master.task_data
        task = [data[0][self.step_counter], data[1][self.step_counter], data[2][self.step_counter], data[3][self.step_counter], data[4][self.step_counter]]
        return task

    def sample_task_from_kubernetes_data_set(self):
        """Samples task from real Kuberenetes choices"""
        data = self.master.task_data
        data_len = len(data[0])
        rand_count =  np.random.randint(0, data_len)
        task = [data[0][rand_count], data[1][rand_count], data[2][rand_count], data[3][rand_count], data[4][rand_count]]
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
            
        # done = True
        # for i in range(int(len(observation)/2)-1):
        #     cpu_i = 2*i
        #     mem_i = 2*i+1 
        #     if observation[cpu_i]>=self.master.req_cpu_current_task and observation[mem_i]>=self.master.req_mem_current_task:
        #         #print('done False : ', False)
        #         return False
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
        if not done:
            self.master.log_statistical_info()
        self.reward_list.append(reward)      
        return observation_, reward, done, info
        

    def update_incoming_task(self, task):
        """ Method to set the current task"""
        self.master.set_incoming_task(task)
        

    def render(self, mode='human'):
        pass


    def close (self):
        self.master.reset_master()