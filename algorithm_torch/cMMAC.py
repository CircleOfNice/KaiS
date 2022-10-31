# Coordinated Multi-Agent Actor-Critic (cMMAC)
import math
import numpy as np
from copy import deepcopy
from algorithm_torch.CMMAC_Policy_Model import *
from algorithm_torch.CMMAC_Value_Model import *
from algorithm_torch.ReplayMemory import *
from typing import Union, Tuple, Type

class Estimator:
    """Class to Define the cMMAC (Actor Critic) model
    """
    def __init__(self, action_dim:int, state_dim:int, number_of_master_nodes:int):
        """Initialisation of arguments

        Args:
            action_dim (int): Dimensions of the output (actions)
            state_dim (int): Dimensions of the input state
            
            number_of_master_nodes (int): [number of eAPs]
        """
        self.number_of_master_nodes = number_of_master_nodes
        self.action_dim = action_dim
        self.state_dim = state_dim
        # Initial value for losses
        self.entropy = 1
        self.pm, self.pm_optimizer = build_policy_model(self.state_dim, self.action_dim)
        
    def action(self, state:np.array, critic: Type[Value_Model], critic_state: np.array, ava_node:list, context: bool)-> Tuple[list, np.array, np.array, np.array,list, np.array, list]:
        """

        Args:
            state ([Numpy Array]): [State Array]
            critic: Global Critic
            critic_state: Critic State
            ava_node ([list]): [currently deployed nodes] #[Confusing name it is the nodes which are currently deployed]
            context ([list]): [Context is basically a flag]

        Returns:
            [Mostly tensors]: [action_tuple: Tuple of actions
            valid_prob : Valid Probabilities
            policy_state :  State of Policy 
               action_choosen_mat : Matrix for the action chosen
               curr_neighbor_mask_policy) : Neighbor masking policy
               next_state_ids : Propagated states
        """
        value_output = critic(np.array(critic_state))
        value_output = value_output.flatten()
        
        action_tuple = []
        valid_prob = []

        # For training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []
        #TODO put things in perspective here
        
        grid_ids = [x for x in range(self.number_of_master_nodes)]
        self.valid_action_mask = np.zeros((self.number_of_master_nodes, self.action_dim))
        for j in ava_node:
            if len(self.valid_action_mask[self.number_of_master_nodes-1]) ==j:

                self.valid_action_mask[self.number_of_master_nodes-1][j] = 1
            else:
                self.valid_action_mask[self.number_of_master_nodes-1][j] = 1
        curr_neighbor_mask = deepcopy(self.valid_action_mask)

        self.valid_neighbor_node_id = [[i for i in range(self.action_dim)] for j in range(self.number_of_master_nodes)]

        # compute policy probability.
        self.pm_out =self.pm(state)
        action_probs,_,_ = sm_prob(self.pm_out, curr_neighbor_mask) 
        curr_neighbor_mask_policy = []
        print('len(grid_ids) : ', len(grid_ids))
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]
            # action probability for state value function

            action_prob = action_prob.detach().numpy()
            valid_prob.append(action_prob)
            
            # To prevent breaking of the code given the context is set to 0
            if int(context[idx]) == 0:
                continue
            
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob / np.sum(action_prob))
            
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1
            print('curr_action_indices : ', curr_action_indices)
            print('action_dim : ', self.action_dim)
            self.valid_neighbor_grid_id = self.valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)
                    print
                    # for training
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(state)
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids
    
    def compute_advantage(self, curr_state_value:list, next_state_ids:list, next_state:np.array, critic: Type[Value_Model], node_reward: np.array, gamma:float)->list:
        """[Calculates difference between predicted Q value and ! Value target]

        Args:
            curr_state_value ([list]): [Q value for current state]
            next_state_ids ([list]): [Next state ids]
            next_state ([Numpy array]): [Next State Grid]
            critic : Global Critic Network
            node_reward ([Numpy Array]): [Node Reward grid]
            gamma ([float]): [Gamma variable for Bellman's equations]

        Returns:
            [list]: [list containing advantages]
        """
        # compute advantage

        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = critic(next_state).flatten()

        for idx, _ in enumerate(next_state_ids):
            temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[idx]
            advantage.append(temp_adv.detach().numpy())

        return advantage
    
    def compute_targets(self, valid_prob: np.array, next_state: np.array, critic: Type[Value_Model], node_reward: np.array, curr_neighbor_mask: np.array, gamma: float)->np.array:
        """[Method for computation of Targets]

        Args:
            valid_prob ([Numpy array]): [Valid probablility]
            next_state ([Numpy array]): [next state matrix]
            critic: Global Critic Network
            node_reward ([Numpy array]): [Reward for the node]
            curr_neighbor_mask : Current mask of the consumption of Resources
            gamma ([float]): [gamma for computaion of bellman's equations]

        Returns:
            [Numpy Array]: [Numpy array containing targets]
        """
        # compute targets
        
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = critic(next_state).flatten()

        for idx in np.arange(len(valid_prob)):
            grid_prob = valid_prob[idx][curr_neighbor_mask[idx] > 0]
            
            curr_grid_target = np.sum(
                grid_prob * (sum(node_reward) + gamma * sum(qvalue_next.detach().numpy())))
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

def to_grid_rewards(node_reward:list)->np.array:
    
    """[Serialises the given node rewards]
    Args:
            node_reward ([list]): [noder rewards in terms of list]
    Returns:
        [list]: [serialised numpy array]
    """
    
    return np.array(node_reward).reshape([-1, 1])



def calculate_reward(master_list: list, cur_done:list, cur_undone:list)-> float:
    """
    Tailored MARDL for Decentralised request dispatch - Reward : Improve the longterm throughput while ensuring the load balancing at the edge
    
    [Function that returns rewards from environment given master nodes and the current tasks]

    Args:
        master_list ([Master Object list]): [Edge Access Point list containing nodes]
        cur_done ([list]): [list containing two elements for tasks done on both master nodes]
        cur_undone ([list]): [list containing two elements for tasks not done yet on both master nodes]

    Returns:
        reward [list]: [list of rewards for both master nodes]
    """
    weight = 1.0
    all_task = []
    fail_task = []
    for i in range(len(master_list)):
        all_task.append(float(cur_done[i] + cur_undone[i]))
        fail_task.append(float(cur_undone[i]))
 
    reward = []
    # The ratio of requests that violate delay requirements
    task_fail_rate = []
    
    for i in range(len(master_list)):
        if all_task[i] != 0:
            task_fail_rate.append(fail_task[i] / all_task[i])
        else:
            task_fail_rate.append(0)
    # The standard deviation of the CPU and memory usage
    
    use_rate_dict = {}
    for i in range(len(master_list)):
        use_rate_dict[i] = []
    
    for i, mstr in enumerate(master_list):
        for j in range(len(mstr.node_list)):
            use_rate_dict[i].append(mstr.node_list[j].cpu / mstr.node_list[j].cpu_max)
            use_rate_dict[i].append(mstr.node_list[j].mem / mstr.node_list[j].mem_max)

    standard_list_dict = {}
    for i in range(len(master_list)):
        standard_list_dict[i] = np.std(use_rate_dict[i], ddof=1)
    reward_dict = {}
    for i in range(len(master_list)):
        reward_dict[i] = math.exp(-task_fail_rate[i]) + weight * math.exp(-standard_list_dict[i])

    for r in range(len(master_list)):
        reward.append(reward_dict[r])
    # Immediate reward   e^(-lambda - weight_of_load_balancing *standard_deviation_of_cpu_memory)
    return reward