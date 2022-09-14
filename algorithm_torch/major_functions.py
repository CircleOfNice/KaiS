from algorithm_torch.helpers_main_pytorch import *
from algorithm_torch.GPG import get_gpg_reward
from env.env_run import get_all_task, update_task_queue
from algorithm_torch.cMMAC import Estimator, to_grid_rewards
from algorithm_torch.ReplayMemory import ReplayMemory
from algorithm_torch.policyReplayMemory import policyReplayMemory
from algorithm_torch.CMMAC_Value_Model import build_value_model, update_value, Value_Model
from algorithm_torch.CMMAC_Policy_Model import update_policy
from algorithm_torch.gcn import GraphCNN
import numpy as np
from torch import optim
import random
import pickle,gzip
from typing import Callable, Union, Tuple, Type

def get_all_task_lists(csv_paths:list, randomize_data:bool)->Tuple:
    '''
    Generate talk list for the csv files
    Args: 
        csv_paths : paths of available csv files 
        randomize_data: Flag for randomisation of data
    Returns:
        all_task_list_init: list of tasks data
        max_task_pool_init: list of maximum tasks in tasks

    '''
    all_task_list_init = []
    max_task_pool_init = []

    for csv_path in csv_paths:
        all_task, max_task = get_all_task(csv_path, randomize=randomize_data)
        all_task_list_init.append(all_task)
        max_task_pool_init.append(max_task)
    return all_task_list_init, max_task_pool_init

def generate_task_lists_for_eaps(total_eaps:int, max_task_pool_init:list, all_task_list_init:list)->list:
    '''
    Generate talk list for the given number of eAP
    Args: 
        total_eaps : Total number of eAPs
        max_task_pool_init: Max tasks in tasks list
        all_task_list_init: All tasks list for task
    Returns:
        all_task_list: list of tasks data
    '''
    all_task_list = []
    for i in range(total_eaps):
        
        val_int = random.randint(0,len(max_task_pool_init) -1)
        all_task_list.append(all_task_list_init[val_int])
    return all_task_list

def generate_edge_list(randomize:bool, nodes_in_cluster:int, all_task_list:list, low_bound_edge_mode:int, upper_bound_edge_mode:int)->list:
    '''
    Generate talk list for the given number of eAP
    Args: 
        randomize : randomisation flag for number of nodes under eAP
        nodes_in_cluster: Avergae number of nodes under eAP
        all_task_list: All tasks list 
        low_bound_edge_mode : Lower bound for number of nodes in eAps
        upper_bound_edge_mode : Upper bound for number of nodes in eAps 
    Returns:
        edge_list: list of edge nodes generate
    '''
    
    if randomize ==False:
        # For uniform Edge Nodes per eAP
        edge_list = [nodes_in_cluster]*len(all_task_list)
    else:
        # For random Edge Nodes per eAP
        edge_list = [random.sample(range(low_bound_edge_mode, upper_bound_edge_mode), 1)[0] for i in range(len(all_task_list))]
    return edge_list

def initialize_eap_params(csv_paths: str, total_eaps : int, nodes_in_cluster : int, low_bound_edge_mode: int, 
                          upper_bound_edge_mode : int, randomize_data: bool, randomize:bool)->Tuple[int, list, list, list, list, list]:
    '''
    Function to initialise eAps and there parameters
    Args: 
        csv_paths : paths of available csv files 
        total_eaps: Total number of eAPs 
        nodes_in_cluster: number of nodes in cluster if no randomisation 
        low_bound_edge_mode : Lower bound for number of nodes in eAps
        upper_bound_edge_mode : Upper bound for number of nodes in eAps 
        
        randomize_data: Flag to randomize the incoming data of csv file paths
        randomize : Flag to randomize the nodes under eAPs
    Returns:
        max_tasks: Maximum number of tasks
        all_task_list: task lists generated 
        edge_list: Generated Edge list
        node_param_lists: Generate edge node parameter list
        master_param_lists: Generate master node parameter list
        action_dims : List of action_dims for actions 

    '''
    all_task_list_init, max_task_pool_init = get_all_task_lists(csv_paths, randomize_data)
        
    max_tasks = max(max_task_pool_init) 

    all_task_list = generate_task_lists_for_eaps(total_eaps, max_task_pool_init, all_task_list_init)

    edge_list = generate_edge_list(randomize, nodes_in_cluster, all_task_list, low_bound_edge_mode, upper_bound_edge_mode)

    _, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)

    action_dims = get_action_dims(node_param_lists)

    return  max_tasks, all_task_list,edge_list, node_param_lists, master_param_lists, action_dims

def initialize_cmmac_agents(MAX_TASK_TYPE:int, all_task_list:list,edge_list:list, master_param_lists:list, action_dims:list, randomize:bool)->Tuple[Type[Value_Model], Callable, list,  list, list]:
    '''
    Function to initialise eAp Agents 
    Args: 
        MAX_TASK_TYPE: Maximum number of tasks
        all_task_list: Taks list      
        edge_list:  Edge nodes list
        master_param_lists:  Master Parameters list
        action_dims:  List of Action dimensions
        randomize: Randomise number of edge nodes under eAP
    Returns:
        critic: Critic Network
        critic_optimizer: Critic Network's optimiser
        q_estimator_list: List of Actors
        ReplayMemory_list: List of Replay Memory for Actor
        policy_replay_list : List of Policy Replay Memory for Actors  
    '''
    # Definition of cMMAc Agent
    q_estimator_list = []
    ReplayMemory_list = []
    policy_replay_list = []
    s_grid_len = estimate_state_size(all_task_list, MAX_TASK_TYPE, edge_list)

    if randomize ==False:
        for i in range(len(master_param_lists)):
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    else:
        for i in range(len(master_param_lists)):
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    
    # Creation of global critic (currently without cloud info of unprocessed requests)
    critic, critic_optimizer = build_value_model(sum(s_grid_len)+ 1) # Length of task queue can be only one digit long
    
    return critic, critic_optimizer, q_estimator_list,  ReplayMemory_list, policy_replay_list

def get_done_undone_context(master_param_lists:list)->Tuple[list, list, bool]:
    '''
    Function to get context of eAP
    Args: 
        master_param_lists: list of parameters
    Returns:
        pre_done: Done tasks
        pre_undone: Taks not done 
        context: Flag for the context
    '''
    pre_done = [] # list to track tasks done previously
    pre_undone = [] # list to track tasks undone (not done) previously
    context = [] # Flag
    for i in range(len(master_param_lists)):
        pre_done.append(0)
        pre_undone.append(0)
        context.append(1)
    return pre_done, pre_undone, context

def initialize_episode_params(all_task_list:list, edge_list:list, MAX_TASK_TYPE:int, cur_time:float)->Tuple[list, Type[Cloud], list, list, list, list, bool]:
    '''
    Function to initialize episode parameters and objects
    Args: 
        all_task_list: list of parameters
        edge_list: list of parameters
        MAX_TASK_TYPE: Maximum Task type
        cur_time: list of parameters
    Returns:
        master_list: Generated List of Masters 
        cloud: Cloud generated
        graph_cnn_list: Graph CNNs Generated
        deploy_states: Current state of deployment of tasks
        pre_done: Done tasks
        pre_undone: Not Done tasks
        context : Flag generated for Context
    '''
    
    deploy_states, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
        
    # Create clusters based on the hardware resources you need
    master_list, cloud = create_eAP_and_Cloud(node_param_lists, master_param_lists, all_task_list, MAX_TASK_TYPE, POD_MEM,  POD_CPU, service_coefficient, cur_time)
    
    # Creation of node Graph CNN
    graph_cnn_list = []
    for master in master_list:
        cpu_list, mem_list, _ = get_node_characteristics(master) 
        graph_cnn_list.append(GraphCNN(len(cpu_list)+ len(mem_list) + len(mem_list), hid_dims, output_dim, max_depth, act_function))

    # Create dockers based on deploy_state
    create_dockers( MAX_TASK_TYPE, deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
    
    pre_done, pre_undone, context = get_done_undone_context(master_param_lists)
        
    return master_list, cloud, graph_cnn_list, deploy_states, pre_done, pre_undone, context

def get_all_node_characteristics(master_list:list)->Tuple[list, list, list]:
    '''
    Function to get all node characteristics
    Args: 
        master_list: List of Masters(eAPs)
    Returns:
        cpu_lists: Current CPU Consumption level list
        mem_lists: Current Memory Consumption level list
        task_lists: Task list in the queue
    '''
    cpu_lists =[]
    mem_lists = []
    task_lists = []
    for master in master_list:
        cpu_list, mem_list, task_list = get_node_characteristics(master)  
        cpu_lists.append(cpu_list)
        mem_lists.append(mem_list)
        task_lists.append(task_list)
    return cpu_lists, mem_lists, task_lists

def get_float_deploy_states(deploy_states:list)->list:
    '''
    Function to get deploy states converted into float
    Args: 
        deploy_states: Current statement of deployment of tasks over edge nodes
    Returns:
        deploy_states_float: Current statement of deployment of tasks over edge nodes in Floating points
    '''
    
    deploy_states_float = []
    for item, deploy_state in enumerate(deploy_states):

        elem_list = []
        for d_state in deploy_state:
            sub_elem_list = []
            for item in d_state:
                sub_elem_list.append(float(item))
            
            elem_list.append(sub_elem_list)

        deploy_states_float.append(elem_list) 
    return deploy_states_float
            
def get_task_state_deploy_state_and_exp(MAX_TASK_TYPE:int, master_list:list, exp:list, deploy_states:list, cur_time:float)->Tuple[list, list, list, list, list, list, list, list ]:
    '''
    Function to get state of tasks deplyment and updating the experience.
    Args: 
        MAX_TASK_TYPE: Maximumm number of tasks
        master_list: List of Masters(eAPs)
        exp: Experiene Dictiorary
        deploy_states: Current Deployed State 
        cur_time: Time at the moment
    Returns:
        done_tasks: No. of done tasks
        undone_tasks: No. of not done tasks
        curr_tasks_in_queue: No. of tasks in the queue
        deploy_states_float:  Deployment states in term of float values
        exp: Accumulated experiences
        cpu_lists: CPU List of consumption of edge devices
        mem_lists: Memory List of consumption of edge devices
        task_lists: list for tasks in the queue
    '''
    
    done_tasks, undone_tasks, curr_tasks_in_queue,  = get_state_characteristics(MAX_TASK_TYPE, master_list)  

    cpu_lists, mem_lists, task_lists =  get_all_node_characteristics(master_list)
    
    reward_val = float(get_gpg_reward(master_list))
    exp['reward'].append(reward_val)
    exp['wall_time'].append(cur_time)
        
    deploy_states_float = get_float_deploy_states(deploy_states)
        
    return   done_tasks, undone_tasks, curr_tasks_in_queue, deploy_states_float, exp, cpu_lists, mem_lists, task_lists 

def update_task_queue_master_list(master_list:list, cur_time:float)->list:
    '''
    Function to update task queue of master list
    Args: 
        master_list: List of Masters(eAPs)
        cur_time: Current time
    Returns:
        master_list: Updated List of Masters(eAPs)
    '''
    for i, master in enumerate(master_list):
        master_list[i] = update_task_queue(master, cur_time, i) 
    return master_list 

def update_current_task_master_list(master_list:list)->list:
    '''
    Function to update current tasks that are executing from master list
    Args: 
        master_list: List of Masters(eAPs)
    Returns:
        curr_task: List of current tasks that are executing from master list
    '''
    curr_task = []
    for master in master_list:
        curr_task.append(get_current_task(master))
    return curr_task

def get_ava_node(curr_task:list, action_dims:list, deploy_states:list, randomize:bool)->list:
    '''
    Function to get availabel nodes for execution of current tasks
    Args: 
        curr_task: Current incoming tasks
        action_dims: Total Action available
        deploy_states: State of deployment
        randomize: Radomisation of eAP flag
    Returns:
        ava_node: Available nodes for execution of tasks
    '''
    
    ava_node = []

    for i in range(len(curr_task)):
        #TODO Repeated chunk of code delete this
        if randomize ==False:
        
            tmp_list = [action_dims[i] -1]  # Cloud is always available
        else:
            
            tmp_list = [action_dims[i] -1]
        deploy_state = deploy_states[i]
        for ii in range(len(deploy_state)):

            if deploy_state[ii][curr_task[i][0]] == 1:
                tmp_list.append(ii)
        ava_node.append(tmp_list)
    return ava_node

def get_critic_state(master_list:list, state_list:list, deploy_states:list)->list:
    '''
    Function to get state for the global critic
    Args: 
        master_list: List of Masters(eAPs)
        state_list: List of states
        deploy_states: State of deployment
    Returns:
        s_grid: state for the global critic
    '''
    
    s_grid = []
    for i, state in enumerate((state_list)):
        sub_deploy_state = deploy_states[i]
        sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]], [[state[4]]], [[state[3]]],[state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
        s_grid.append(sub_elem)
        
    return  s_grid

def get_updated_tasks_ava_node_states(master_list:list, cloud, deploy_states:list, action_dims:list, cur_time: float, max_tasks:int, randomize:bool)->Tuple[list,list, list,list,list]:
    '''
    Function to update tasks, avalilable nodes and states
    Args: 
        master_list: List of Masters(eAPs)
        cloud: cloud
        deploy_states: State of deployment
        action_dims : List of action_dims
        cur_time : current time
        max_tasks : Maximum number of tasks
        randomize : randomization flag for eAPs
    Returns:
        master_list: updated List of Masters(eAPs)
        curr_task: updated List of current tasks
        ava_node: Available nodes for execution
        s_grid: Critic state without cloud_info
        critic_state : complete critic state
    '''    
    master_list = update_task_queue_master_list(master_list, cur_time)    
    
    curr_task = update_current_task_master_list(master_list)

    ava_node = get_ava_node(curr_task, action_dims, deploy_states, randomize)
    state_list = get_state_list(master_list, max_tasks)    
    
    s_grid = get_critic_state(master_list, state_list, deploy_states)
    critic_state = flatten(s_grid)
    critic_state.append(len(cloud.task_queue))
    
    return master_list, curr_task, ava_node, s_grid, critic_state

def get_estimators_output(q_estimator_list:list, s_grid:list, critic: Type[Value_Model], critic_state:list, ava_node:list, context:bool)->Tuple[list,list,list,list,list,list,list]:
    '''
    Function to get state for the global critic
    Args: 
        q_estimator_list: list of actors
        s_grid: state of all eAPs
        critic: Critic Network
        critic_state: State for critic network  
        ava_node: Available nodes for execution
        context: Context
    Returns:
        act: Tuple of actions
        valid_action_prob_mat: Valid Probabilities
        policy_state: State of Policy 
        action_choosen_mat: Matrix for the action chosen
        curr_neighbor_mask: Neighbor mask
        curr_state_value: Current value of state
        next_state_ids:Propagated states
    '''    
    act = []
    valid_action_prob_mat = []
    policy_state = []
    action_choosen_mat = []
    curr_state_value = []
    curr_neighbor_mask = []
    next_state_ids = []
    for i in range(len(s_grid)):
        
        act_, valid_action_prob_mat_, policy_state_, action_choosen_mat_, \
        curr_state_value_, curr_neighbor_mask_, next_state_ids_ = q_estimator_list[i].action(np.array(s_grid[i]), critic, critic_state, ava_node[i], context,)

        act.append(act_[0])
        valid_action_prob_mat.append(valid_action_prob_mat_[0])
        policy_state.append(policy_state_[0])
        action_choosen_mat.append(action_choosen_mat_[0])
        curr_state_value.append(curr_state_value_[0])
        curr_neighbor_mask.append(curr_neighbor_mask_[0])
        next_state_ids.append(next_state_ids_[0])
    valid_action_prob_mat = np.array(valid_action_prob_mat)
    policy_state = np.array(policy_state)
    action_choosen_mat = np.array(action_choosen_mat)
    curr_neighbor_mask = np.array(curr_neighbor_mask)
    
    return act, valid_action_prob_mat, policy_state, action_choosen_mat, curr_neighbor_mask, curr_state_value, next_state_ids


def get_done_status(master_list:list, pre_done:list, pre_undone:list)->Tuple[list,list, list, list]:
    '''
    Function to update the status
    Args: 
        master_list: List of Masters(eAPs)
        pre_done: List of previously done tasks
        pre_undone: List of previously not done tasks
    Returns:
        pre_done : Updated done tasks
        pre_undone : Updated not done tasks
        cur_done : status of current tasks that are done
        cur_undone : status of current tasks that are not done
    '''
    cur_done = []
    cur_undone = []
    ch_pre_done = []
    ch_pre_undone = []
    for i, mstr in enumerate(master_list):
        cur_done.append(mstr.done - pre_done[i])
        cur_undone.append(mstr.undone - pre_undone[i])
        
        ch_pre_done.append(mstr.done)
        ch_pre_undone.append(mstr.undone)

    pre_done = ch_pre_done
    pre_undone = ch_pre_undone
    
    return pre_done, pre_undone, cur_done, cur_undone

def put_and_update_tasks(act:list, curr_task:list, action_dims:list, cloud:Type[Cloud], master_list:list,check_queue:Callable, cur_time:float, pre_done:list, pre_undone:list)-> Tuple[list,list, list, list, Type[Cloud]]:
    '''
    Function to update the status of tasks
    Args: 
        act:Tuple of actions
        curr_task: current tasks 
        action_dims: Action Dimensions list
        cloud: cloud
        master_list: master list
        check_queue: function to calculate queue information (task_queue, undone, undone_kind)
        cur_time: Current time
        pre_done: previousl done tasks
        pre_undone: Previously not done tasks
    Returns:
        pre_done : Updated done tasks
        pre_undone : Updated not done tasks
        cur_done : status of current tasks that are done
        cur_undone : status of current tasks that are not done
        cloud :  Cloud
    '''
    # Put the current task on the queue based on dispatch decision
    put_current_task_on_queue(act, curr_task, action_dims, cloud, master_list)
    # Update state of task
    update_state_of_task(cur_time, check_queue, cloud, master_list)
    
    # Update state of dockers in every node
    cloud = update_state_of_dockers(cur_time, cloud, master_list)
        
    pre_done, pre_undone, cur_done, cur_undone = get_done_status(master_list, pre_done, pre_undone)
    return pre_done, pre_undone, cur_done, cur_undone, cloud 

# TODO :Comment for remaining functions and add type hints


def update_exp_replays(immediate_reward:np.array, q_estimator_list:list, ReplayMemory_list:list, policy_replay_list:list, action_mat_prev:np.array, critic_state:np.array, critic:Type[Value_Model], s_grid:list, 
                       curr_task:list, state_mat_prev:list, curr_neighbor_mask_prev:np.array, curr_state_value_prev:list, next_state_ids_prev:list, policy_state_prev:np.array, action_choosen_mat_prev:np.array)->Tuple:
    '''
    Update experience replay
    Args: 
        immediate_reward: Immediate reward 
        q_estimator_list: List of CmmAC agent
        ReplayMemory_list: Replay Memmory list
        policy_replay_list: Replay memory List for Policy model
        action_mat_prev: Action Probablities
        critic_state: state for Critic
        critic: Critic Network
        s_grid: State Grid issue
        curr_task: Current tasks which are executing
        state_mat_prev: Previous Current state 
        curr_neighbor_mask_prev: last Neighbor mask
        curr_state_value_prev: Previous value of current state
        next_state_ids_prev: Valid next states
        policy_state_prev: State for Policy network
        action_choosen_mat_prev: Matrix for the action chosen prevously
    Returns:
        ReplayMemory_list: Replay Memmory list
        policy_replay_list: Replay memory List for Policy model
    '''
    r_grid = to_grid_rewards(immediate_reward)
    for m in range(len(r_grid)):

        targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m]], np.array(critic_state), critic, r_grid[[m]], curr_neighbor_mask_prev[m], gamma)
        # Advantage for policy network.
        advantage = q_estimator_list[m].compute_advantage([curr_state_value_prev[m]], [next_state_ids_prev[m]] ,
                                                np.array(critic_state), critic, r_grid[[m],:], gamma)

        test_cond_list = []
        for i, elem in enumerate(curr_task):
            test_cond_list.append(elem[0] != -1)
        
        cond = test_cond_list[0]
        
        if len(test_cond_list)>1:
            for i in range(1,len(test_cond_list)):
                cond = cond and test_cond_list[i]
        if cond:
            ReplayMemory_list[m].add(np.array([state_mat_prev]), action_mat_prev[[m]], targets_batch[[0]], np.array([s_grid[m]]))
            policy_replay_list[m].add(policy_state_prev[[m]], action_choosen_mat_prev[[m]], advantage , curr_neighbor_mask_prev[[m]])  
    return ReplayMemory_list, policy_replay_list

def train_critic(TRAIN_TIMES:int, master_list:list, ReplayMemory_list:list, critic:Type[Value_Model], critic_optimizer:torch.optim, log_estimator_value_loss:list)->list:
    '''
    Train Critic Network
    Args: 
        TRAIN_TIMES: Number of times to train
        master_list: master (Eap) List
        ReplayMemory_list:Replay Memory list
        critic: Critic
        critic_optimizer: Critic optimizer
        log_estimator_value_loss: List for logging value Loss
    Returns:
        log_estimator_value_loss: List for logging value Loss
    '''
    for _ in np.arange(TRAIN_TIMES):
        for m in range(len(master_list)):
            batch_s, _, batch_r, _ = ReplayMemory_list[m].sample()
            value_loss = update_value(batch_s, batch_r, 1e-3, critic, critic_optimizer)
            log_estimator_value_loss.append(value_loss.item())
    return log_estimator_value_loss
    
        
def train_actors(TRAIN_TIMES:int, master_list:list, policy_replay_list:list, q_estimator_list:list, log_estimator_policy_loss:list)->list:
    '''
    Train Actors Network
    Args: 
        TRAIN_TIMES: Number of times to train
        master_list: master (Eap) List
        policy_replay_list:Replay Memory list for policy network
        critic: Critic
        critic_optimizer: Critic Optimizer
        log_estimator_policy_loss: List for logging policy Loss
    Returns:
        log_estimator_policy_loss: List for logging policy Loss
    '''
    for _ in np.arange(TRAIN_TIMES):

        for m in range(len(master_list)):
            batch_s, batch_a, batch_r, batch_mask = policy_replay_list[m].sample()
            policy_loss = update_policy(q_estimator_list[m], batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask )
            log_estimator_policy_loss[m].append(policy_loss.item())
    return log_estimator_policy_loss

        
def train_actor_critic_without_orchestration(ReplayMemory_list:list, policy_replay_list:list, master_list:list, q_estimator_list:list, critic: Type[Value_Model], critic_optimizer:torch.optim, log_estimator_value_loss:list, 
                                             log_estimator_policy_loss:list, TRAIN_TIMES:int,)->Tuple[list, list]:
    '''
    Train Actors and Critic Network
    Args: 
        ReplayMemory_list:Replay Memory list
        policy_replay_list : Replay Memory list for policy network
        master_list: master (Eap) List
        q_estimator_list : List of Estimators (cMMAC Agents)
        critic: Critic
        critic_optimizer: Critic Optimizer
        log_estimator_value_loss:  List for logging value Loss
        log_estimator_policy_loss: List for logging policy Loss
        TRAIN_TIMES: Number of times to train
        
    Returns:
        log_estimator_value_loss: List for logging value Loss
        log_estimator_policy_loss: List for logging policy Loss
    '''
    
    log_estimator_value_loss = train_critic(TRAIN_TIMES, master_list, ReplayMemory_list, critic, critic_optimizer, log_estimator_value_loss)
    log_estimator_policy_loss = train_actors(TRAIN_TIMES, master_list, policy_replay_list, q_estimator_list, log_estimator_policy_loss)
    
    return log_estimator_value_loss, log_estimator_policy_loss

        
def check_and_dump(name:str, time_str:str, record, throughput_list:list)-> bool:
    
    '''
    Train Actors and Critic Network
    Args: 
        name: Descriptive name for the experiment
        time_str: Current time stamp
        record: list containing lists of master list, currently done tasks, currently not done tasks, immediate reward recieved]
        throughput_list : List containing the throughput across episodes
        
    Returns:
        True: Flag that it is done dumping data
    '''
    
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(record, f)
                
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", 'rb') as fp:
        record = pickle.load(fp)

    with gzip.open("./result/throughput_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(throughput_list, f)
        
    with gzip.open("./result/throughput_" + name + time_str + ".obj", 'rb') as fp:
        throughput_list = pickle.load(fp)
    return True
    
        
        
def generate_plots(all_task_list:list, throughput_list:list, log_orchestration_loss:list, log_estimator_value_loss:list, log_estimator_policy_loss:list, randomize:bool, low_bound_edge_mode:int, upper_bound_edge_mode:int, nodes_in_cluster:int)->bool:
    '''
    Train Actors and Critic Network
    Args: 
        all_task_list: Task Lists used to simulate the experiment
        throughput_list: Throughtput rate list
        log_orchestration_loss: Orchestration Loss List
        log_estimator_value_loss: Value Loss List
        log_estimator_policy_loss: Policy Loss Lists
        randomize: Randomising eAPs 
        low_bound_edge_mode: Lower bound of node if randomize true
        upper_bound_edge_mode: Upper bound of node if randomize true
        nodes_in_cluster: Total number of nodes in cluster
        
    Returns:
        True: Flag that it is done dumping data
    '''
    if randomize==True:
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_low_bound_edge_mpde_'+ str(low_bound_edge_mode) + '_upper_bound_edge_mpde_'+ str(upper_bound_edge_mode) 
    else : 
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_nodes_in_cluster_'+ str(nodes_in_cluster)
    
    plot_list(throughput_list, title, "Number of Episodes", "Throughput rate")
    plot_list(log_orchestration_loss, title +'log_orchestration_loss', "Number of Episodes", "Orchestration loss")
    plot_list(log_estimator_value_loss, title + 'log_estimator_value_loss', "Number of Episodes", "Value loss")
    for i in range(len(log_estimator_policy_loss)):
        plot_list(log_estimator_policy_loss[i], title + 'log_estimator_policy_loss_' + str(i), "Number of Episodes", "log_estimator_policy_loss" + str(i))
    return True