from helpers_main_pytorch import *#def_initial_state_values, get_action_dims, estimate_state_size,POD_CPU, POD_MEM, service_coefficient
from algorithm_torch.GPG import get_gpg_reward
from env.env_run import get_all_task, update_task_queue
from algorithm_torch.cMMAC import Estimator, to_grid_rewards
from algorithm_torch.ReplayMemory import ReplayMemory
from algorithm_torch.policyReplayMemory import policyReplayMemory
from algorithm_torch.CMMAC_Value_Model import build_value_model, update_value
from algorithm_torch.CMMAC_Policy_Model import update_policy
from algorithm_torch.gcn import GraphCNN
import numpy as np
import random
import pickle,gzip

def get_all_task_lists(csv_paths, randomize_data):
    all_task_list_init = []
    max_task_pool_init = []

    for csv_path in csv_paths:
        all_task, max_task = get_all_task(csv_path, randomize=randomize_data)
        all_task_list_init.append(all_task)
        max_task_pool_init.append(max_task)
    return all_task_list_init, max_task_pool_init

def generate_task_lists_for_eaps(total_eaps, max_task_pool_init, all_task_list_init):
    all_task_list = []
    for i in range(total_eaps):
        
        val_int = random.randint(0,len(max_task_pool_init) -1)
        all_task_list.append(all_task_list_init[val_int])
    return all_task_list

def generate_edge_list(randomize, nodes_in_cluster, all_task_list, low_bound_edge_mode, upper_bound_edge_mode):
    if randomize ==False:
        # For uniform Edge Nodes per eAP
        edge_list = [nodes_in_cluster]*len(all_task_list)
    else:
        # For random Edge Nodes per eAP
        edge_list = [random.sample(range(low_bound_edge_mode, upper_bound_edge_mode), 1)[0] for i in range(len(all_task_list))]
    return edge_list

def initialize_eap_params(csv_paths, total_eaps, nodes_in_cluster,low_bound_edge_mode, upper_bound_edge_mode, randomize_data, randomize):

    all_task_list_init, max_task_pool_init = get_all_task_lists(csv_paths, randomize_data)
        
    max_tasks = max(max_task_pool_init) 

    all_task_list = generate_task_lists_for_eaps(total_eaps, max_task_pool_init, all_task_list_init)

    edge_list = generate_edge_list(randomize, nodes_in_cluster, all_task_list, low_bound_edge_mode, upper_bound_edge_mode)

    _, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)

    action_dims = get_action_dims(node_param_lists)

    return  max_tasks, all_task_list,edge_list, node_param_lists, master_param_lists, action_dims



def initialize_cmmac_agents(MAX_TASK_TYPE, all_task_list,edge_list, master_param_lists, action_dims, randomize):
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
    
    return critic, critic_optimizer, q_estimator_list, ReplayMemory_list, ReplayMemory_list, policy_replay_list

def get_done_undone_context(master_param_lists):
    pre_done = [] # list to track tasks done previously
    pre_undone = [] # list to track tasks undone (not done) previously
    context = [] # Flag
    for i in range(len(master_param_lists)):
        pre_done.append(0)
        pre_undone.append(0)
        context.append(1)
    return pre_done, pre_undone, context

def initialize_episode_params(all_task_list, edge_list, MAX_TASK_TYPE, cur_time):
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

def get_all_node_characteristics(master_list):
    cpu_lists =[]
    mem_lists = []
    task_lists = []
    for master in master_list:
        cpu_list, mem_list, task_list = get_node_characteristics(master)  
        cpu_lists.append(cpu_list)
        mem_lists.append(mem_list)
        task_lists.append(task_list)
    return cpu_lists, mem_lists, task_lists

def get_float_deploy_states(deploy_states):
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
            
def get_task_state_deploy_state_and_exp(MAX_TASK_TYPE, master_list, exp, deploy_states, cur_time):
    done_tasks, undone_tasks, curr_tasks_in_queue,  = get_state_characteristics(MAX_TASK_TYPE, master_list)  

    cpu_lists, mem_lists, task_lists =  get_all_node_characteristics(master_list)
    
    reward_val = float(get_gpg_reward(master_list))
    exp['reward'].append(reward_val)
    exp['wall_time'].append(cur_time)
        
    deploy_states_float = get_float_deploy_states(deploy_states)
        
    return   done_tasks, undone_tasks, curr_tasks_in_queue, deploy_states_float, exp, cpu_lists, mem_lists, task_lists 

def update_task_queue_master_list(master_list, cur_time):
    for i, master in enumerate(master_list):
        master_list[i] = update_task_queue(master, cur_time, i) 
    return master_list 

def update_current_task_master_list(master_list):
    curr_task = []
    for master in master_list:
        curr_task.append(get_current_task(master))
    return curr_task

def get_ava_node(curr_task, action_dims, deploy_states, randomize):
    ava_node = []

    for i in range(len(curr_task)):
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

def get_critic_state(master_list, state_list, deploy_states):
    s_grid = []
    for i, state in enumerate((state_list)):
        sub_deploy_state = deploy_states[i]
        sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]], [[state[4]]], [[state[3]]],[state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
        s_grid.append(sub_elem)
        
    return  s_grid

def get_updated_tasks_ava_node_states(master_list, cloud, deploy_states, action_dims, cur_time, max_tasks, randomize):
        
    master_list = update_task_queue_master_list(master_list, cur_time)    
    
    curr_task = update_current_task_master_list(master_list)

    ava_node = get_ava_node(curr_task, action_dims, deploy_states, randomize)
    state_list = get_state_list(master_list, max_tasks)    
    
    s_grid = get_critic_state(master_list, state_list, deploy_states)
    critic_state = flatten(s_grid)
    critic_state.append(len(cloud.task_queue))
    
    return master_list, curr_task, ava_node, s_grid, critic_state

def get_estimators_output(q_estimator_list, s_grid,critic, critic_state, ava_node, context):
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


def get_done_status(master_list, pre_done, pre_undone):
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

def put_and_update_tasks(act, curr_task, action_dims, cloud, master_list,check_queue, cur_time, pre_done, pre_undone):
    # Put the current task on the queue based on dispatch decision
    put_current_task_on_queue(act, curr_task, action_dims, cloud, master_list)
    # Update state of task
    update_state_of_task(cur_time, check_queue, cloud, master_list)
    
    # Update state of dockers in every node
    cloud = update_state_of_dockers(cur_time, cloud, master_list)
        
    pre_done, pre_undone, cur_done, cur_undone = get_done_status(master_list, pre_done, pre_undone)
    return pre_done, pre_undone, cur_done, cur_undone, cloud 


def update_exp_replays(immediate_reward, q_estimator_list, ReplayMemory_list, policy_replay_list, action_mat_prev, critic_state, critic, s_grid, 
                       curr_task, state_mat_prev, curr_neighbor_mask_prev, curr_state_value_prev, next_state_ids_prev, policy_state_prev, action_choosen_mat_prev):
    r_grid = to_grid_rewards(immediate_reward)
    for m in range(len(r_grid)):

        targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m]], np.array(critic_state), critic, r_grid[[m]], curr_neighbor_mask_prev[m], gamma)
        # Advantage for policy network.
        advantage = q_estimator_list[m].compute_advantage([curr_state_value_prev[m]], [next_state_ids_prev[m]] ,
                                                np.array(critic_state), critic, r_grid[[m],:], gamma)

        test_cond_list = []
        for i, elem in enumerate(curr_task):
            test_cond_list.append(elem[0] != -1)# != -1
        
        cond = test_cond_list[0]
        
        if len(test_cond_list)>1:
            for i in range(1,len(test_cond_list)):
                cond = cond and test_cond_list[i]
        if cond:
            ReplayMemory_list[m].add(np.array([state_mat_prev]), action_mat_prev[[m]], targets_batch[[0]], np.array([s_grid[m]]))
            policy_replay_list[m].add(policy_state_prev[[m]], action_choosen_mat_prev[[m]], advantage , curr_neighbor_mask_prev[[m]])  

def train_critic(TRAIN_TIMES, master_list, ReplayMemory_list, critic, critic_optimizer, log_estimator_value_loss, global_step1):
    for _ in np.arange(TRAIN_TIMES):
        for m in range(len(master_list)):
            batch_s, _, batch_r, _ = ReplayMemory_list[m].sample()
            value_loss = update_value(batch_s, batch_r, 1e-3, critic, critic_optimizer)
            log_estimator_value_loss.append(value_loss.item())
        global_step1 += 1
        
def train_actors(TRAIN_TIMES, master_list, policy_replay_list, q_estimator_list, log_estimator_policy_loss, global_step2):
    for _ in np.arange(TRAIN_TIMES):

        for m in range(len(master_list)):
            batch_s, batch_a, batch_r, batch_mask = policy_replay_list[m].sample()
            
            policy_loss = update_policy(q_estimator_list[m], batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,)
            log_estimator_policy_loss[m].append(policy_loss.item())

        global_step2 += 1
        
        
def train_actor_critic_without_orchestration(ReplayMemory_list, policy_replay_list, master_list, q_estimator_list, critic, critic_optimizer, log_estimator_value_loss, 
                                             log_estimator_policy_loss, TRAIN_TIMES, global_step1, global_step2):
    
    train_critic(TRAIN_TIMES, master_list, ReplayMemory_list, critic, critic_optimizer, log_estimator_value_loss, global_step1)
    train_actors(TRAIN_TIMES, master_list, policy_replay_list, q_estimator_list, log_estimator_policy_loss, global_step2)

        
def check_and_dump(name, time_str, record, throughput_list):
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(record, f)
                
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", 'rb') as fp:
        record = pickle.load(fp)

    with gzip.open("./result/throughput_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(throughput_list, f)
        
    with gzip.open("./result/throughput_" + name + time_str + ".obj", 'rb') as fp:
        throughput_list = pickle.load(fp)
        
        
def generate_plots(all_task_list, throughput_list, log_orchestration_loss, log_estimator_value_loss, log_estimator_policy_loss, randomize, low_bound_edge_mode, upper_bound_edge_mode, nodes_in_cluster):
    if randomize==True:
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_low_bound_edge_mpde_'+ str(low_bound_edge_mode) + '_upper_bound_edge_mpde_'+ str(upper_bound_edge_mode) 
    else : 
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_nodes_in_cluster_'+ str(nodes_in_cluster)
    
    plot_list(throughput_list, title, "Number of Episodes", "Throughput rate")
    plot_list(log_orchestration_loss, title +'log_orchestration_loss', "Number of Episodes", "Orchestration loss")
    plot_list(log_estimator_value_loss, title + 'log_estimator_value_loss', "Number of Episodes", "Value loss")
    for i in range(len(log_estimator_policy_loss)):
        plot_list(log_estimator_policy_loss[i], title + 'log_estimator_policy_loss_' + str(i), "Number of Episodes", "log_estimator_policy_loss" + str(i))