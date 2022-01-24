import math
import numpy as np
import random
from env.platform import *
from env.env_run import update_docker
#from algorithm_torch.GPG import act_offload_agent
import sys
############ Set up according to your own needs  ###########
# The parameters are set to support the operation of the program, and may not be consistent with the actual system

vaild_node = 6  # Number of edge nodes available
SLOT_TIME = 0.5  # Time of one slot
MAX_TESK_TYPE = 12  # Number of tesk types
POD_CPU = 15.0  # CPU resources required for a POD
POD_MEM = 1.0  # Memory resources required for a POD
# Resource demand coefficients for different types of services
service_coefficient = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4]
# Parameters related to DRL
gamma = 0.9 # Discounting Coefficient
learning_rate = 1e-3 # Learning Rate
#action_dim = 7 #Number of actions possibly edge nodes 6 plus Cluster
state_dim = 54#90#88 #Dimension of state for cMMAC (flattened Deployed state, task num, cpu_list, min_list)

node_input_dim = 24 # Input dimension of Node part of the Orchestration Net 
scale_input_dim = 24 # Input dimension for scale part of the Orchestration Net
high_value_edge_nodes = 2
# notes on input dimensions
# node_inputs[i, :12] = curr_tasks_in_queue[i, :12]
# node_inputs[i, 12:] = deploy_state[i, :12]
# cluster_inputs[0, :12] = done_tasks[:12]
# cluster_inputs[0, 12:] = undone_tasks[:12] 

hid_dims = [16, 8] # hidden dimensions of the Graph Neural Networks
output_dim = 8 # Output dimension of Graph Neural Networks
max_depth = 8 # The Depth for aggregation of Graph Neural Networks
entropy_weight_init = 1 # Initial Entropy weight for weight scaling of orchestration loss
exec_cap = 24 # Execution capacity (#TODO Not so sure yet)
entropy_weight_min = 0.0001 # Minimum allowed entropy weight
entropy_weight_decay = 1e-3 # Entropy weight decay rate
number_of_master_nodes  = 2 # number of eAPs # Only in this case whole thing makes sense

#num_edge_nodes_per_eAP =3
cluster_action_value = 6
latency = 0

def initial_state_values():
    deploy_state = []
    #print(np.random.randint(low=0, high=2, size=MAX_TESK_TYPE).tolist())
    '''
    for i in range(cluster_action_value):
        sub_list = []
        for j in range(MAX_TESK_TYPE):
            
            sub_list.append(0)
            #
            #if random.random()< 0.25:
            #    sub_list.append(1)
            #else:
            #    sub_list.append(0)
            #

        deploy_state.append(sub_list)
                    
        #deploy_state.append(np.random.randint(low=0, high=2, size=MAX_TESK_TYPE).tolist())
    #print(deploy_state)
    #a=b
    '''
    deploy_state = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]]
    
    node_list_1 = [[100.0, 4.0], [200.0, 6.0], [100.0, 8.0]]
    node_list_2 = [[200.0, 8.0], [100.0, 2.0], [200.0, 6.0]]

    node_param_lists = [node_list_1, node_list_2]

    master_param_lists = [[200.0, 8.0], [200.0, 8.0]]
    return deploy_state, node_param_lists, master_param_lists

def get_action_dim(node_param_lists):
    action_dim = 0
    for _list in node_param_lists:
        for node_param in _list:
            action_dim+=1
    return action_dim + 1 # +1 because of cluster

def set_lr( optimizer, lr):    
    """Method to set the Learning rate of the given optimizer

    Args:
        optimizer ([Pytorch optimizer]): [Optimizer]
        lr ([Float]): [Learning Rate]
    """
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr    

def get_state_list(master_list, max_tasks):
    state_list = []
    for mast in master_list:
        state_list.append(state_inside_eAP(mast, len(mast.node_list), max_tasks))
    return state_list 
def estimate_state_size(all_task_list, max_tasks):

    deploy_state, node_param_lists, master_param_lists = initial_state_values()
    
    master_list = create_master_list(node_param_lists, master_param_lists, all_task_list)
    '''
    state_list = []
    for mast in master_list:
        state_list.append(state_inside_eAP(mast, len(mast.node_list)))
    '''                
    last_length, length_list = get_last_length(master_list)
    state_list = get_state_list(master_list, max_tasks)
    s_grid_len = []
    for i, state in enumerate((state_list)):
        sub_deploy_state = deploy_state[length_list[i]:length_list[i+1]]
        #print('sub_deploy_state : ', sub_deploy_state)
        sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]],[[state[4]]], [[state[3]]], [state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
        s_grid_len.append(len(sub_elem))
        
    return s_grid_len

def flatten(list):
    """Function to serialize sublists inside a given list

    Args:
        list (list): [ A list of lists]

    Returns:
        [list]: [Serialized list ]
    """    
    return [y for x in list for y in x]
 

def remove_docker_from_master_node(master, change_node_idx, service_index, deploy_state):
    """Function to remove appropriate docker container from a given EAP (master node)

    Args:
        master ([Master Object]): [eAP Object]
        change_node_idx ([int]): [index of Edge Node]
        service_index ([int]): [Service indexes needed to be changed]
        deploy_state ([list]): [list of lists representing the deployed serices at each edge point]
    """
    docker_idx = 0
    while docker_idx < len(master.node_list[change_node_idx].service_list):
        if docker_idx >= len(master.node_list[change_node_idx].service_list):
            break
        if master.node_list[change_node_idx].service_list[docker_idx].kind == service_index:
            master.node_list[change_node_idx].mem = master.node_list[change_node_idx].mem + \
                                                    master.node_list[
                                                        change_node_idx].service_list[
                                                        docker_idx].mem
            del master.node_list[change_node_idx].service_list[docker_idx]
            deploy_state[change_node_idx][service_index] = deploy_state[change_node_idx][
                                                                service_index] - 1.0
        else:
            docker_idx = docker_idx + 1


def deploy_new_docker(master, POD_MEM, POD_CPU, cur_time, change_node_idx, service_coefficient, service_index, deploy_state):
    """Function to deploy new docker containers

    Args:
        master ([Master Object]): [eAP Object]
        POD_MEM ([type]): [Memory Resources required for a POD]
        POD_CPU ([type]): [Computation Resources required ]
        cur_time ([time]): [Current time]
        change_node_idx ([int]):[index of Edge Node]
        service_coefficient ([type]): [description]
        service_index ([type]): [Service indexes needed to be changed]
        deploy_state ([list]): [list of lists representing the deployed serices at each edge point]
    """
    docker = Docker(POD_MEM * service_coefficient[service_index],
                                                POD_CPU * service_coefficient[service_index],
                                                cur_time, service_index, [-1])
    master.node_list[change_node_idx].mem = master.node_list[
                                                change_node_idx].mem - POD_MEM * \
                                            service_coefficient[service_index]
    master.node_list[change_node_idx].service_list.append(docker)
    deploy_state[change_node_idx][service_index] = deploy_state[change_node_idx][
                                                        service_index] + 1    
    
    
def get_current_task(master):
    """Get Current task from the task queue

    Args:
        master ([Master Object]): [eAP object]

    Returns:
        [list]: [list containing the current task]
    """
    task = [-1]
    if len(master.task_queue) != 0:
        task = master.task_queue[0]
        del master.task_queue[0]
    
    return task
            
def state_inside_eAP(master, num_edge_nodes_per_eAP, max_tasks):
    """Get the state inside the given the edge nodes per eAP

    Args:
        master ([Master Object]): [eAP object]
        num_edge_nodes_per_eAP ([int]): [Number of edge nodes in eAP]

    Returns:
        [list]: [CPU , memory and tasks in queue]
    """
    cpu_list = []
    mem_list  = []
    task_num  = [len(master.task_queue)]
    if len(master.task_queue)==0:
        service_type = max_tasks + 1
        delay_requirement = 100000
    else:
        service_type = master.task_queue[0][0]
        
        delay_requirement = master.task_queue[0][2]-master.task_queue[0][1]
        
    undone_tasks = master.undone
    for i in range(num_edge_nodes_per_eAP):
        cpu_list.append([master.node_list[i].cpu, master.node_list[i].cpu_max])
        mem_list.append([master.node_list[i].mem, master.node_list[i].mem_max])
        task_num.append(len(master.node_list[i].task_queue))
        #print('master.node_list[i]. : ', master.undone)
    #a=b
    #print('service_type : ', service_type)
    return cpu_list, mem_list, task_num, undone_tasks, service_type, delay_requirement

def create_node_list(node_specification):
    """Creates a node list

    Args:
        node_specification ([list]): [list containing the specifications of node]

    Returns:
        node_list(List): [list of created nodes]
    """
    node_list = []
    for node_spec in node_specification:
        node_list.append(Node(node_spec[0], node_spec[1], [], []))
    return node_list
def create_master_list(node_param_lists, master_param_lists, all_task_list):
    node_lists = []
    for node_params in node_param_lists:
        node_lists.append(create_node_list(node_params))
    
    # (cpu, mem,..., achieve task num, give up task num)
    master_list = []
    for i, master_params in enumerate(master_param_lists):
        master_list.append(Master(master_params[0], master_params[1], node_lists[i], [], all_task_list[i], 0, 0, 0, [0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE))
    return master_list

def create_eAP_and_Cloud(node_param_lists, master_param_lists, all_task_list, MAX_TESK_TYPE, POD_MEM,  POD_CPU, service_coefficient, cur_time):
    """Create Edge Access Points and Cloud

    Args:
        node_param_lists ([list]): [list of node specifications]
        master_param_lists ([list]): [list of master specifications]
        all_task_list (list): [list of all tasks data]
        MAX_TESK_TYPE (int) : Maximum number of task types
        POD_MEM: (float): Memory of POD
        POD_CPU (float): CPU of POD
        service_coefficient(list): Service Coefficients
        cur_time (float) : Time
        
    Returns:
        master_list (list) :  List of created Master Nodes
        cloud (Cloud Object) : Created cloud object
    """
    master_list = create_master_list(node_param_lists, master_param_lists, all_task_list)
    cloud = Cloud([], [], sys.maxsize, sys.maxsize)  # (..., cpu, mem)
    ################################################################################################
    for i in range(MAX_TESK_TYPE):
        docker = Docker(POD_MEM * service_coefficient[i], POD_CPU * service_coefficient[i], cur_time, i, [-1])
        cloud.service_list.append(docker)
    
    return master_list, cloud

def get_last_length(master_list):
    length_list = [0]
    last_length = length_list[0]
    for mstr in master_list:
        length_list.append(last_length + len(mstr.node_list))
        last_length = last_length + len(mstr.node_list)
    return last_length, length_list
        
def put_current_task_on_queue(act, curr_task, cluster_action_value, cloud, master_list): 
    """Put current tasks on queue

    Args:
        act ([list]): actions
        curr_task ([list]): [list of tasks]
        cluster_action_value (int): Action value
        cloud (Cloud Object) : Created cloud object
        master_list (list) :  List of created Master Nodes

    """
    last_length, length_list = get_last_length(master_list)    
    for i in range(len(act)):
        if curr_task[i][0] == -1:
            continue
        if act[i] == cluster_action_value:
            cloud.task_queue.append(curr_task[i])
            continue
        
        for j in range(len(length_list)-1):
            if act[i] >= length_list[j] and act[i] < length_list[j+1] :
                master_list[j].node_list[act[i] - length_list[j]].task_queue.append(curr_task[i])
                
            
def update_state_of_task( cur_time, check_queue, cloud, master_list):
    """Update the state of tasks

    Args:
        cur_time (float) : Time
        check_queue ([function]): function to calculate queue information (task_queue, undone, undone_kind)
        cloud (Cloud Object) : Cloud object
        master_list (list) :  List of created Master Nodes

    """
    
    for mstr in master_list:
        for i in range(len(mstr.node_list)):
            mstr.node_list[i].task_queue, undone, undone_kind = check_queue(mstr.node_list[i].task_queue, cur_time, len(master_list))
            for j in undone_kind:
                mstr.undone_kind[j] = mstr.undone_kind[j] + 1
            
            #TODO I do not understand why the undone list is used to update every other list
            
            for i, master_entity in enumerate(master_list):
                master_entity.update_undone(undone[i])

    cloud.task_queue, undone, undone_kind = check_queue(cloud.task_queue, cur_time, len(master_list))
    for i, master_entity in enumerate(master_list):
        master_entity.update_undone(undone[i])
    
def update_state_of_dockers(cur_time, cloud, master_list):
    """Updates the state of dockers

    Args:
        cur_time (float) : Time
        cloud (Cloud Object) : Cloud object
        master_list (list) :  List of created Master Nodes

    """
    for mstr in master_list:
        for i in range(len(mstr.node_list)):
            mstr.node_list[i], undone, done, done_kind, undone_kind = update_docker(mstr.node_list[i], cur_time, service_coefficient, POD_CPU)
            for j in range(len(done_kind)):
                mstr.done_kind[done_kind[j]] = mstr.done_kind[done_kind[j]] + 1
            for j in range(len(undone_kind)):
                mstr.undone_kind[undone_kind[j]] = mstr.undone_kind[undone_kind[j]] + 1
            
            for i, master_entity in enumerate(master_list):
                master_entity.update_undone(undone[i])
                master_entity.update_done(done[i])
    cloud, undone, done, done_kind, undone_kind = update_docker(cloud, cur_time, service_coefficient, POD_CPU)

    for i, master_entity in enumerate(master_list):
                master_entity.update_undone(undone[i])
                master_entity.update_done(done[i])

    return cloud  

def create_dockers(vaild_node, MAX_TESK_TYPE, deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list):
    """Creation of dockers

    Args:
        vaild_node (int) : Number of valid nodes for execution of tasks
        MAX_TESK_TYPE (int) : Maximum number of task types
        deploy_state(list of lists): List containing the tasks running on all of the Nodes  
        service_coefficient(list): Service Coefficients
        POD_MEM: (float): Memory of POD
        POD_CPU (float): CPU of POD
        cur_time (float) : Time
        master_list (list) :  List of created Master Nodes

    """
    
    length_list = [0]
    last_length = length_list[0]
    for mstr in master_list:
        length_list.append(last_length + len(mstr.node_list))
        last_length = last_length + len(mstr.node_list)
    
    for i in range(vaild_node):
            for ii in range(MAX_TESK_TYPE):
                dicision = deploy_state[i][ii]
                for j in range(len(length_list)-1):
                    if i>= length_list[j] and i < length_list[j+1] and dicision == 1:
                        k= i-length_list[j]
                        if master_list[j].node_list[k].mem >= POD_MEM * service_coefficient[ii]:
                            docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                            ii, [-1])
                            master_list[j].add_to_node_attribute(k, 'mem', - POD_MEM * service_coefficient[ii])
                            master_list[j].append_docker_to_node_service_list(k, docker)
                            
                        
                        
def get_state_characteristics(MAX_TESK_TYPE, master_list):
    """Get lists of tasks that are done, undone and current task in queue

    Args:
        MAX_TESK_TYPE ([int]): Maximum types of tasks
        master_list ([Master Object list]): [Edge Access Point list containing nodes]
        num_edge_nodes_per_eAP ([type]): [Number of edge nodes under an edge access point]

    Returns:
        [lists]: [lists of number of done, undone and current tasks in queue]
    """
    done_tasks = []
    undone_tasks = []
    curr_tasks_in_queue = []
    
    # Get task state, include successful, failed, and unresolved
    for i in range(MAX_TESK_TYPE):
        done_val = 0.0
        undone_val = 0.0
        for master in master_list:
            done_val += master.done_kind[i]
            undone_val += master.undone_kind[i]
        done_tasks.append(done_val)
        undone_tasks.append(undone_val)     
        
    for master in master_list:
        for i in range(len(master.node_list)):
            tmp = [0.0] * MAX_TESK_TYPE
            for j in range(len(master.node_list[i].task_queue)):
                tmp[master.node_list[i].task_queue[j][0]] = tmp[master.node_list[i].task_queue[j][0]] + 1.0
            curr_tasks_in_queue.append(tmp)
    return done_tasks, undone_tasks, curr_tasks_in_queue


