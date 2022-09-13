from ast import Call
from typing import Callable, Tuple, Type
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from env.platform import *
from env.env_run import update_docker
import random
#from algorithm_torch.GPG import act_offload_agent
import sys
############ Set up according to your own needs  ###########
# The parameters are set to support the operation of the program, and may not be consistent with the actual system

vaild_node = 6  # Number of edge nodes available
SLOT_TIME = 0.5  # Time of one slot
#MAX_TESK_TYPE = 12  # Number of tesk types
POD_CPU = 15.0  # CPU resources required for a POD
POD_MEM = 1.0  # Memory resources required for a POD
# Resource demand coefficients for different types of services
service_coefficient = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4]
# Parameters related to DRL
gamma = 0.9 # Discounting Coefficient
learning_rate = 1e-3 # Learning Rate

state_dim = 54#90#88 #Dimension of state for cMMAC (flattened Deployed state, task num, cpu_list, min_list)

#node_input_dim = 2*MAX_TESK_TYPE#24 # Input dimension of Node part of the Orchestration Net 
#scale_input_dim = 2*MAX_TESK_TYPE # Input dimension for scale part of the Orchestration Net
high_value_edge_nodes = 2

hid_dims = [16, 8] # hidden dimensions of the Graph Neural Networks
output_dim = 8 # Output dimension of Graph Neural Networks
max_depth = 8 # The Depth for aggregation of Graph Neural Networks
entropy_weight_init = 1 # Initial Entropy weight for weight scaling of orchestration loss
exec_cap = 24 # Execution capacity (#TODO Not so sure yet)
entropy_weight_min = 0.0001 # Minimum allowed entropy weight
entropy_weight_decay = 1e-3 # Entropy weight decay rate
number_of_master_nodes  = 2 # number of eAPs # Only in this case whole thing makes sense

latency = 0


act_function = nn.functional.leaky_relu
opt_function = torch.optim.Adam

def def_initial_state_values(len_all_task_list:int=3, list_length_edge_nodes_per_eap:list=[3, 3, 3])->Tuple:
    """Method to define initial values for the input cluster : 

    Args:
        len_all_task_list (int) : Number of Master Node data required  (Number of task lists available)
        list_length_edge_nodes_per_eap ([list]): Number of Edge Node data required 
        
    Returns: 
        deploy_states : Deploy State list Values
        node_param_lists :  Parameter list for Edge nodes required
        master_param_lists : Parameter list for Master nodes required
    """
    deploy_state_stack = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],]
    node_list_stack = [[100.0, 4.0], [200.0, 6.0], [100.0, 8.0], [200.0, 8.0], [100.0, 2.0], [200.0, 6.0]]
    master_param_list_stack = [[200.0, 8.0]]
    deploy_states = []
    master_param_lists =[]
    for item in list_length_edge_nodes_per_eap:
        deploy_state = []
        for i in range(item):
            deploy_state.append(random.choice(deploy_state_stack))
        deploy_states.append(deploy_state)
    
    
    for i in range(len_all_task_list):
        master_param_lists.append(random.choice(master_param_list_stack))
    
    node_param_lists = []    
    for i in range(len_all_task_list):

        node_list = []
        for k in range(list_length_edge_nodes_per_eap[i]):
            node_list.append(random.choice(node_list_stack))
        node_param_lists.append(node_list)
    #print('deploy_states: ', deploy_states)
    #print('node_param_lists : ', node_param_lists)
    #print('master_param_lists: ', master_param_lists)
    #a=B
    return deploy_states, node_param_lists, master_param_lists

def estimate_state_size(all_task_list:list, MAX_TASK_TYPE:int, edge_list:list)-> list:
    
    """Estimate the size of state for the grid : 

    Args:
        len_all_task_list (int) : Number of Master Node data required  (Number of task lists available)
        list_length_edge_nodes_per_eap ([list]): Number of Edge Node data required 
        
    Returns: 
        s_grid_len : list of size of state for corresponding all_task_list
    """

    deploy_states, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
    
    master_list = create_master_list(node_param_lists, master_param_lists, all_task_list, MAX_TASK_TYPE)

    state_list = get_state_list(master_list, MAX_TASK_TYPE)
    s_grid_len = []
    for i, state in enumerate((state_list)):

        sub_deploy_state = deploy_states[i]
        sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]],[[state[4]]], [[state[3]]], [state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))

        s_grid_len.append(len(sub_elem))
 
    return s_grid_len

def get_action_dims(node_param_lists:list)->list:
    """Estimate the size of Action dimensions for for the Node parameters in the given list : 

    Args:
        node_param_lists (List) : eAP's Node Parameters 
        
    Returns: 
        action_dims : Overall action dimensions
    """

    action_dims = []
    for _list in node_param_lists:

        action_dim = 0
        for _ in _list:
            action_dim+=1
        action_dim+=1            
        action_dims.append(action_dim)
    return action_dims  # because of cluster

def set_lr( optimizer:torch.optim, lr:float)->None:    
    """Method to set the Learning rate of the given optimizer

    Args:
        optimizer ([Pytorch optimizer]): [Optimizer]
        lr ([Float]): [Learning Rate]
    """
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr    

def get_state_list(master_list:list, max_tasks:int)->list:
    state_list = []
    for mast in master_list:
        state_list.append(state_inside_eAP(mast, len(mast.node_list), max_tasks))
    return state_list 

def flatten(list:list)->list:
    """Function to serialize sublists inside a given list

    Args:
        list (list): [ A list of lists]

    Returns:
        [list]: [Serialized list ]
    """    
    return [y for x in list for y in x]
 

def remove_docker_from_master_node(master:Type[Master], change_node_idx:int, service_index:int, deploy_state:list):
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


def deploy_new_docker(master:Type[Master], POD_MEM:float, POD_CPU:float, cur_time:str, change_node_idx:int, service_coefficient:list, service_index:list, deploy_state:list):
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
    
    
def get_current_task(master:Type[Master])->list:
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
            
def state_inside_eAP(master:Type[Master], num_edge_nodes_per_eAP:list, max_tasks:int)->Tuple[list,list,list,list,int, float]:
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
        #print('service_type: ', service_type, master.task_queue)
        #a=b
        delay_requirement = master.task_queue[0][2]-master.task_queue[0][1]
        
    undone_tasks = master.undone
    for i in range(num_edge_nodes_per_eAP):
        cpu_list.append([master.node_list[i].cpu, master.node_list[i].cpu_max])
        mem_list.append([master.node_list[i].mem, master.node_list[i].mem_max])
        task_num.append(len(master.node_list[i].task_queue))

    return cpu_list, mem_list, task_num, undone_tasks, service_type, delay_requirement

def create_node_list(node_specification:list)->list:
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
def create_master_list(node_param_lists:list, master_param_lists:list, all_task_list:list, MAX_TASK_TYPE:int)->list:
    """Creates List of Masters (eAPs)

    Args:
        node_param_lists ([list]): [params of underlying nodes]
        master_param_lists ([list]): [params of master nodes (eAps)]
        all_task_list (list): list of incoming tasks with
        MAX_TASK_TYPE : Number of tasks

    Returns:
        master_list(List): [list of Masters (eAPs)]
    """
    
    node_lists = []
    for node_params in node_param_lists:
        node_lists.append(create_node_list(node_params))
    
    # (cpu, mem,..., achieve task num, give up task num)
    master_list = []
    for i, master_params in enumerate(master_param_lists):
        master_list.append(Master(master_params[0], master_params[1], node_lists[i], [], all_task_list[i], 0, 0, 0, [0] * MAX_TASK_TYPE, [0] * MAX_TASK_TYPE))
    return master_list

def create_eAP_and_Cloud(node_param_lists:list, master_param_lists:list, all_task_list:list, MAX_TASK_TYPE:int, POD_MEM:float,  POD_CPU:float, service_coefficient:list, cur_time:float)->Tuple[list, Type[Cloud]]:
    """Create Edge Access Points and Cloud

    Args:
        node_param_lists ([list]): [list of node specifications]
        master_param_lists ([list]): [list of master specifications]
        all_task_list (list): [list of all tasks data]
        MAX_TASK_TYPE (int) : Maximum number of task types
        POD_MEM: (float): Memory of POD
        POD_CPU (float): CPU of POD
        service_coefficient(list): Service Coefficients
        cur_time (float) : Time
        
    Returns:
        master_list (list) :  List of created Master Nodes
        cloud (Cloud Object) : Created cloud object
    """
    master_list = create_master_list(node_param_lists, master_param_lists, all_task_list, MAX_TASK_TYPE)
    cloud = Cloud([], [], sys.maxsize, sys.maxsize)  # (..., cpu, mem)
    ################################################################################################
    for i in range(MAX_TASK_TYPE):
        docker = Docker(POD_MEM * service_coefficient[i], POD_CPU * service_coefficient[i], cur_time, i, [-1])
        cloud.service_list.append(docker)
    
    return master_list, cloud

def get_last_length(master_list:list)->Tuple[int, list]:
    """Get length of nodes in each eAPs and the total 

    Args:
        master_list ([list]): [list containing the master eAPs]

    Returns:
        last_length(List): total length of action space for each eAP
        length_list : Total length of action space 
    """
    length_list = [0]
    last_length = length_list[0]
    for mstr in master_list:
        length_list.append(last_length + len(mstr.node_list))
        last_length = last_length + len(mstr.node_list)
    return last_length, length_list
        
def put_current_task_on_queue(act:list, curr_task:list, action_dims:list, cloud:Cloud, master_list:list)->None: 
    """Put current tasks on queue

    Args:
        act ([list]): actions
        curr_task ([list]): [list of tasks]
        action_dims (int): Action dimension
        cloud (Cloud Object) : Created cloud object
        master_list (list) :  List of created Master Nodes

    """
    _, length_list = get_last_length(master_list)    
    #print('curr_task : ', curr_task)
    cluster_action_values = [action-1 for action in action_dims]

    for i in range(len(act)):
        if curr_task[i][0] == -1:
            continue
        if act[i] == cluster_action_values[i]:
            cloud.task_queue.append(curr_task[i])
            continue
        
        for j in range(len(length_list)-1):
            if act[i] >= length_list[j] and act[i] < length_list[j+1] :
                master_list[j].node_list[act[i] - length_list[j]].task_queue.append(curr_task[i])
                
            
def update_state_of_task( cur_time:list, check_queue:Callable, cloud:Type[Cloud], master_list:list)->None:
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
    
def update_state_of_dockers(cur_time:float, cloud: Type[Cloud], master_list: list)->Type[Cloud]:
    """Updates the state of dockers

    Args:
        cur_time (float) : Time
        cloud (Cloud Object) : Cloud object
        master_list (list) :  List of created Master Nodes

    """
    for mstr in master_list:
        for i in range(len(mstr.node_list)):
            mstr.node_list[i], undone, done, done_kind, undone_kind = update_docker(mstr.node_list[i], master_list,  cur_time, service_coefficient, POD_CPU)
            for j in range(len(done_kind)):
                mstr.done_kind[done_kind[j]] = mstr.done_kind[done_kind[j]] + 1
            for j in range(len(undone_kind)):
                mstr.undone_kind[undone_kind[j]] = mstr.undone_kind[undone_kind[j]] + 1
            
            for i, master_entity in enumerate(master_list):
                master_entity.update_undone(undone[i])
                master_entity.update_done(done[i])
    cloud, undone, done, done_kind, undone_kind = update_docker(cloud, master_list, cur_time, service_coefficient, POD_CPU)

    for i, master_entity in enumerate(master_list):
                master_entity.update_undone(undone[i])
                master_entity.update_done(done[i])

    return cloud  

def create_dockers(MAX_TASK_TYPE:int, deploy_states:list, service_coefficient:list, POD_MEM:float, POD_CPU:float, cur_time:float, master_list:float)->None:
    """Creation of dockers

    Args:
        
        MAX_TASK_TYPE (int) : Maximum number of task types
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
    for i, deploy_state in enumerate(deploy_states):
        for dep in deploy_state:
            
            for ii in range(MAX_TASK_TYPE):

                decision = dep[ii]
                for j in range(len(length_list)-1):
                    if i>= length_list[j] and i < length_list[j+1] and decision == 1:
                        k= i-length_list[j]
                        if master_list[j].node_list[k].mem >= POD_MEM * service_coefficient[ii]:
                            docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                            ii, [-1])
                            master_list[j].add_to_node_attribute(k, 'mem', - POD_MEM * service_coefficient[ii])
                            master_list[j].append_docker_to_node_service_list(k, docker)                              
    
                            
def get_node_characteristics(master:Type[Master])->Tuple[list, list, list]:
    """Get the node characteristics of masters eAPs

    Args:
        cpu_list (list) : CPU Parameters
        mem_list (list) : Memory Parameters
        task_list (list) :  Task Lists

    """
    cpu_list = []
    mem_list = []
    task_list = []
    for i in range(len(master.node_list)):
        cpu_list.append(master.node_list[i].cpu)
        mem_list.append(master.node_list[i].mem)
        task_list.append(len(master.node_list[i].task_queue))                
    return cpu_list, mem_list, task_list              
      
def get_state_characteristics(MAX_TASK_TYPE:int, master_list:list)->Tuple[list, list, list, list]:
    """Get lists of tasks that are done, undone and current task in queue

    Args:
        MAX_TESK_TYPE ([int]): Maximum types of tasks
        master_list ([Master Object list]): [Edge Access Point list containing nodes]
        

    Returns:
        done_tasks (list) : Done Tasks list
        undone_tasks (list) : Not Done Tasks list
        curr_tasks_in_queue (list) : Current tasks in queue
    """
    done_tasks = []
    undone_tasks = []
    curr_tasks_in_queue = []
    
    # Get task state, include successful, failed, and unresolved
    for i in range(MAX_TASK_TYPE):
        done_val = 0.0
        undone_val = 0.0
        for master in master_list:
            done_val += master.done_kind[i]
            undone_val += master.undone_kind[i]
        done_tasks.append(done_val)
        undone_tasks.append(undone_val)     
        
    for master in master_list:

        for i in range(len(master.node_list)):
            tmp = [0.0] * MAX_TASK_TYPE
            for j in range(len(master.node_list[i].task_queue)):
                tmp[master.node_list[i].task_queue[j][0]] = tmp[master.node_list[i].task_queue[j][0]] + 1.0
            curr_tasks_in_queue.append(tmp)
    
    return done_tasks, undone_tasks, curr_tasks_in_queue


def plot_list(data_list:list, title:str, x_label:str, y_label:str)->None:
    """Plot the given list

    Args:
        data_list (list) : CPU Parameters
        mem_list (list) : Memory Parameters
        task_list (list) :  Task Lists

    """

    plt.figure(figsize=(15,10))

    plt.plot(data_list)#throughput_list)
    plt.title(title)
    plt.xlabel(x_label)#"Number of Episodes")
    plt.ylabel(y_label)#"Throughput rate")
    #plt.ylim([0, 100])
    #plt.show()
    plt.savefig('./plots/'+title + '.png')