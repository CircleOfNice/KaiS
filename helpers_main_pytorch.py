import math
import numpy as np
from env.platform import *
from env.env_run import update_docker
from algorithm_torch.GPG import act_offload_agent

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
action_dim = 7 #Number of actions possibly edge nodes 6 plus Cluster
state_dim = 88 #Dimension of state for cMMAC (flattened Deployed state, task num, cpu_list, min_list)

node_input_dim = 24 # Input dimension of Node part of the Orchestration Net 
cluster_input_dim = 24 # Input dimension for Cluster part of the Orchestration Net

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

num_edge_nodes_per_eAP =3
cluster_action_value = 6


def flatten(list):
    """Function to serialize sublists inside a given list

    Args:
        list (list): [ A list of lists]

    Returns:
        [list]: [Serialized list ]
    """    
    return [y for x in list for y in x]
    
def calculate_reward(master1, master2, cur_done, cur_undone, num_edge_nodes_per_eAP):
    """
    Tailored MARDL for Decentralised request dispatch - Reward : Improve the longterm throughput while ensuring the load balancing at the edge
    
    [Function that returns rewards from environment given master nodes and the current tasks]

    Args:
        master1 ([Class]): [Master Node containing the cpu and memory values]
        master2 ([Class]): [Master Node containing the cpu and memory values]
        cur_done ([list]): [list containing two elements for tasks done on both master nodes]
        cur_undone ([list]): [list containing two elements for tasks not done yet on both master nodes]

    Returns:
        reward [list]: [list of rewards for both master nodes]
    """
    
    weight = 1.0
    all_task = [float(cur_done[0] + cur_undone[0]), float(cur_done[1] + cur_undone[1])]
    fail_task = [float(cur_undone[0]), float(cur_undone[1])]
    reward = []
    # The ratio of requests that violate delay requirements
    task_fail_rate = []
    if all_task[0] != 0:
        task_fail_rate.append(fail_task[0] / all_task[0])
    else:
        task_fail_rate.append(0)

    if all_task[1] != 0:
        task_fail_rate.append(fail_task[1] / all_task[1])
    else:
        task_fail_rate.append(0)

    # The standard deviation of the CPU and memory usage
    standard_list = []
    use_rate1 = []
    use_rate2 = []
    for i in range(num_edge_nodes_per_eAP):
        use_rate1.append(master1.node_list[i].cpu / master1.node_list[i].cpu_max)
        use_rate1.append(master1.node_list[i].mem / master1.node_list[i].mem_max)
        use_rate2.append(master2.node_list[i].cpu / master2.node_list[i].cpu_max)
        use_rate2.append(master2.node_list[i].mem / master2.node_list[i].mem_max)

    standard_list.append(np.std(use_rate1, ddof=1))
    standard_list.append(np.std(use_rate2, ddof=1))

    reward.append(math.exp(-task_fail_rate[0]) + weight * math.exp(-standard_list[0]))
    reward.append(math.exp(-task_fail_rate[1]) + weight * math.exp(-standard_list[1]))
    # Immediate reward   e^(-lambda - weight_of_load_balancing *standard_deviation_of_cpu_memory)
    
    # Two rewards for each eAP
    return reward
    
    
    
def to_grid_rewards(node_reward):
    
    """[Serialises the given node rewards]

    Returns:
        [list]: [serialised numpy array]
    """
    
    return np.array(node_reward).reshape([-1, 1])
    

def get_state_characteristics(MAX_TESK_TYPE, master1, master2, num_edge_nodes_per_eAP):
    """Get lists of tasks that are done, undone and current task in queue

    Args:
        MAX_TESK_TYPE ([int]): Maximum types of tasks
        master1 ([Master Object]): [Edge Access Point 1]
        master2 ([Master Object]): [Edge Access Point 2]
        num_edge_nodes_per_eAP ([type]): [Number of edge nodes under an edge access point]

    Returns:
        [lists]: [lists of number of done, undone and current tasks in queue]
    """
    done_tasks = []
    undone_tasks = []
    curr_tasks_in_queue = []
    # Get task state, include successful, failed, and unresolved
    for i in range(MAX_TESK_TYPE):
        done_tasks.append(float(master1.done_kind[i] + master2.done_kind[i]))
        undone_tasks.append(float(master1.undone_kind[i] + master2.undone_kind[i]))
        
    for i in range(num_edge_nodes_per_eAP):
        tmp = [0.0] * MAX_TESK_TYPE
        for j in range(len(master1.node_list[i].task_queue)):
            tmp[master1.node_list[i].task_queue[j][0]] = tmp[master1.node_list[i].task_queue[j][0]] + 1.0
        curr_tasks_in_queue.append(tmp)
    for i in range(num_edge_nodes_per_eAP):
        tmp = [0.0] * MAX_TESK_TYPE
        for k in range(len(master2.node_list[i].task_queue)):
            tmp[master2.node_list[i].task_queue[k][0]] = tmp[master2.node_list[i].task_queue[k][0]] + 1
        curr_tasks_in_queue.append(tmp)    
    return done_tasks, undone_tasks, curr_tasks_in_queue

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
            
def state_inside_eAP(master, num_edge_nodes_per_eAP):
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
    for i in range(num_edge_nodes_per_eAP):
        cpu_list.append([master.node_list[i].cpu, master.node_list[i].cpu_max])
        mem_list.append([master.node_list[i].mem, master.node_list[i].mem_max])
        task_num.append(len(master.node_list[i].task_queue))
    return cpu_list, mem_list, task_num



def create_dockers(vaild_node, MAX_TESK_TYPE, deploy_state, num_edge_nodes_per_eAP, service_coefficient, POD_MEM, POD_CPU, cur_time, master1, master2):
    for i in range(vaild_node):
            for ii in range(MAX_TESK_TYPE):
                dicision = deploy_state[i][ii]
                if i < num_edge_nodes_per_eAP and dicision == 1:
                    j = i
                    if master1.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master1.add_to_node_attribute(j, 'mem', - POD_MEM * service_coefficient[ii])
                        master1.append_docker_to_node_service_list(j, docker)

                if i >= num_edge_nodes_per_eAP and dicision == 1:
                    j = i - num_edge_nodes_per_eAP
                    if master2.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master2.add_to_node_attribute(j, 'mem', - POD_MEM * service_coefficient[ii])
                        master2.append_docker_to_node_service_list(j, docker)
                        
                        
def orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue,deploy_state_float, MAX_TESK_TYPE,num_edge_nodes_per_eAP,
                         deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master1, master2):
    # Make decision of orchestration
    change_node, change_service, exp = act_offload_agent(orchestrate_agent, exp, done_tasks,
                                                                     undone_tasks, curr_tasks_in_queue,
                                                                     deploy_state_float, MAX_TESK_TYPE)

    # Execute orchestration
    for i in range(len(change_node)):
        if change_service[i] < 0:
            # Delete docker and free memory
            service_index = -1 * change_service[i] - 1
            if change_node[i] < num_edge_nodes_per_eAP:
                remove_docker_from_master_node(master1, change_node[i], service_index, deploy_state)

            else:
                node_index = change_node[i] - num_edge_nodes_per_eAP
                remove_docker_from_master_node(master2, node_index, service_index, deploy_state)

        else:
            # Add docker and tack up memory
            service_index = change_service[i] - 1
            if change_node[i] < num_edge_nodes_per_eAP:
                if master1.node_list[change_node[i]].mem >= POD_MEM * service_coefficient[service_index]:
                    deploy_new_docker(master1, POD_MEM, POD_CPU, cur_time, change_node[i], service_coefficient, service_index, deploy_state)
            else:
                node_index = change_node[i] - num_edge_nodes_per_eAP
                if master2.node_list[node_index].mem >= POD_MEM * service_coefficient[service_index]:
                    deploy_new_docker(master2, POD_MEM, POD_CPU, cur_time, node_index, service_coefficient, service_index, deploy_state)
                    
def put_current_task_on_queue(act, curr_task, cluster_action_value, num_edge_nodes_per_eAP, cloud, master1, master2):
    for i in range(len(act)):
        if curr_task[i][0] == -1:
            continue
        if act[i] == cluster_action_value:
            cloud.task_queue.append(curr_task[i])
            continue
        if act[i] >= 0 and act[i] < num_edge_nodes_per_eAP:
            master1.node_list[act[i]].task_queue.append(curr_task[i])
            continue
        if act[i] >= num_edge_nodes_per_eAP and act[i] < cluster_action_value:
            master2.node_list[act[i] - num_edge_nodes_per_eAP].task_queue.append(curr_task[i])
            continue
        else:
            pass         
        
def update_state_of_task(num_edge_nodes_per_eAP, cur_time, check_queue, cloud, master1, master2):
                
    for i in range(num_edge_nodes_per_eAP):
        master1.node_list[i].task_queue, undone, undone_kind = check_queue(master1.node_list[i].task_queue,
                                                                        cur_time)
        for j in undone_kind:
            master1.undone_kind[j] = master1.undone_kind[j] + 1
        master1.update_undone(undone[0])
        master2.update_undone(undone[1])
        
        master2.node_list[i].task_queue, undone, undone_kind = check_queue(master2.node_list[i].task_queue,
                                                                        cur_time)
        for j in undone_kind:
            master2.undone_kind[j] = master2.undone_kind[j] + 1
        master1.update_undone(undone[0])
        master2.update_undone(undone[1])
        

    cloud.task_queue, undone, undone_kind = check_queue(cloud.task_queue, cur_time)
    master1.update_undone(undone[0])
    master2.update_undone(undone[1])    
    
    

def update_state_of_dockers(cur_time, cloud, master1, master2):
                
    for i in range(num_edge_nodes_per_eAP):
        master1.node_list[i], undone, done, done_kind, undone_kind = update_docker(master1.node_list[i],
                                                                                cur_time,
                                                                                service_coefficient, POD_CPU)
        for j in range(len(done_kind)):
            master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
        for j in range(len(undone_kind)):
            master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
        
        master1.update_undone(undone[0])
        master2.update_undone(undone[1])
        master1.update_done(done[0])
        master2.update_done(done[1])

        master2.node_list[i], undone, done, done_kind, undone_kind = update_docker(master2.node_list[i],
                                                                                cur_time,
                                                                                service_coefficient, POD_CPU)
        for j in range(len(done_kind)):
            master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
        for j in range(len(undone_kind)):
            master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
        
        master1.update_undone(undone[0])
        master2.update_undone(undone[1])
        master1.update_done(done[0])
        master2.update_done(done[1])

    cloud, undone, done, done_kind, undone_kind = update_docker(cloud, cur_time, service_coefficient, POD_CPU)


    master1.update_undone(undone[0])
    master2.update_undone(undone[1])
    master1.update_done(done[0])
    master2.update_done(done[1])
    
    return cloud   