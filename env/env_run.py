# Functions comprising the environment

import csv
import random
import numpy as np
from typing import Type, Tuple
from env.platform import Master, Node
from env.platform import Docker
from env import load_data
import random


def _convert_mem_units(mem_list:list) -> list:
    """This method converts the units for memory from Millibytes (Kubernetes) to Gigabytes (Kais)

    Args:
        mem_list (list): List of memory requests for the tasks (in Millibytes)

    Returns:
        list: List of memory requests for the tasks (in Gigabytes)
    """
    result_list = [i / 1e+12 for i in mem_list]
    return result_list


def _convert_cpu_units(cpu_list:list) -> list:
    """This method converts the units for cpu requests from Kubernetes to KaiS-Format.

    Kubernetes uses millicore format, whereas were not 100% sure what the unit in KaiS actually is?
    Maybe % of cpu used?
    
    TODO this method does currently nothing as were not sure how to correctly convert the data

    Args:
        cpu_list (list): List of cpu requests for the tasks (in millicores)

    Returns:
        list: Returns the same input list back
    """
    return cpu_list


def _get_normalized_start_end_time_lists(start_time_list:list, duration_list:list) -> Tuple:
    """Method to normalize the start times of tasks and calculating the end times of tasks based on the task duration

    The normalized start_time_list starts at time 0.

    Args:
        start_time_list (list): List of integers with the start times of tasks
        duration_list (list): List of integers with the end times of tasks.

    Returns:
        Tuple: start_time_list, end_time_list
    """
    start_time = start_time_list[0]
    start_time_list = [i - start_time for i in start_time_list]
    end_time_list = np.array(start_time_list) + np.array(duration_list)
    end_time_list = end_time_list.tolist()

    return start_time_list, end_time_list


def _generate_fake_start_end_time(duration_list:list, time_scale_fac:float=1) -> Tuple:
    """Method to generate fake start and end times for data.

    With a time_scale_fac of 1 the naive assumption is, that the start and endtimes exactly fit all the tasks
    as defined in duration lists (on a 1 node system).

    Increasing of the time_scale_fac increases the endtime and vice versa.

    Args:
        duration_list (list): List of the duration of all tasks.
        time_scale_fac (float, optional): Scaling factor. Defaults to 1.

    Returns:
        Tuple: start_time_list, end_time_list
    """
    full_duration = sum(duration_list)
    max_time = full_duration * time_scale_fac

    time_list = random.sample(range(0, max_time), k=len(duration_list))

    start_time_list = sorted(time_list)
    end_time_list = [start_time_list[i] + duration_list[i] for i in range(len(duration_list))]

    return (start_time_list, end_time_list)



def get_all_task_kubernetes(path:str, randomize:bool = True) -> Tuple:
    """Method to load kubernetes data from a .csv file.

    Method outputs data in the same format as 'get_all_task()' Method.

    Since we currently do not have access to start and endtimes, we generate them ourselves

    The type of Tasks are all set to 1, since we dont differentiate (for now) between  different tasks in kubernetes

    Args:
        path (str): Path to the .csv file
        randomize (bool, optional): Wether to randomize the endtimes of tasks. Currently not used. Defaults to True.

    Returns:
        Tuple: [0]: The list with all the task information
                [type_list, start_time_list, end_time_list, cpu_req_list, mem_req_list]
                [1]: The highest service type in the list

    """
    task_info_lists = load_data.get_all_task(path)

    cpu_list = task_info_lists.cpu_req
    cpu_list = _convert_cpu_units(cpu_list=cpu_list)

    mem_list = task_info_lists.mem_req
    mem_list = _convert_mem_units(mem_list=mem_list)

    # TODO setting task type to 1 for every task, is that a problem?
    type_list = [1] * len(cpu_list) 

    # TODO with a step size of 0.5 secs do we need to double the duration?
    duration_list = task_info_lists.task_duration

    # TODO Tasks which 0 duration potentially run forever, put high number here?
    duration_list = [i or 10 for i in duration_list] 

    # start_time_list, end_time_list = _generate_fake_start_end_time(duration_list=duration_list, time_scale_fac=1)
    start_time_list = task_info_lists.start_time
    start_time_list, end_time_list = _get_normalized_start_end_time_lists(start_time_list, duration_list)

    result_list = [type_list, start_time_list, end_time_list, cpu_list, mem_list]
    return result_list, max(type_list)


def get_all_task(path:str, randomize:bool = True)->Tuple:
    """Get Processed data from the file given in path

    Args:
        path ([str]): [path to the file]

    Returns:
        [list]: [lists containing type of task , start time & end time of tasks, cpu and memory]
    """
    type_list = []
    start_time = []
    end_time = []
    cpu_list = []
    mem_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            type_list.append(row[3])
            start_time.append(row[5])
            end_time.append(row[6])
            cpu_list.append(row[7])
            mem_list.append(row[8])

    init_time = int(start_time[0])
    new_type_list = []
    new_start_time = []
    new_end_time = []
    new_cpu_list = []
    new_mem_list = []

    for i in range(len(start_time)):
        
        
        start_time_ = (int(start_time[i]) - init_time)
        end_time_ = (int(end_time[i]) - init_time) 
        type_list_ = (int(type_list[i]) - 1)
        cpu_list_ = (int(cpu_list[i]) / 100.0)
        mem_list_ = (float(mem_list[i]))
        time_diff = int(end_time_ - start_time_)
        
        
        if randomize:
        
            if time_diff > 200:
                end_time_ = end_time_ + random.randint(-10, 10)
            if time_diff > 500:
                end_time_ = end_time_ + random.randint(-50, 50)
            if time_diff > 1000:
                end_time_ = end_time_ + random.randint(-100, 100)
            if time_diff > 1500:
                end_time_ = end_time_ + random.randint(-200, 200)

        if time_diff<0:
            continue
        
        new_start_time.append(start_time_)
        new_end_time.append(end_time_)
        new_cpu_list.append(cpu_list_)
        new_mem_list.append(mem_list_)
        new_type_list.append(type_list_)
    new_all_task = [new_type_list, new_start_time, new_end_time, new_cpu_list, new_mem_list]
    
    return new_all_task, max(new_type_list)


def put_task(task_queue:list, task:int)->list:
    """Puts tasks on the task queue

    Args:
        task_queue ([list]): [list of queued tasks ]
        task ([type]): [task to be put on the task_queue]

    Returns:
        [list]: [list of queued tasks ]]
    """
    for i in range(len(task_queue) - 1):
        j = len(task_queue) - i - 1
        task_queue[j] = task_queue[j - 1]
    task_queue[0] = task
    return task_queue


def update_task_queue(master:Type[Master], cur_time:float, master_id:int)->Type[Master]:
    """[summary]

    Args:
        master ([Master Object]): [eAP object]
        cur_time ([int]): [Current time]
        master_id ([int]): [number of the eAP object]

    Returns:
        ([Master Object]): [eAP object]
    """
    # clean task for overtime
    i = 0
    while len(master.task_queue) > i:
        if master.task_queue[i][0] == -1:
            i = i + 1
            continue
        if cur_time >= master.task_queue[i][2]:
            master.undone = master.undone + 1
            master.undone_kind[master.task_queue[i][0]] = master.undone_kind[master.task_queue[i][0]] + 1
            del master.task_queue[i]
        else:
            i = i + 1
    while master.all_task[1][master.all_task_index] < cur_time:
        task = [master.all_task[0][master.all_task_index], master.all_task[1][master.all_task_index],
                master.all_task[2][master.all_task_index], master.all_task[3][master.all_task_index],
                master.all_task[4][master.all_task_index], master_id]
        master.task_queue.append(task)
        master.all_task_index = master.all_task_index + 1

    tmp_list = []
    for i in range(len(master.task_queue)):
        if master.task_queue[i][0] != -1:
            tmp_list.append(master.task_queue[i])
    tmp_list = sorted(tmp_list, key=lambda x: (x[2], x[1]))

    master.task_queue = tmp_list
    return master


def check_queue(task_queue:list, cur_time:float, length_masterlist:int)->Tuple:
    """[Check the queue for tasks in queue]

    Args:
        task_queue ([list]): [list containing the queue]
        cur_time ([int]): [current time]

    Returns:
        [list]: [tasks in the queue, tasks undone(not done), there type(kind)]
    """

    task_queue = sorted(task_queue, key=lambda x: (x[2], x[1]))
    undone =[]
    for un in range(length_masterlist):
        undone.append(0)
        
    undone_kind = []
    # clean task for overtime
    i = 0
    while len(task_queue) != i:
        flag = 0
        if cur_time >= task_queue[i][2]:
            
            undone[task_queue[i][5]] = undone[task_queue[i][5]] + 1
            undone_kind.append(task_queue[i][0])
            del task_queue[i]
            flag = 1
        if flag == 1:
            flag = 0
        else:
            i = i + 1
        
    return task_queue, undone, undone_kind

def update_docker(node:Type[Node], master_list:list, cur_time:float, service_coefficient:list)->Tuple:
    done = []
    undone = []
    done_kind = []
    undone_kind = []    
    flag = 0
    if node.task_queue!=[-1]:
        
        task_queue = []
        for i in range(len(node.task_queue)):
            
            if node.cpu >= node.task_queue[i][3] * service_coefficient[node.task_queue[i][0]] and \
                node.mem >= node.task_queue[i][4] * service_coefficient[node.task_queue[i][0]]:       
                flag  = 1  
                docker_container = Docker(node.task_queue[i][3] * service_coefficient[node.task_queue[i][0]], 
                                        node.task_queue[i][4] * service_coefficient[node.task_queue[i][0]], 
                                        node.task_queue[i][2], node.task_queue[i][0], node.task_queue[i]) 
                node.service_list.append(docker_container)
                node.cpu = node.cpu - node.task_queue[i][3] * service_coefficient[node.task_queue[i][0]]
                node.mem = node.mem - node.task_queue[i][4] * service_coefficient[node.task_queue[i][0]]
                task_queue.append(i)
                done_kind.append(node.task_queue[i][0]) 
                done.append(1)
        if flag ==1:
            new_node_task_queue= [node.task_queue[x] for x in range(len(node.task_queue)) if x not in task_queue]
            node.task_queue = new_node_task_queue
    else : 
        pass    
    remove_elements=  []
    remove_element_flag = 0
    for i, service in enumerate(node.service_list):
        if service.available_time > cur_time:
            continue
        elif service.available_time <= cur_time:
            node.cpu = node.cpu + service.doing_task[3] * service_coefficient[service.kind]
            node.mem = node.mem + service.doing_task[4] * service_coefficient[service.kind]
            undone.append(1)
            undone_kind.append(service.doing_task[0])
            remove_elements.append(i)
            remove_element_flag =1  
    if remove_element_flag ==1:      
        new_node_service_list = [node.service_list[x] for x in range(len(node.service_list)) if x not in remove_elements]
        node.service_list = new_node_service_list
    return node, undone, done, done_kind, undone_kind 