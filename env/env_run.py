# Functions comprising the environment

import csv
import random
from typing import Type, Tuple
from env.platform import Master, Node
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
    # get new task
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


def update_docker(node:Type[Node], master_list:list, cur_time:float, service_coefficient:list, POD_CPU:float)->Tuple:
    """[Update the docker given the node]

    Args:
        node ([Node Object]): [Edge node]
        cur_time ([int]): [current time]
        service_coefficient ([float]): [number identifying intensity (demand ) of the tasks]
        POD_CPU ([float]): [CPU resources required to make a POD]

    Returns:
        [list]: [Edge node, undone and done task lists with list of tasks that are done and undone]
    """
    
    done = []
    undone = []
    for i in range(len(master_list)):
        done.append(0)
        undone.append(0)
        
    done_kind = []
    undone_kind = []

    # find achieved task in current time
    for i in range(len(node.service_list)):
        if node.service_list[i].available_time <= cur_time and len(node.service_list[i].doing_task) > 1:
            done[node.service_list[i].doing_task[5]] = done[node.service_list[i].doing_task[5]] + 1
            done_kind.append(node.service_list[i].doing_task[0])
            node.service_list[i].doing_task = [-1]
            node.service_list[i].available_time = cur_time
    # execute task in queue
    i = 0
    while i != len(node.task_queue):
        flag = 0
        for j in range(len(node.service_list)):
            if i == len(node.task_queue):
                break
            if node.task_queue[i][0] == node.service_list[j].kind:
                if node.service_list[j].available_time > cur_time:
                    continue
                if node.service_list[j].available_time <= cur_time:
                    to_do = (node.task_queue[i][3]) / node.service_list[j].cpu
                    if cur_time + to_do <= node.task_queue[i][2] and node.cpu >= POD_CPU * service_coefficient[
                        node.task_queue[i][0]]:
                        node.cpu = node.cpu - POD_CPU * service_coefficient[node.task_queue[i][0]]
                        node.service_list[j].available_time = cur_time + to_do
                        node.service_list[j].doing_task = node.task_queue[i]
                        del node.task_queue[i]
                        flag = 1

                    elif cur_time + to_do > node.task_queue[i][2]:
                        undone[node.task_queue[i][5]] = undone[node.task_queue[i][5]] + 1
                        print('undone[node.task_queue[i][5]] : ', undone[node.task_queue[i][5]])
                        undone_kind.append(node.task_queue[i][0])
                        del node.task_queue[i]
                        flag = 1
                    elif node.cpu < POD_CPU * service_coefficient[node.task_queue[i][0]]:
                        pass

        if flag == 1:
            flag = 0
        else:
            i = i + 1

    return node, undone, done, done_kind, undone_kind
