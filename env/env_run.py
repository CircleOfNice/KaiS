# Functions comprising the environment

import csv


def get_all_task(path):
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
    for i in range(len(start_time)):
        type_list[i] = int(type_list[i]) - 1
        start_time[i] = int(start_time[i]) - init_time
        end_time[i] = int(end_time[i]) - init_time
        cpu_list[i] = int(cpu_list[i]) / 100.0
        mem_list[i] = float(mem_list[i])
    all_task = [type_list, start_time, end_time, cpu_list, mem_list]

    return all_task, max(type_list)


def put_task(task_queue, task):
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


def update_task_queue(master, cur_time, master_id):
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


def check_queue(task_queue, cur_time, length_masterlist):
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


def update_docker(node, master_list, cur_time, service_coefficient, POD_CPU):
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
