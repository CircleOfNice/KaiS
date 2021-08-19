class Cloud:
    """[This class serves as framework for definition of Cloud with properties such as 
    task queue, service_list, cpu processing and memory]
    """
    def __init__(self, task_queue, service_list, cpu, mem):
        self.task_queue = task_queue
        self.service_list = service_list
        self.cpu = cpu  # GHz
        self.mem = mem  # GB


class Node:
    """[This class serves as framework for definition of Edge Node with properties such as 
    task queue, service_list, cpu processing and  memory]
    """
    def __init__(self, cpu, mem, service_list, task_queue):
        self.cpu = cpu
        self.cpu_max = cpu
        self.mem = mem
        self.mem_max = mem
        self.service_list = service_list
        self.task_queue = task_queue


class Master:
    """[This class serves as framework for definition of Master Node with properties such as 
    task queue, service_list, cpu processing, memory, done and undone tasks, Kind of tasks done and undone. all task index]
    """
    def __init__(self, cpu, mem, node_list, task_queue, all_task, all_task_index, done, undone, done_kind, undone_kind):
        self.cpu = cpu  # GHz
        self.mem = mem  # MB
        self.node_list = node_list
        self.task_queue = task_queue
        self.all_task = all_task
        self.all_task_index = all_task_index
        self.done = done
        self.undone = undone
        self.done_kind = done_kind
        self.undone_kind = undone_kind

    def update_done(self,done):
        self.done = self.done + done
        
    def update_undone(self, undone):
        self.undone = self.undone + undone
        
    def add_to_node_attribute(self, node_list_idx, attribute, value):
        updated_value = getattr(self.node_list[node_list_idx], attribute) + value
        setattr(self.node_list[node_list_idx], attribute, value)
        
    def append_docker_to_node_service_list(self, node_list_idx, docker_cont):
        self.node_list[node_list_idx].service_list.append(docker_cont)
        
class Docker:
    """[This class serves as framework for definition of Docker containers with properties such as 
    cpu processing, memory, Time since availability, Service kind,  task being done( generally [-1])]
    """
    def __init__(self, mem, cpu, available_time, kind, doing_task):
        self.mem = mem
        self.cpu = cpu
        self.available_time = available_time
        self.kind = kind
        self.doing_task = doing_task
