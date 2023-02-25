# Functions comprising the environment

import csv
import random
import numpy as np
from typing import Type, Tuple
import load_data
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
    start_time_list.sort()
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
    #print(min(duration_list), max(duration_list))
    #a=b
    # TODO Tasks which 0 duration potentially run forever, put high number here?
    #duration_list = [i or 10 for i in duration_list] 

    # start_time_list, end_time_list = _generate_fake_start_end_time(duration_list=duration_list, time_scale_fac=1)
    start_time_list = task_info_lists.start_time
    start_time_list, end_time_list = _get_normalized_start_end_time_lists(start_time_list, duration_list)

    result_list = [type_list, start_time_list, end_time_list, cpu_list, mem_list]
    return result_list, max(type_list)
