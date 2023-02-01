from env.platform import *
from env.env_run import *
from algorithm_torch.major_functions import get_all_task_lists, generate_task_lists_for_eaps, get_state_list, put_and_update_tasks
from algorithm_torch.major_functions import create_eAP_and_Cloud, create_dockers, get_done_undone_context, update_task_queue_master_list, update_current_task_master_list 
from algorithm_torch.helpers_main_pytorch import  get_state_list, def_initial_state_values
from algorithm_torch.helpers_main_pytorch import service_coefficient, POD_MEM, POD_CPU
import time

total_eaps = 1
csv_paths = ['./data/Task_1.csv']
nodes_in_cluster =1
randomize_data = False
BREAK_POINT= 5000
SLOT_TIME = 0.5
deploy_state_stack = [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]
node_list_stack = [[100.0, 4.0]]
master_param_list_stack = [[200.0, 8.0]]
deploy_states = deploy_state_stack#[]
master_param_lists =[]
list_length_edge_nodes_per_eap = [1]
act =[0]
cur_time = 0
time_var = 0
all_task_list_init, max_task_pool_init = get_all_task_lists(csv_paths, randomize_data)
max_tasks = max(max_task_pool_init) 
MAX_TASK_TYPE = max_tasks+1

all_task_list = generate_task_lists_for_eaps(total_eaps, max_task_pool_init, all_task_list_init)

edge_list = [nodes_in_cluster]*len(all_task_list)

_, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)

# Time based instances
master_list = create_eAP_and_Cloud(node_param_lists, master_param_lists, all_task_list, MAX_TASK_TYPE, POD_MEM,  POD_CPU, service_coefficient, cur_time)

pre_done, pre_undone, context = get_done_undone_context(master_param_lists)
len_list = []
empty_list = []
# Create dockers based on deploy_state

for slot in range(BREAK_POINT):
    cur_time = cur_time + SLOT_TIME
    master_list = update_task_queue_master_list(master_list, cur_time)    
    curr_task = update_current_task_master_list(master_list)

    state_list = get_state_list(master_list, max_tasks)
    
    if state_list[0][5] != 100000:
        time.sleep(time_var)
        pre_done, pre_undone, cur_done, cur_undone  = put_and_update_tasks(act, curr_task,  master_list,check_queue, cur_time, pre_done, pre_undone)
        #pre_done, pre_undone, cur_done, cur_undone  = put_and_update_tasks(act, curr_task,  master_list,check_queue, cur_time, pre_done, pre_undone)
    else:
        empty_list.append(state_list[0][5])


print('len(len_list) : ', len(len_list))
print('len(empty_list) : ', len(empty_list))



