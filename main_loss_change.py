
import time
import logging
logger = logging.getLogger(__name__)  
import random

# set log level
logger.setLevel(logging.ERROR)

# define file handler and set formatter
file_handler = logging.FileHandler('logfile.log', 'w')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)

from algorithm_torch.cMMAC import *
from algorithm_torch.GPG import *
from env.platform import *
from env.env_run import *
from helpers_main_pytorch import *                

from major_functions import initialize_eap_params, initialize_cmmac_agents, initialize_episode_params, get_task_state_deploy_state_and_exp, generate_plots
from major_functions import get_updated_tasks_ava_node_states, get_estimators_output, put_and_update_tasks, update_exp_replays, train_actor_critic_without_orchestration, check_and_dump

def execution(RUN_TIMES, BREAK_POINT, TRAIN_TIMES, CHO_CYCLE, randomize, total_eaps, low_bound_edge_mpde, upper_bound_edge_mpde, nodes_in_cluster, randomize_data, epsilon_exploration):
    """[Function to execute the KAIS Algorithm ]

    Args:
        RUN_TIMES ([int]): [Number of Episodes to run]
        BREAK_POINT ([int]): [Time for each Episode Ending]
        TRAIN_TIMES ([list]): [list containing two elements for tasks done on both master nodes]
        CHO_CYCLE ([list]): Orchestration cycle
        
    Returns:
        [List]: [Throughput List (Achieved task/ total number of tasks)]
    """
    ########### Init ###########
    record = [] # list used to dump all the eAPS, tasks in queue, done tasks and undone tasks etcs
    throughput_list = [] # list of the progress of task done / total number of tasks
    achieve_num = [] # List to contain the currently tasks done in the requirement space  
    fail_num = [] # Number of tasks failed to meet the requirement
    order_response_rate_episode = [] # List to monitor the average throughput rate
    episode_rewards = [] # Accumulated reward over episodes
    
    log_orchestration_loss, log_estimator_value_loss, log_estimator_policy_loss = [], [], []
    global_step1 = 0
    global_step2 = 0
    
    csv_paths = ['./data/Task_1.csv', './data/Task_2.csv']
    
    exp = {'node_inputs': [], 'scale_inputs': [], 'reward': [], 'wall_time': [], 'node_act_vec': [], 'scale_act_vec': []}
    
    max_tasks, all_task_list,edge_list, _, master_param_lists, action_dims = initialize_eap_params(csv_paths, total_eaps, nodes_in_cluster,
                                                                                                                                  low_bound_edge_mode, upper_bound_edge_mode, randomize_data, randomize)
    MAX_TASK_TYPE = max_tasks +1
    critic, critic_optimizer, q_estimator_list, ReplayMemory_list, ReplayMemory_list, policy_replay_list = initialize_cmmac_agents(MAX_TASK_TYPE, all_task_list,edge_list, master_param_lists, action_dims, randomize)
    
    logger.debug('Multiple Actors initialised')
    logger.debug('centralised critic initialised')
    
    for i in range(len(q_estimator_list)):
        log_estimator_policy_loss.append([])
    
    orchestrate_agent = OrchestrateAgent(output_dim*len(master_param_lists) + 2*MAX_TASK_TYPE , hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1), MAX_TASK_TYPE, entropy_weight_init,eps=1e-6, act_fn = act_function,optimizer=opt_function)
    logger.debug('Initialization of orchestration agent complete')
    
    for n_iter in np.arange(RUN_TIMES):
        ########### Initialize the setup and repeat the experiment many times ###########
        batch_reward = []
        cur_time = 0
        order_response_rates = []
        
        ############ Set up according to your own needs  ###########
        # The parameters here are set only to support the operation of the program, and may not be consistent with the actual system
        # At each edge node 1 denotes a kind of service which is running
        master_list, cloud, graph_cnn_list, deploy_states, pre_done, pre_undone, context = initialize_episode_params(all_task_list, edge_list, MAX_TASK_TYPE, cur_time)
        logger.debug('Outer loop initialization done')
        ########### Each slot ###########
        for slot in range(BREAK_POINT):
            cur_time = cur_time + SLOT_TIME
            ########### Each frame ###########
            if slot % CHO_CYCLE == 0 and slot != 0:
                logger.info('Orchestration Cycle n_iter, CHO_CYCLE : {} {}', str(n_iter), str(CHO_CYCLE))
                
                # Get task state, include successful, failed, and unresolved
                done_tasks, undone_tasks, curr_tasks_in_queue, deploy_states_float, exp, cpu_lists, mem_lists, task_lists = get_task_state_deploy_state_and_exp(MAX_TASK_TYPE,
                                                                                                                                                                master_list, exp, deploy_states, cur_time)
                # Orchestration
                node_choice, service_scaling_choice, exp = orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue,deploy_states_float,
                                                                                cpu_lists, mem_lists, task_lists, graph_cnn_list, MAX_TASK_TYPE, epsilon_exploration)
                logger.info('Orchestration of Decision done ')
                
                # Randomising Orchestration
                if epsilon_exploration:
                    if random.uniform(0, 1)< 0.05:
                        service_scaling_choice = torch.randint(-max_tasks-1, max_tasks+1, (len(service_scaling_choice),))
                        node_choice = torch.randint(0, sum(action_dims), (len(service_scaling_choice),))
                
                # Here is the code for orchestration and service scaling
                execute_orchestration(node_choice, service_scaling_choice, 
                         deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
                logger.info('Execution orchestration')
                
                # Save data
                if slot % CHO_CYCLE/10==0:
                    orchestrate_agent.entropy_weight, loss = train_orchestrate_agent(orchestrate_agent, exp)
                    log_orchestration_loss.append(loss.item())
                    logger.info('Training orchestration agent')
                    
            master_list, curr_task, ava_node, s_grid, critic_state = get_updated_tasks_ava_node_states(master_list, cloud, deploy_states, action_dims, cur_time, max_tasks, randomize)
            
            # Dispatch decision
            act, valid_action_prob_mat, policy_state, action_choosen_mat, curr_neighbor_mask, curr_state_value, next_state_ids = get_estimators_output(q_estimator_list, s_grid,critic, critic_state, ava_node, context)
            ###### Randomising if 0.05 then it is epsilor exploration
            if epsilon_exploration:
                if random.uniform(0, 1)< 0.05:
                	act = [random.randint(0,sum(action_dims)), random.randint(0,sum(action_dims))] 

            pre_done, pre_undone, cur_done, cur_undone, cloud  = put_and_update_tasks(act, curr_task, action_dims, cloud, master_list,check_queue, cur_time, pre_done, pre_undone)
            achieve_num.append(sum(cur_done))
            fail_num.append(sum(cur_undone))

            immediate_reward = calculate_reward(master_list, cur_done, cur_undone)
            record.append([master_list, cur_done, cur_undone, immediate_reward])

            if slot != 0:
                logger.debug('Computing targets for cMMAC')
                update_exp_replays(immediate_reward, q_estimator_list, ReplayMemory_list, policy_replay_list, action_mat_prev, critic_state, critic, s_grid, curr_task, 
                                   state_mat_prev, curr_neighbor_mask_prev, curr_state_value_prev, next_state_ids_prev, policy_state_prev, action_choosen_mat_prev)      
            # For updating
            state_mat_prev = critic_state
            action_mat_prev = valid_action_prob_mat
            action_choosen_mat_prev = action_choosen_mat
            curr_neighbor_mask_prev = curr_neighbor_mask
            policy_state_prev = policy_state

            # for computing advantage
            curr_state_value_prev = curr_state_value
            next_state_ids_prev = next_state_ids
            global_step1 += 1
            global_step2 += 1
            
            batch_reward.append(immediate_reward)

            if (sum(cur_done) + sum(cur_undone)) != 0:
                order_response_rates.append(float(sum(cur_done) / (sum(cur_done) + sum(cur_undone))))
            else:
                order_response_rates.append(0)

        all_number = sum(achieve_num) + sum(fail_num)
        throughput_list.append(sum(achieve_num) / float(all_number))
        print('throughput_list_all =', throughput_list, '\ncurrent_achieve_number =', sum(achieve_num), ', current_fail_number =', sum(fail_num))
        
        achieve_num = []
        fail_num = []
        episode_reward = np.sum(batch_reward[1:])
        episode_rewards.append(episode_reward)
        n_iter_order_response_rate = np.mean(order_response_rates[1:])
        order_response_rate_episode.append(n_iter_order_response_rate)
        
        train_actor_critic_without_orchestration(ReplayMemory_list, policy_replay_list, master_list, q_estimator_list, critic, critic_optimizer, 
                                                 log_estimator_value_loss, log_estimator_policy_loss, TRAIN_TIMES)
        logger.info('Done training for run time {}', str(n_iter))
        
    print('Average throughput Achieved : ', sum(throughput_list)/len(throughput_list))
    
    name = 'full_randomisation_orchestration_no_randomisation_'
    time_str = str(time.time())
    check_and_dump(name, time_str, record, throughput_list)
    
    generate_plots(all_task_list, throughput_list, log_orchestration_loss, log_estimator_value_loss, log_estimator_policy_loss, randomize, low_bound_edge_mpde, upper_bound_edge_mpde, nodes_in_cluster)
    print('dumped')
    return throughput_list
    
if __name__ == "__main__":
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    RUN_TIMES = 5#10#20#0#20 #500 # Number of Episodes to run
    TASK_NUM = 5000 # 5000 Time for each Episode Ending # Though episodes are actually longer
    TRAIN_TIMES = 10#50 # Training Iterations for policy and value networks (Actor , Critic)
    CHO_CYCLE = 1000 # Orchestration cycle

    ##############################################################
    # New configuration settings
    nodes_in_cluster =3
    low_bound_edge_mode = 2
    upper_bound_edge_mode = 6
    total_eaps = 2 #random.sample(range(low_bound_edge_mpde, upper_bound_edge_mpde), 1)[0]
    randomize = False #False # Change it as per needs
    randomize_data = False
    epsilon_exploration = False # Not the default implementation for this project

    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE, randomize, total_eaps, low_bound_edge_mode, upper_bound_edge_mode, nodes_in_cluster, randomize_data, epsilon_exploration)