
import time
import logging
logger = logging.getLogger(__name__)  
import random
import matplotlib.pyplot as plt
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
import pickle,gzip
from helpers_main_pytorch import *                
from algorithm_torch.CMMAC_Value_Model import build_value_model, update_value

# Make the initial state for 

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
    #####################################################################
    ########### Init ###########
    record = [] # list used to dump all the eAPS, tasks in queue, done tasks and undone tasks etcs
    throughput_list = [] # list of the progress of task done / total number of tasks
    #sum_rewards = [] # Used to store average rewards in the list 
    achieve_num = [] # List to contain the currently tasks done in the requirement space  
    fail_num = [] # Number of tasks failed to meet the requirement
    #deploy_reward = [] # List of sum of immediate rewards to be stored in experience
    
    all_rewards = [] # List to accumulate all rewards 
    order_response_rate_episode = [] # List to monitor the average throughput rate
    episode_rewards = [] # Accumulated reward over episodes
    record_all_order_response_rate = [] # List to record all the throughput rate througout episodes
    csv_paths = ['./data/Task_1.csv', './data/Task_2.csv']
    all_task_list_init = []
    max_task_pool_init = []
    for csv_path in csv_paths:
        all_task, max_task = get_all_task(csv_path, randomize=randomize_data)
        all_task_list_init.append(all_task)
        max_task_pool_init.append(max_task)
        
    max_tasks = max(max_task_pool_init) 
    
    all_task_list = []
    if total_eaps !=0:
        for i in range(total_eaps):
            
            val_int = random.randint(0,len(max_task_pool_init) -1)
            all_task_list.append(all_task_list_init[val_int])
    else:
        pass

    if randomize ==False:
        # For uniform Edge Nodes per eAP
        edge_list = [nodes_in_cluster]*len(all_task_list)
        # For random Edge Nodes per eAP
    else:
        edge_list = [random.sample(range(low_bound_edge_mode, upper_bound_edge_mode), 1)[0] for i in range(len(all_task_list))]

    _, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
    
    action_dims = get_action_dims(node_param_lists)

    # Definition of cMMAc Agent
    q_estimator_list = []
    ReplayMemory_list = []
    policy_replay_list = []
    s_grid_len = estimate_state_size(all_task_list, max_tasks, edge_list)

    if randomize ==False:
        for i in range(len(master_param_lists)):
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    else:
        for i in range(len(master_param_lists)):
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    logger.debug('Multiple Actors initialised')
    # Creation of global critic (currently without cloud info of unprocessed requests)
    critic, critic_optimizer = build_value_model(sum(s_grid_len)+ 1) # Length of task queue can be only one digit long
    logger.debug('centralised critic initialised')
    global_step1 = 0
    global_step2 = 0

    orchestrate_agent = OrchestrateAgent(output_dim*len(master_param_lists) + node_input_dim , scale_input_dim, hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1), MAX_TESK_TYPE, eps=1e-6, act_fn = act_function,optimizer=opt_function)
    
    logger.debug('Initialization of orchestration agent complete')
    exp = {'node_inputs': [], 'scale_inputs': [], 'reward': [], 'wall_time': [], 'node_act_vec': [],
           'scale_act_vec': []}

    for n_iter in np.arange(RUN_TIMES):
        ########### Initialize the setup and repeat the experiment many times ###########
        batch_reward = []
        cur_time = 0
        entropy_weight = entropy_weight_init
        order_response_rates = []

        pre_done = [] # list to track tasks done previously
        pre_undone = [] # list to track tasks undone (not done) previously
        context = [] # Flag
        for i in range(len(master_param_lists)):
            pre_done.append(0)
            pre_undone.append(0)
            context.append(1)

        ############ Set up according to your own needs  ###########
        # The parameters here are set only to support the operation of the program, and may not be consistent with the actual system
        # At each edge node 1 denotes a kind of service which is running

        deploy_states, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
        
        # Create clusters based on the hardware resources you need

        master_list, cloud = create_eAP_and_Cloud(node_param_lists, master_param_lists, all_task_list, MAX_TESK_TYPE, POD_MEM,  POD_CPU, service_coefficient, cur_time)
        
        # Creation of node Graph CNN
        graph_cnn_list = []
        for master in master_list:
            cpu_list, mem_list, task_list = get_node_characteristics(master) 
            
            graph_cnn_list.append(GraphCNN(len(cpu_list)+ len(mem_list) + len(mem_list), hid_dims, output_dim, max_depth, act_function))

        # Create dockers based on deploy_state

        create_dockers( MAX_TESK_TYPE, deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
        logger.debug('Outer loop initialization done')
        ########### Each slot ###########
        for slot in range(BREAK_POINT):
            cur_time = cur_time + SLOT_TIME
            ########### Each frame ###########
            if slot % CHO_CYCLE == 0 and slot != 0:
                logger.info('Orchestration Cycle n_iter, CHO_CYCLE : {} {}', str(n_iter), str(CHO_CYCLE))
                # Get task state, include successful, failed, and unresolved
                done_tasks, undone_tasks, curr_tasks_in_queue,  = get_state_characteristics(MAX_TESK_TYPE, master_list)  

                cpu_lists =[]
                mem_lists = []
                task_lists = []
                for master in master_list:
                    cpu_list, mem_list, task_list = get_node_characteristics(master)  
                    cpu_lists.append(cpu_list)
                    mem_lists.append(mem_list)
                    task_lists.append(task_list)
                
                reward_val = float(get_gpg_reward(master_list))
                exp['reward'].append(reward_val)
                #deploy_reward = []
                exp['wall_time'].append(cur_time)
                deploy_states_float = []
                 
                deploy_states_float = []
                for item, deploy_state in enumerate(deploy_states):

                    elem_list = []
                    for d_state in deploy_state:
                        sub_elem_list = []
                        for item in d_state:
                            sub_elem_list.append(float(item))
                        
                        elem_list.append(sub_elem_list)
 
                    deploy_states_float.append(elem_list)    

                
                # Orchestration
                node_choice, service_scaling_choice, exp = orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue,deploy_states_float, cpu_lists, mem_lists, task_lists, graph_cnn_list, MAX_TESK_TYPE, epsilon_exploration)

                logger.info('Orchestration of Decision done ')
                # Randomising Orchestration
                if epsilon_exploration:
                    
                    if random.uniform(0, 1)< 0.05:
                        service_scaling_choice = torch.randint(-max_tasks-1, max_tasks+1, (len(service_scaling_choice),))
                        node_choice = torch.randint(0, sum(action_dims), (len(service_scaling_choice),))
                
                # Here is the code for orchestration and service scaling
                execute_orchestration(node_choice, service_scaling_choice, #num_edge_nodes_per_eAP,
                         deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
                
                logger.info('Execution orchestration')
                # Save data
                if slot % CHO_CYCLE/10==0:
                    exp_tmp = exp
                    
                    entropy_weight, loss = train_orchestrate_agent(orchestrate_agent, exp_tmp, entropy_weight,
                                                                   entropy_weight_min, entropy_weight_decay)
                    entropy_weight = decrease_var(entropy_weight,
                                                  entropy_weight_min, entropy_weight_decay)
                    logger.info('Training orchestration agent')

            # Get current task
            for i, master in enumerate(master_list):
                master_list[i] = update_task_queue(master, cur_time, i)  
            curr_task = []
            for master in master_list:
                curr_task.append(get_current_task(master))
  
            ava_node = []

            for i in range(len(curr_task)):
                if randomize ==False:
                
                    tmp_list = [action_dims[i] -1]  # Cloud is always available
                else:
                    
                    tmp_list = [action_dims[i] -1]
                deploy_state = deploy_states[i]
                for ii in range(len(deploy_state)):

                    if deploy_state[ii][curr_task[i][0]] == 1:
                        tmp_list.append(ii)
                ava_node.append(tmp_list)

            state_list = get_state_list(master_list, max_tasks)    

            s_grid = []
            for i, state in enumerate((state_list)):
                sub_deploy_state = deploy_states[i]
                sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]], [[state[4]]], [[state[3]]],[state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
                s_grid.append(sub_elem)
            critic_state = flatten(s_grid)

            critic_state.append(len(cloud.task_queue))
            # Dispatch decision

            act = []
            valid_action_prob_mat = []
            policy_state = []
            action_choosen_mat = []
            curr_state_value = []
            curr_neighbor_mask = []
            next_state_ids = []
            for i in range(len(s_grid)):
                
                act_, valid_action_prob_mat_, policy_state_, action_choosen_mat_, \
                curr_state_value_, curr_neighbor_mask_, next_state_ids_ = q_estimator_list[i].action(np.array(s_grid[i]), critic, critic_state, ava_node[i], context,)

                act.append(act_[0])
                valid_action_prob_mat.append(valid_action_prob_mat_[0])
                policy_state.append(policy_state_[0])
                action_choosen_mat.append(action_choosen_mat_[0])
                curr_state_value.append(curr_state_value_[0])
                curr_neighbor_mask.append(curr_neighbor_mask_[0])
                next_state_ids.append(next_state_ids_[0])
            valid_action_prob_mat = np.array(valid_action_prob_mat)
            policy_state = np.array(policy_state)
            action_choosen_mat = np.array(action_choosen_mat)
            curr_neighbor_mask = np.array(curr_neighbor_mask)

            ###### Randomising if 0.05 then it is epsilor exploration
            if epsilon_exploration:
                if random.uniform(0, 1)< 0.05:
                	act = [random.randint(0,sum(action_dims)), random.randint(0,sum(action_dims))] 
            ####
            # Put the current task on the queue based on dispatch decision
            
            put_current_task_on_queue(act, curr_task, action_dims, cloud, master_list)
            # Update state of task
            update_state_of_task(cur_time, check_queue, cloud, master_list)
            
            # Update state of dockers in every node
            cloud = update_state_of_dockers(cur_time, cloud, master_list)
                
            cur_done = []
            cur_undone = []
            ch_pre_done = []
            ch_pre_undone = []
            for i, mstr in enumerate(master_list):
                cur_done.append(mstr.done - pre_done[i])
                cur_undone.append(mstr.undone - pre_undone[i])
                
                ch_pre_done.append(mstr.done)
                ch_pre_undone.append(mstr.undone)

            pre_done = ch_pre_done
            pre_undone = ch_pre_undone
            achieve_num.append(sum(cur_done))
            fail_num.append(sum(cur_undone))
            immediate_reward = calculate_reward(master_list, cur_done, cur_undone)

            record.append([master_list, cur_done, cur_undone, immediate_reward])

            #deploy_reward.append(sum(immediate_reward))

            if slot != 0:
                logger.debug('Computing targets for cMMAC')
                r_grid = to_grid_rewards(immediate_reward)
                for m in range(len(r_grid)):

                    targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m]], np.array(critic_state), critic, r_grid[[m]], curr_neighbor_mask_prev[m], gamma)
                    # Advantage for policy network.
                    advantage = q_estimator_list[m].compute_advantage([curr_state_value_prev[m]], [next_state_ids_prev[m]] ,
                                                            np.array(critic_state), critic, r_grid[[m],:], gamma)

                    
                    test_cond_list = []
                    for i, elem in enumerate(curr_task):
                        test_cond_list.append(elem[0] != -1)# != -1
                    
                    cond = test_cond_list[0]
                    
                    if len(test_cond_list)>1:
                        for i in range(1,len(test_cond_list)):
                            cond = cond and test_cond_list[i]
                    if cond:

                        ReplayMemory_list[m].add(np.array([state_mat_prev]), action_mat_prev[[m]], targets_batch[[0]], np.array([s_grid[m]]))
                        policy_replay_list[m].add(policy_state_prev[[m]], action_choosen_mat_prev[[m]], advantage , curr_neighbor_mask_prev[[m]])
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

            all_rewards.append(sum(immediate_reward))
            batch_reward.append(immediate_reward)

            if (sum(cur_done) + sum(cur_undone)) != 0:
                order_response_rates.append(float(sum(cur_done) / (sum(cur_done) + sum(cur_undone))))
            else:
                order_response_rates.append(0)

        #sum_rewards.append(float(sum(all_rewards)) / float(len(all_rewards)))
        all_rewards = []

        all_number = sum(achieve_num) + sum(fail_num)
        throughput_list.append(sum(achieve_num) / float(all_number))
        #logger.info('Logging for run time throughput_list_all =', throughput_list, '\ncurrent_achieve_number =', sum(achieve_num),
        #      ', current_fail_number =', sum(fail_num))
        print('throughput_list_all =', throughput_list, '\ncurrent_achieve_number =', sum(achieve_num),
              ', current_fail_number =', sum(fail_num))
        achieve_num = []
        fail_num = []

        episode_reward = np.sum(batch_reward[1:])
        episode_rewards.append(episode_reward)
        n_iter_order_response_rate = np.mean(order_response_rates[1:])
        order_response_rate_episode.append(n_iter_order_response_rate)
        record_all_order_response_rate.append(order_response_rates)
        # update value network
        for _ in np.arange(TRAIN_TIMES):

            for m in range(len(master_list)):
                batch_s, _, batch_r, _ = ReplayMemory_list[m].sample()
                update_value(batch_s, batch_r, 1e-3, critic, critic_optimizer)
            global_step1 += 1

        # update policy network
        for _ in np.arange(TRAIN_TIMES):

            for m in range(len(master_list)):
                batch_s, batch_a, batch_r, batch_mask = policy_replay_list[m].sample()
                q_estimator_list[m].update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,)
            
            global_step2 += 1
            
        logger.info('Done training for run time {}', str(n_iter))
        
    print('Average throughput Achieved : ', sum(throughput_list)/len(throughput_list))
    name = 'full_randomisation_orchestration_no_randomisation_'
    time_str = str(time.time())
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(record, f)
                
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", 'rb') as fp:
        record = pickle.load(fp)

    with gzip.open("./result/throughput_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(throughput_list, f)
    
    if randomize==True:
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_low_bound_edge_mpde_'+ str(low_bound_edge_mpde) + '_upper_bound_edge_mpde_'+ str(upper_bound_edge_mpde) 
    else : 
        title =     "Total_Eaps_" + str(len(all_task_list)) + '_nodes_in_cluster_'+ str(nodes_in_cluster)
    plt.figure(figsize=(15,10))

    plt.plot(throughput_list)
    plt.title(title)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Throughput rate")
    #plt.ylim([0, 100])
    #plt.show()
    plt.savefig('./plots/'+title + '.png') 
    
    return throughput_list
    
if __name__ == "__main__":
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    RUN_TIMES = 10#20#0#20 #500 # Number of Episodes to run
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