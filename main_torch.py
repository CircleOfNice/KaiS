import time
import sys

from numpy.lib.index_tricks import s_

from algorithm_torch.cMMAC import *
from algorithm_torch.GPG import *
from env.platform import *
from env.env_run import *
import pickle,gzip
from helpers_main_pytorch import *                
       
def execution(RUN_TIMES, BREAK_POINT, TRAIN_TIMES, CHO_CYCLE):
    
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
    sum_rewards = [] # Used to store average rewards in the list 
    achieve_num = [] # List to contain the currently tasks done in the requirement space  
    fail_num = [] # Number of tasks failed to meet the requirement
    deploy_reward = [] # List of sum of immediate rewards to be stored in experience
    
    all_rewards = [] # List to accumulate all rewards 
    order_response_rate_episode = [] # List to monitor the average throughput rate
    episode_rewards = [] # Accumulated reward over episodes
    record_all_order_response_rate = [] # List to record all the throughput rate througout episodes

    all_task1 = get_all_task('./data/Task_1.csv')# processed data [type_list, start_time, end_time, cpu_list, mem_list] fed to eAP 1
    all_task2 = get_all_task('./data/Task_2.csv')# processed data fed to eAP 2
    all_task_list = [all_task1, all_task2]
    
    #q_estimator = Estimator(action_dim, state_dim, number_of_master_nodes) # Definition of cMMAc Agent
    q_estimator_list = []
    ReplayMemory_list = []
    policy_replay_list = []
    s_grid_len = estimate_state_size(all_task_list)

    for i in range(number_of_master_nodes):
        q_estimator_list.append(Estimator(action_dim, s_grid_len[i], 1)) # Definition of cMMAc Agent
        ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
        policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent

    global_step1 = 0
    global_step2 = 0

    orchestrate_agent = OrchestrateAgent(node_input_dim, cluster_input_dim, hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1), MAX_TESK_TYPE, eps=1e-6, act_fn = nn.functional.leaky_relu,optimizer=torch.optim.Adam)
    exp = {'node_inputs': [], 'cluster_inputs': [], 'reward': [], 'wall_time': [], 'node_act_vec': [],
           'cluster_act_vec': []}

    for n_iter in np.arange(RUN_TIMES):
        ########### Initialize the setup and repeat the experiment many times ###########
        batch_reward = []
        cur_time = 0
        entropy_weight = entropy_weight_init
        order_response_rates = []

        pre_done = [0, 0] # list to track tasks done previously
        pre_undone = [0, 0] # list to track tasks undone (not done) previously
        context = [1, 1] # Flag
        ############ Set up according to your own needs  ###########
        # The parameters here are set only to support the operation of the program, and may not be consistent with the actual system
        # At each edge node 1 denotes a kind of service which is running
        '''
        deploy_state = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]]

        
        node_list_1 = [[100.0, 4.0], [200.0, 6.0], [100.0, 8.0]]
        node_list_2 = [[200.0, 8.0], [100.0, 2.0], [200.0, 6.0]]
        
        node_param_lists = [node_list_1, node_list_2]
        
        master_param_lists = [[200.0, 8.0], [200.0, 8.0]]
        '''
        deploy_state, node_param_lists, master_param_lists = return_state_values()
        # Create clusters based on the hardware resources you need

        master_list, cloud = create_eAP_and_Cloud(node_param_lists, master_param_lists, all_task_list, MAX_TESK_TYPE, POD_MEM,  POD_CPU, service_coefficient, cur_time)
        
        # Crerate dockers based on deploy_state
        valid_node = 0
        for mast in master_list:
            valid_node = valid_node + len(mast.node_list)
        create_dockers(valid_node, MAX_TESK_TYPE, deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
        ########### Each slot ###########
        for slot in range(BREAK_POINT):
            cur_time = cur_time + SLOT_TIME
            ########### Each frame ###########
            if slot % CHO_CYCLE == 0 and slot != 0:
                # Get task state, include successful, failed, and unresolved
                done_tasks, undone_tasks, curr_tasks_in_queue = get_state_characteristics(MAX_TESK_TYPE, master_list)  
                   
                if slot != CHO_CYCLE:
                    exp['reward'].append(float(sum(deploy_reward)) / float(len(deploy_reward)))
                    deploy_reward = []
                    exp['wall_time'].append(cur_time)

                deploy_state_float = []
                for i in range(len(deploy_state)):
                    tmp = []
                    for j in range(len(deploy_state[0])):
                        tmp.append(float(deploy_state[i][j]))
                    deploy_state_float.append(tmp)

                # Orchestration
                change_node, change_service, exp = orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue,deploy_state_float, MAX_TESK_TYPE)
                
                # Randomising Orchestration
                
                #if random.uniform(0, 1)< 0.05:
                #    change_service = torch.randint(-12, 12, (3,))
                #    change_node = torch.randint(0, 6, (3,))
                #    print('Randomising Orchestration')
                #print('Not Randomising Orchestration')
                
                execute_orchestration(change_node, change_service, #num_edge_nodes_per_eAP,
                         deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
                # Save data
                if slot > 3 * CHO_CYCLE:
                    exp_tmp = exp

                    del exp_tmp['node_inputs'][-1]
                    del exp_tmp['cluster_inputs'][-1]
                    del exp_tmp['node_act_vec'][-1]
                    del exp_tmp['cluster_act_vec'][-1]
                    
                    entropy_weight, loss = train_orchestrate_agent(orchestrate_agent, exp_tmp, entropy_weight,
                                                                   entropy_weight_min, entropy_weight_decay)
                    entropy_weight = decrease_var(entropy_weight,
                                                  entropy_weight_min, entropy_weight_decay)

            # Get current task
            for i, master in enumerate(master_list):
                master_list[i] = update_task_queue(master, cur_time, i)
                
            curr_task = []
            for master in master_list:
                curr_task.append(get_current_task(master))
                
            ava_node = []

            for i in range(len(curr_task)):
                tmp_list = [cluster_action_value]  # Cloud is always available
                for ii in range(len(deploy_state)):
                    if deploy_state[ii][curr_task[i][0]] == 1:
                        tmp_list.append(ii)
                ava_node.append(tmp_list)

            #state_list = []
            #for mast in master_list:
            #    state_list.append(state_inside_eAP(mast, len(mast.node_list)))
            state_list = get_state_list(master_list)    
            last_length, length_list = get_last_length(master_list)
            s_grid = []
            for i, state in enumerate((state_list)):
                sub_deploy_state = deploy_state[length_list[i]:length_list[i+1]]
                #print('sub_deploy_state : ', sub_deploy_state)
                sub_elem = flatten(flatten([sub_deploy_state, [state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
                s_grid.append(sub_elem)
              
            # Dispatch decision
            #TODO Determine the Action Precisely 
            act = []
            valid_action_prob_mat = []
            policy_state = []
            action_choosen_mat = []
            curr_state_value = []
            curr_neighbor_mask = []
            next_state_ids = []
            for i in range(len(s_grid)):
                #print('s_grid : ', len(s_grid[i]))
                act_, valid_action_prob_mat_, policy_state_, action_choosen_mat_, \
                curr_state_value_, curr_neighbor_mask_, next_state_ids_ = q_estimator_list[i].action(np.array(s_grid[i]), ava_node[i], context,)
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
            #if random.uniform(0, 1)< 0.05:
            #    	act = [random.randint(0,6), random.randint(0,6)] 
            #act = [random.randint(0,6), random.randint(0,6)] 
            #print('act :', act)
            
            ####
            # Put the current task on the queue based on dispatch decision
            put_current_task_on_queue(act, curr_task, cluster_action_value, cloud, master_list)

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

            deploy_reward.append(sum(immediate_reward))

            if slot != 0:
                
                r_grid = to_grid_rewards(immediate_reward)
                for m in range(len(r_grid)):
                    targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m],:], np.array([s_grid[m]]), r_grid[[m],:], curr_neighbor_mask_prev[[m],:], gamma)
                    # Advantage for policy network.
                    advantage = q_estimator_list[m].compute_advantage([curr_state_value_prev[m]], [next_state_ids_prev[m]] ,
                                                            np.array([s_grid[m]]), r_grid[[m],:], gamma)
                                      
                    if curr_task[0][0] != -1 and curr_task[1][0] != -1:
                        ReplayMemory_list[m].add(np.array([state_mat_prev[m]]), action_mat_prev[[m],:], targets_batch[[0],:], np.array([s_grid[m]]))
                        policy_replay_list[m].add(policy_state_prev[[m],:], action_choosen_mat_prev[[m],:], advantage , curr_neighbor_mask_prev[[m],:])

            # For updating
            state_mat_prev = s_grid
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

        sum_rewards.append(float(sum(all_rewards)) / float(len(all_rewards)))
        all_rewards = []

        all_number = sum(achieve_num) + sum(fail_num)
        throughput_list.append(sum(achieve_num) / float(all_number))
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
                q_estimator_list[m].update_value(batch_s, batch_r, 1e-3)
            global_step1 += 1

        # update policy network
        for _ in np.arange(TRAIN_TIMES):
            for m in range(len(master_list)):
                batch_s, batch_a, batch_r, batch_mask = policy_replay_list[m].sample()
                q_estimator_list[m].update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,)
            
            global_step2 += 1
    name = 'full_randomisation_orchestration_no_randomisation_'
    time_str = str(time.time())
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(record, f)
                
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", 'rb') as fp:
        record = pickle.load(fp)

    with gzip.open("./result/throughput_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(throughput_list, f)
    return throughput_list
    
if __name__ == "__main__":
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    RUN_TIMES = 10 #500 # Number of Episodes to run
    TASK_NUM = 5000 # Time for each Episode Ending
    TRAIN_TIMES = 50 # list containing two elements for tasks done on both master nodes
    CHO_CYCLE = 1000 # Orchestration cycle

    ##############################################################
    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE)