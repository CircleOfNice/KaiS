import time
import sys
from algorithm_torch.cMMAC import *
from algorithm_torch.GPG import *
from env.platform import *
from env.env_run import *
import pickle,gzip
from check_helpers_main_pytorch import *                


def calculate_rewards(master1, master2, cur_done, cur_undone):
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
    for i in range(3):
        use_rate1.append(master1.node_list[i].cpu / master1.node_list[i].cpu_max)
        use_rate1.append(master1.node_list[i].mem / master1.node_list[i].mem_max)
        use_rate2.append(master2.node_list[i].cpu / master2.node_list[i].cpu_max)
        use_rate2.append(master2.node_list[i].mem / master2.node_list[i].mem_max)

    standard_list.append(np.std(use_rate1, ddof=1))
    standard_list.append(np.std(use_rate2, ddof=1))

    reward.append(math.exp(-task_fail_rate[0]) + weight * math.exp(-standard_list[0]))
    reward.append(math.exp(-task_fail_rate[1]) + weight * math.exp(-standard_list[1]))
    return reward
       
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

    q_estimator = Estimator(action_dim, state_dim, number_of_master_nodes) # Definition of cMMAc Agent
    replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3)) # experience Replay for value network for cMMMac Agent
    policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3)) #experience Replay for Policy network for cMMMac Agent

    global_step1 = 0
    global_step2 = 0
    all_task1 = get_all_task('./data/Task_1.csv')# processed data [type_list, start_time, end_time, cpu_list, mem_list] fed to eAP 1
    all_task2 = get_all_task('./data/Task_2.csv')# processed data fed to eAP 2

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
        deploy_state = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]]

        # Create clusters based on the hardware resources you need
        node1_1 = Node(100.0, 4.0, [], [])  # (cpu, mem,...)
        node1_2 = Node(200.0, 6.0, [], [])
        node1_3 = Node(100.0, 8.0, [], [])
        node_list1 = [node1_1, node1_2, node1_3]

        node2_1 = Node(200.0, 8.0, [], [])
        node2_2 = Node(100.0, 2.0, [], [])
        node2_3 = Node(200.0, 6.0, [], [])
        node_list2 = [node2_1, node2_2, node2_3]
        # (cpu, mem,..., achieve task num, give up task num)
        master1 = Master(200.0, 8.0, node_list1, [], all_task1, 0, 0, 0, [0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE)
        master2 = Master(200.0, 8.0, node_list2, [], all_task2, 0, 0, 0, [0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE)
        cloud = Cloud([], [], sys.maxsize, sys.maxsize)  # (..., cpu, mem)

        valid_node = get_valid_nodes([node_list1, node_list2])
        # Crerate dockers based on deploy_state
        #create_dockers(vaild_node, MAX_TESK_TYPE, deploy_state, num_edge_nodes_per_eAP, service_coefficient, POD_MEM, POD_CPU, cur_time, master1, master2)
        #create_dockers(valid_node, MAX_TESK_TYPE, deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
        # Crerate dockers based on deploy_state
        for i in range(valid_node):
            for ii in range(MAX_TESK_TYPE):
                dicision = deploy_state[i][ii]
                if i < 3 and dicision == 1:
                    j = i
                    if master1.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master1.node_list[j].mem = master1.node_list[j].mem - POD_MEM * service_coefficient[ii]
                        master1.node_list[j].service_list.append(docker)

                if i >= 3 and dicision == 1:
                    j = i - 3
                    if master2.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,
                                        ii, [-1])
                        master2.node_list[j].mem = master2.node_list[j].mem - POD_MEM * service_coefficient[ii]
                        master2.node_list[j].service_list.append(docker)
        ########### Each slot ###########
        for slot in range(BREAK_POINT):
            cur_time = cur_time + SLOT_TIME
            ########### Each frame ###########
            if slot % CHO_CYCLE == 0 and slot != 0:
                # Get task state, include successful, failed, and unresolved
                done_tasks, undone_tasks, curr_tasks_in_queue = get_state_characteristics(MAX_TESK_TYPE, master1, master2, num_edge_nodes_per_eAP)
                
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
                
                #print(change_service, 'change_service')
                #print(change_node, 'change_node')
                execute_orchestration(change_node, change_service, num_edge_nodes_per_eAP,
                         deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master1, master2)
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
            
            master1 = update_task_queue(master1, cur_time, 0)
            master2 = update_task_queue(master2, cur_time, 1)
            task1 = [-1]
            task2 = [-1]
            if len(master1.task_queue) != 0:
                task1 = master1.task_queue[0]
                del master1.task_queue[0]
            if len(master2.task_queue) != 0:
                task2 = master2.task_queue[0]
                del master2.task_queue[0]
            curr_task = [task1, task2]
            ava_node = []

            for i in range(len(curr_task)):
                tmp_list = [cluster_action_value]  # Cloud is always available
                for ii in range(len(deploy_state)):
                    if deploy_state[ii][curr_task[i][0]] == 1:
                        tmp_list.append(ii)
                ava_node.append(tmp_list)
            
            
            # Current state of CPU and memory
            #TODO it is only possible to generalise this after separation of Q estimaters
            cpu_list1 = []
            mem_list1 = []
            cpu_list2 = []
            mem_list2 = []
            task_num1 = [len(master1.task_queue)]
            task_num2 = [len(master2.task_queue)]
            for i in range(3):
                cpu_list1.append([master1.node_list[i].cpu, master1.node_list[i].cpu_max])
                mem_list1.append([master1.node_list[i].mem, master1.node_list[i].mem_max])
                task_num1.append(len(master1.node_list[i].task_queue))
            for i in range(3):
                cpu_list2.append([master2.node_list[i].cpu, master2.node_list[i].cpu_max])
                mem_list2.append([master2.node_list[i].mem, master2.node_list[i].mem_max])
                task_num2.append(len(master2.node_list[i].task_queue))
            
            
            #print(task_num1, task_num2)
            #a=b    
            s_grid = np.array([flatten(flatten([deploy_state, [task_num1], cpu_list1, mem_list1, [[latency]], [[num_edge_nodes_per_eAP]]])),
                               flatten(flatten([deploy_state, [task_num2],  cpu_list2, mem_list2, [[latency]], [[num_edge_nodes_per_eAP]]]))])
            # Dispatch decision
            #print(s_grid.shape)
            
            #TODO Determine the Action Precisely 
            
            act, valid_action_prob_mat, policy_state, action_choosen_mat, \
            curr_state_value, curr_neighbor_mask, next_state_ids = q_estimator.action(s_grid, ava_node, context,)
            
            ###### Randomising if 0.05 then it is epsilor exploration
            #if random.uniform(0, 1)< 0.05:
            #    	act = [random.randint(0,6), random.randint(0,6)] 
            #act = [random.randint(0,6), random.randint(0,6)] 
            #print('act :', act)
            
            ####
            # Put the current task on the queue based on dispatch decision
            #act = [2, 5]
            #put_current_task_on_queue(act, curr_task, cluster_action_value, cloud, master_list)
            #put_current_task_on_queue(act, curr_task, cluster_action_value, num_edge_nodes_per_eAP, cloud, master1, master2)
            for i in range(len(act)):
                if curr_task[i][0] == -1:
                    continue
                if act[i] == 6:
                    cloud.task_queue.append(curr_task[i])
                    continue
                if act[i] >= 0 and act[i] < 3:
                    master1.node_list[act[i]].task_queue.append(curr_task[i])
                    continue
                if act[i] >= 3 and act[i] < 6:
                    master2.node_list[act[i] - 3].task_queue.append(curr_task[i])
                    continue
                else:
                    pass
            # Update state of task
            #update_state_of_task(num_edge_nodes_per_eAP, cur_time, check_queue, cloud, master1, master2)
            #update_state_of_task(num_edge_nodes_per_eAP, cur_time, check_queue, cloud, master_list[0], master_list[1])
            #TODO update check_queue
            for i in range(3):
                master1.node_list[i].task_queue, undone, undone_kind = check_queue(master1.node_list[i].task_queue,
                                                                                   cur_time)
                for j in undone_kind:
                    master1.undone_kind[j] = master1.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]

                master2.node_list[i].task_queue, undone, undone_kind = check_queue(master2.node_list[i].task_queue,
                                                                                   cur_time)
                for j in undone_kind:
                    master2.undone_kind[j] = master2.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]

            cloud.task_queue, undone, undone_kind = check_queue(cloud.task_queue, cur_time)
            master1.undone = master1.undone + undone[0]
            master2.undone = master2.undone + undone[1]

            # Update state of dockers in every node
            for i in range(3):
                master1.node_list[i], undone, done, done_kind, undone_kind = update_docker(master1.node_list[i],
                                                                                           cur_time,
                                                                                           service_coefficient, POD_CPU)
                for j in range(len(done_kind)):
                    master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
                for j in range(len(undone_kind)):
                    master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]
                master1.done = master1.done + done[0]
                master2.done = master2.done + done[1]

                master2.node_list[i], undone, done, done_kind, undone_kind = update_docker(master2.node_list[i],
                                                                                           cur_time,
                                                                                           service_coefficient, POD_CPU)
                for j in range(len(done_kind)):
                    master1.done_kind[done_kind[j]] = master1.done_kind[done_kind[j]] + 1
                for j in range(len(undone_kind)):
                    master1.undone_kind[undone_kind[j]] = master1.undone_kind[undone_kind[j]] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]
                master1.done = master1.done + done[0]
                master2.done = master2.done + done[1]

            cloud, undone, done, done_kind, undone_kind = update_docker(cloud, cur_time, service_coefficient, POD_CPU)
            
            master1.undone = master1.undone + undone[0]
            master2.undone = master2.undone + undone[1]
            master1.done = master1.done + done[0]
            master2.done = master2.done + done[1]

            cur_done = [master1.done - pre_done[0], master2.done - pre_done[1]]
            cur_undone = [master1.undone - pre_undone[0], master2.undone - pre_undone[1]]

            pre_done = [master1.done, master2.done]
            pre_undone = [master1.undone, master2.undone]

            achieve_num.append(sum(cur_done))
            fail_num.append(sum(cur_undone))
            immediate_reward = calculate_rewards(master1, master2, cur_done, cur_undone)
            #print('immediate_reward : ', immediate_reward)
            record.append([master1, master2, cur_done, cur_undone, immediate_reward])

            deploy_reward.append(sum(immediate_reward))

            if slot != 0:
                r_grid = to_grid_rewards(immediate_reward)
                targets_batch = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)

                # Advantage for policy network.
                advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                          s_grid, r_grid, gamma)
                                                          
                if curr_task[0][0] != -1 and curr_task[1][0] != -1:
                    replay.add(state_mat_prev, action_mat_prev, targets_batch, s_grid)
                    policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage, curr_neighbor_mask_prev)

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
        #a=b

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
            batch_s, _, batch_r, _ = replay.sample()
            q_estimator.update_value(batch_s, batch_r, 1e-3)
            global_step1 += 1

        # update policy network
        for _ in np.arange(TRAIN_TIMES):
            batch_s, batch_a, batch_r, batch_mask = policy_replay.sample()
            q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, batch_mask, learning_rate,)
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
    RUN_TIMES =  3 #500 Number of Episodes to run
    TASK_NUM = 5000 # Time for each Episode Ending
    TRAIN_TIMES = 50 # list containing two elements for tasks done on both master nodes
    CHO_CYCLE = 1000 # Orchestration cycle

    ##############################################################
    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE)