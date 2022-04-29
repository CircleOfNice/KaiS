
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

'''
def def_initial_state_values(len_all_task_list=3, list_length_edge_nodes_per_eap=[3, 3, 3]):
    
    deploy_state_stack = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],]
    node_list_stack = [[100.0, 4.0], [200.0, 6.0], [100.0, 8.0], [200.0, 8.0], [100.0, 2.0], [200.0, 6.0]]
    master_param_list_stack = [[200.0, 8.0]]
    print('list_length_edge_nodes_per_eap : ', list_length_edge_nodes_per_eap)
    length_deploy_state = sum(list_length_edge_nodes_per_eap)
    deploy_state = []
    master_param_lists =[]
    for i in range(length_deploy_state):
        deploy_state.append(random.choice(deploy_state_stack))
    for i in range(len_all_task_list):
        master_param_lists.append(random.choice(master_param_list_stack))
    
    node_param_lists = []    
    for i in range(len_all_task_list):

        node_list = []
        for k in range(list_length_edge_nodes_per_eap[i]):
            node_list.append(random.choice(node_list_stack))
        node_param_lists.append(node_list)
    print('deploy_state : ', len(deploy_state))
    return deploy_state, node_param_lists, master_param_lists

'''

def def_initial_state_values(len_all_task_list=3, list_length_edge_nodes_per_eap=[3, 3, 3]):
    
    deploy_state_stack = [[0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],]
    node_list_stack = [[100.0, 4.0], [200.0, 6.0], [100.0, 8.0], [200.0, 8.0], [100.0, 2.0], [200.0, 6.0]]
    master_param_list_stack = [[200.0, 8.0]]
    print('list_length_edge_nodes_per_eap : ', list_length_edge_nodes_per_eap)
    length_deploy_state = sum(list_length_edge_nodes_per_eap)
    deploy_states = []
    master_param_lists =[]
    for item in list_length_edge_nodes_per_eap:
        deploy_state = []
        for i in range(item):
            deploy_state.append(random.choice(deploy_state_stack))
        deploy_states.append(deploy_state)
    
    
    for i in range(len_all_task_list):
        master_param_lists.append(random.choice(master_param_list_stack))
    
    node_param_lists = []    
    for i in range(len_all_task_list):

        node_list = []
        for k in range(list_length_edge_nodes_per_eap[i]):
            node_list.append(random.choice(node_list_stack))
        node_param_lists.append(node_list)
    print('deploy_states : ', deploy_states)
    return deploy_states, node_param_lists, master_param_lists

def estimate_state_size(all_task_list, max_tasks, edge_list):

    deploy_states, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
    
    master_list = create_master_list(node_param_lists, master_param_lists, all_task_list)
    '''
    state_list = []
    for mast in master_list:
        state_list.append(state_inside_eAP(mast, len(mast.node_list)))
    '''                
    last_length, length_list = get_last_length(master_list)
    state_list = get_state_list(master_list, max_tasks)
    s_grid_len = []
    for i, state in enumerate((state_list)):
        print('i : ' , i)
        print('state : ', state)
        print('len(state) : ', len(state))
        #print('length_list[i]:length_list[i+1] : ', length_list[i], length_list[i+1])
        print('deploy_states[i] : ', deploy_states[i])
        #print('deploy_states[length_list[i]:length_list[i+1]] : ', deploy_states[length_list[i]:length_list[i+1]])
        sub_deploy_state = deploy_states[i]#deploy_states[length_list[i]:length_list[i+1]]
        sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]],[[state[4]]], [[state[3]]], [state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
        print('sub_elem : ', sub_elem)
        print('len(sub_elem) : ', len(sub_elem))
        s_grid_len.append(len(sub_elem))
 
    return s_grid_len

def get_action_dims(node_param_lists):
    action_dims = []
    for _list in node_param_lists:

        action_dim = 0
        for node_param in _list:
            action_dim+=1
        action_dim+=1            
        action_dims.append(action_dim)
    return action_dims  # because of cluster

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

    all_task1, max_task_type1 = get_all_task('./data/Task_1.csv')# processed data [type_list, start_time, end_time, cpu_list, mem_list] fed to eAP 1
    all_task2, max_task_type2 = get_all_task('./data/Task_2.csv')# processed data fed to eAP 2
    
    
    # Creation of Dummy Data and eAPs and Edge Nodes
    all_task_list = [all_task1, all_task2]
    max_task_pool = [max_task_type1, max_task_type2]
    max_tasks = max(max_task_type1, max_task_type2)
    extra_eaps = 0 
    nodes_in_cluster =3
    low_bound_edge_mpde = 2
    upper_bound_edge_mpde =6
    
    randomize = False # Change it as per needs
    
    if extra_eaps !=0:
        for i in range(extra_eaps):
            val_int = random.randint(0,len(max_task_pool) -1)

            all_task_list.append(all_task_list[val_int])
    else:
        pass
    
    if randomize ==False:
        # For uniform Edge Nodes per eAP
        edge_list = [nodes_in_cluster]*len(all_task_list)
        print('inside not randomise')
        # For random Edge Nodes per eAP
    else:
        print('inside randomise')
        edge_list = [random.sample(range(low_bound_edge_mpde, upper_bound_edge_mpde), 1)[0] for i in range(len(all_task_list))]
    print('edge_list : ', edge_list)
    _, node_param_lists, master_param_lists = def_initial_state_values(len(all_task_list), edge_list)
    print('len(node_param_lists) : ',  len(node_param_lists))
    #a=b
    action_dim = get_action_dim(node_param_lists)
    
    cluster_action_value = action_dim -1
    
    action_dims = get_action_dims(node_param_lists)

    #print('Action Dims : ', action_dims)
    #cluster_action_values = []
    #for action_dim in action_dims:
    #    cluster_action_values = action_dims -1
    # Definition of cMMAc Agent
    q_estimator_list = []
    ReplayMemory_list = []
    policy_replay_list = []
    s_grid_len = estimate_state_size(all_task_list, max_tasks, edge_list)

    if randomize ==False:
        for i in range(len(master_param_lists)):
            print('dont randomise')
            print('s_grid_len[i] : ', s_grid_len[i])
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    else:
        print('randomise')
        for i in range(len(master_param_lists)):
            q_estimator_list.append(Estimator(action_dims[i], s_grid_len[i], 1)) # Definition of cMMAc Agent
            ReplayMemory_list.append(ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) # experience Replay for value network for cMMMac Agent
            policy_replay_list.append(policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))) #experience Replay for Policy network for cMMMac Agent
    #a=b
    logger.debug('Multiple Actors initialised')
    # Creation of global critic (currently without cloud info of unprocessed requests)
    critic, critic_optimizer = build_value_model(sum(s_grid_len)+ 1) # Length of task queue can be only one digit long
    logger.debug('centralised critic initialised')
    global_step1 = 0
    global_step2 = 0

    orchestrate_agent = OrchestrateAgent(output_dim*len(master_param_lists) + node_input_dim , scale_input_dim, hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1), MAX_TESK_TYPE, eps=1e-6, act_fn = act_function,optimizer=opt_function)
    
    logger.debug('Initialization of orchestration agent complete')
    #orchestrate_agent = OrchestrateAgent(node_input_dim, scale_input_dim, hid_dims, output_dim, max_depth,
    #                                     range(1, exec_cap + 1), MAX_TESK_TYPE, eps=1e-6, act_fn = act_function,optimizer=opt_function)
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
        valid_node = 0
        for mast in master_list:
            print('len(mast.node_list) : ', len(mast.node_list))
            valid_node = valid_node + len(mast.node_list)
        for deploy_state in deploy_states:
            print('deploy_state : ', len(deploy_state))    
        #a=b
        create_dockers(valid_node, MAX_TESK_TYPE, deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
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
                deploy_reward = []
                exp['wall_time'].append(cur_time)
                deploy_states_float = []
                
                #print('deploy_state : ', deploy_state)
                 
                deploy_states_float = []
                for item, deploy_state in enumerate(deploy_states):
                    deploy_state_float = []
                    elem_list = []
                    #print('deploy_state : ', deploy_state)
                    for d_state in deploy_state:
                        #print('d_state : ',d_state)
                        sub_elem_list = []
                        for item in d_state:
                            sub_elem_list.append(float(item))
                        #print('sub_elem_list : ', sub_elem_list)
                        
                        elem_list.append(sub_elem_list)
                    '''
                    for i in range(len(deploy_state)):
                        tmp = []
                        for j in range(len(deploy_state[i])):
                            print('deploy_state[i] : ', deploy_state[j])
                            sub_temp = []
                            for elem in deploy_state[j]:
                                sub_temp.append(float(elem))
                                print('elem : ', elem)
                            #tmp.append(float(deploy_state[i][j]))
                            tmp.append(sub_temp)
                            
                        deploy_state_float.append(tmp)
                      
                    print('elem_list : ', elem_list)
                    print('elem_list len: ', len(elem_list))
                    '''  
                    deploy_states_float.append(elem_list)    
                    
                print('deploy_states initial: ', deploy_states)    
                print('deploy_states_float after : ', deploy_states_float)
                print('deploy_states_float before len : ', len(deploy_states))
                print('deploy_states_float after len : ', len(deploy_states_float))
                #print('deploy_states : ', deploy_states)
                
                # Orchestration
                node_choice, service_scaling_choice, exp = orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue,deploy_states_float, cpu_lists, mem_lists, task_lists, graph_cnn_list, MAX_TESK_TYPE)
                a=B
                logger.info('Orchestration of Decision done ')
                # Randomising Orchestration
                
                #if random.uniform(0, 1)< 0.05:
                #    service_scaling_choice = torch.randint(-12, 12, (3,))
                #    node_choice = torch.randint(0, 6, (3,))

                
                
                # Here is the code for orchestration and service scaling
                execute_orchestration(node_choice, service_scaling_choice, #num_edge_nodes_per_eAP,
                         deploy_states, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list)
                
                logger.info('Execution orchestration')
                # Save data
                if slot > 50 * CHO_CYCLE:
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
            print('curr_task : ', curr_task)
            #print('Deploy state : ', len(deploy_state))    
            #print('Deploy state : ', deploy_state)  
            ava_node = []
            #a=b
            for i in range(len(curr_task)):
                #print('len(curr_task) : ', len(curr_task))
                if randomize ==False:
                    tmp_list = [action_dims[i] -1]  # Cloud is always available
                else:
                    
                    tmp_list = [action_dims[i] -1]
                    print('Randomised tmp_list as: ', tmp_list)
                print('length deploy state : ', len(deploy_states))
                for ii in range(len(deploy_states)):
                    #print('ii, i : ', ii, i)
                    #print('deploy_state[ii][curr_task[i][0]] : ', deploy_state[ii][curr_task[i][0]])
                    if deploy_states[i][ii][curr_task[i][0]] == 1:
                        print('ii : ', ii)
                        tmp_list.append(ii)
                ava_node.append(tmp_list)
            
            print('deploy_state befores : ', deploy_states)    
            print('ava_node : ', ava_node)
            #a=b 
            state_list = get_state_list(master_list, max_tasks)    

            last_length, length_list = get_last_length(master_list)
            s_grid = []
            for i, state in enumerate((state_list)):
                #sub_deploy_state = deploy_state[length_list[i]:length_list[i+1]]
                sub_deploy_state = deploy_states[i]
                print('sub_deploy_state : ', sub_deploy_state)
                sub_elem = flatten(flatten([sub_deploy_state, [[state[5]]], [[state[4]]], [[state[3]]],[state[2]], state[0], state[1], [[latency]], [[len(master_list[i].node_list)]]]))
                print('len(sub_elem) : ', len(sub_elem))
                s_grid.append(sub_elem)
            critic_state = flatten(s_grid)
            #a=b
            critic_state.append(len(cloud.task_queue))
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
                
                act_, valid_action_prob_mat_, policy_state_, action_choosen_mat_, \
                curr_state_value_, curr_neighbor_mask_, next_state_ids_ = q_estimator_list[i].action(np.array(s_grid[i]), critic, critic_state, ava_node[i], context,)
                
                print('ava_node[i], action_dim, act_[0] : ', ava_node[i], action_dims[i], act_[0])
                
                act.append(act_[0])
                valid_action_prob_mat.append(valid_action_prob_mat_[0])
                policy_state.append(policy_state_[0])
                print('action_choosen_mat_ : ', action_choosen_mat_)
                print('')
                print('')
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
            ####
            # Put the current task on the queue based on dispatch decision
            print('Action Dims : ', action_dims)
            #put_current_task_on_queue(act, curr_task, cluster_action_value, cloud, master_list)
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

            deploy_reward.append(sum(immediate_reward))

            if slot != 0:
                logger.debug('Computing targets for cMMAC')
                r_grid = to_grid_rewards(immediate_reward)
                for m in range(len(r_grid)):
                    print('')
                    print('')
                    print('')
                    print('[m] : ', [m])
                    '''
                    try:
                        
                        print('issue less')
                        print('action_choosen_mat_prev : ', action_choosen_mat_prev)
                        print('action_choosen_mat_prev[[m],:] loop : ', action_choosen_mat_prev[m,:])
                        
                    except:
                        print('')
                        print('')
                        print('')
                        print('With issues')
                        #print('curr_neighbor_mask_prev[m] outer loop : ', curr_neighbor_mask_prev[m])
                        #print('curr_neighbor_mask_prev outer loop : ', curr_neighbor_mask_prev)
                        
                        #print('r_grid[[m],:]outer loop : ', r_grid[[m],:])
                        #print('r_grid[[m],:] outer loop : ', r_grid[[m],:])
                        print('action_choosen_mat_prev : ', action_choosen_mat_prev)
                        print('action_choosen_mat_prev[[m],:]  : ', action_choosen_mat_prev[[m],:])
                        a=b
                        
                    ''' 
                    print('curr_neighbor_mask_prev outer loop : ', curr_neighbor_mask_prev)
                    print('r_grid[[m]] : ', r_grid[[m]])   
                    print('action_choosen_mat_prev : ', action_choosen_mat_prev, len(action_choosen_mat_prev))
                    print('action_choosen_mat_prev[[m]]  : ', action_choosen_mat_prev[[m]])
                    #print('action_choosen_mat_prev[[m],:]  : ', action_choosen_mat_prev[[m],:])
                    #targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m],:], np.array(critic_state), critic, r_grid[[m],:], curr_neighbor_mask_prev[[m],:], gamma)
                    targets_batch = q_estimator_list[m].compute_targets(action_mat_prev[[m]], np.array(critic_state), critic, r_grid[[m]], curr_neighbor_mask_prev[m], gamma)
                    # Advantage for policy network.
                    advantage = q_estimator_list[m].compute_advantage([curr_state_value_prev[m]], [next_state_ids_prev[m]] ,
                                                            np.array(critic_state), critic, r_grid[[m],:], gamma)
                                      
                    if curr_task[0][0] != -1 and curr_task[1][0] != -1:
                        ReplayMemory_list[m].add(np.array([state_mat_prev]), action_mat_prev[[m]], targets_batch[[0]], np.array([s_grid[m]]))
                        policy_replay_list[m].add(policy_state_prev[[m]], action_choosen_mat_prev[[m]], advantage , curr_neighbor_mask_prev[[m]])
                        #ReplayMemory_list[m].add(np.array([state_mat_prev]), action_mat_prev[[m],:], targets_batch[[0],:], np.array([s_grid[m]]))
                        #policy_replay_list[m].add(policy_state_prev[[m],:], action_choosen_mat_prev[[m],:], advantage , curr_neighbor_mask_prev[[m],:])
            #a=B
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

        sum_rewards.append(float(sum(all_rewards)) / float(len(all_rewards)))
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
    name = 'full_randomisation_orchestration_no_randomisation_'
    time_str = str(time.time())
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(record, f)
                
    with gzip.open("./result/torch_out_time_" + name + time_str + ".obj", 'rb') as fp:
        record = pickle.load(fp)

    with gzip.open("./result/throughput_" + name + time_str + ".obj", "wb") as f:
        pickle.dump(throughput_list, f)
    
    title =     "Eaps_" + str(len(all_task_list)) + '_nodes in cluster_'+ str(nodes_in_cluster) 
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
    RUN_TIMES = 2 #500 # Number of Episodes to run
    TASK_NUM = 5000 # 5000 Time for each Episode Ending
    TRAIN_TIMES = 50 # Training Iterations for policy and value networks (Actor , Critic)
    CHO_CYCLE = 1000 # Orchestration cycle

    ##############################################################
    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE)