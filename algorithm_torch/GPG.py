# GNN-based Learning for Service Orchestration
from math import exp
import statistics
import numpy as np
import bisect
from algorithm_torch.Orchestration_Agent import *
#from algorithm_torch.policyReplayMemory import policyReplayMemory
from algorithm_torch.helpers_main_pytorch import remove_docker_from_master_node, deploy_new_docker, entropy_weight_min, entropy_weight_decay
from algorithm_torch.losses import act_loss
import matplotlib.pyplot as plt

def get_gpg_reward(master_list):
    task_len = 0
    for master in master_list:
        for node in master.node_list:
            task_len += len(node.task_queue)
    reward = np.exp(-(task_len))
    #print('task_len : ', task_len)
    #print('orchestration_reward : ', reward)   
    return reward

def discount(x, gamma):
    """Calculate the discounted cumulative reward
        Cumulative Reward = r_t + gamma * r_t+1 + gamma ^2 * r_t+2 + ________

    Args:
        x (Numpy array): numpy array of rewards over time
        gamma (float): Discount factor

    Returns:
        numpy array: Calculated cumulative discounted reward
    """
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out

def orchestrate_decision(orchestrate_agent, exp, done_tasks,undone_tasks, curr_tasks_in_queue, deploy_state_float, cpu_lists, mem_lists, task_lists, gcnn_list,  MAX_TESK_TYPE, epsilon_exploration):
    """Generate Orchestration Decision

    Args:
        orchestrate_agent ([Orchestration Network Object]): [Orchestration Network]
        exp ([list]): list of recorded experiences (dictionary)
        done_tasks (list): List of done tasks
        undone_tasks: (list): List of undone tasks
        curr_tasks_in_queue (list): List of tasks currently in queue
        deploy_state_float(list of lists): List containing the tasks running on all of the Nodes  
        MAX_TESK_TYPE (int) : Maximum number of task types

    Returns:
        change_node (list) :  Nodes to be changed
        change_service (list) : Services to be changed
        exp (list): updated recorded experiences
    """
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state_float, cpu_lists, mem_lists, task_lists, gcnn_list]
    # Invokes model (propagate the observation through the orchestration model) 
    #and return the chosen node, chosen services and the appended experience, after the orchestration step]
    # Propagate the observation of the environment and produces
    node_act, scale_act, _, _, node_inputs, scale_inputs = \
        orchestrate_agent.invoke_model(obs, epsilon_exploration)
    node_choice = [x for x in node_act[0]]# nodes chosen for deployment
    service_scaling_choice = [] # Server choice here is chosen services
    
    for x in scale_act[0][0]: 
        if x >= MAX_TESK_TYPE:
            service_scaling_choice.append(x - MAX_TESK_TYPE - 1)
        else:
            service_scaling_choice.append(x - MAX_TESK_TYPE)
    
    # Both of them are always one just used to allow matrix multiplication
    # Store experience
    exp['node_inputs'].append(node_inputs)
    exp['scale_inputs'].append(scale_inputs)
    return node_choice, service_scaling_choice, exp
    
def plot(x, y, title):
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(x,y)
    plt.title(title)
    
    plt.savefig(title + '.png')
    plt.show()

def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    """Generate a piecewise linear fit for the given reward function along with time
        this is done to generate Q Value targets so that advantage i.e Q_Value_target - Q_Value_Predicted can be calculated
        
    Args:
        all_cum_rewards ([list of floats]): [All Cumulative Rewards] (oldest reward the first)
        all_wall_time ([list of floats]): [Time]

    Returns:
        [baselines]: [returns a list of piecewise linear data extrapolation]
    """
    print('all_wall_time: ', all_wall_time)
    print('all_cum_rewards : ', all_cum_rewards)
    
    assert len(all_cum_rewards) == len(all_wall_time)
    
    # All time
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    # Find baseline value
    baseline_values = {}
    #baseline_values_as_is = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            # Locate the insertion point for t in all_wall_time[i] to maintain sorted order. 
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline #/ float(len(all_wall_time[0]))
        
        #baseline_values_as_is[t] =  baseline 
    #if len(all_wall_time[0])>3:    
    #    print('baseline_values : ', baseline_values)
        #print('baseline_values_as_is : ', baseline_values_as_is)
        #a=b
    # Output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)
        
        
    #if len(baselines[0])>1:
    #    plot(all_wall_time[0], all_cum_rewards[0],  'all_cum_rewards_without_baseline_' + str(len(all_cum_rewards[0])))
    #    plot(all_wall_time[0], baselines[0], 'baselines_with_baseline_'+ str(len(baselines[0])) )
    return baselines



def compute_orchestrate_loss(orchestrate_agent, exp, batch_adv):
    """[Computation of orchestration loss for given experience (for one orchestration cycle)]

    Args:
        orchestrate_agent ([Orchestrate Agent Class]): [Orchestrate Agent]
        exp ([dictionary]): [Experience]
        batch_adv ([numpy array]): [difference between qvalue target and q value predicted]

    Returns:
        [Tensor]: [Computed Loss]
    """
    batch_adv = np.array(batch_adv)

    node_inputs = exp['node_inputs']
    scale_inputs = exp['scale_inputs']

    adv = batch_adv

    # Convert to numpy array
    node_inputs = np.array(node_inputs)
    scale_inputs = np.array(scale_inputs)
    
    #new 
    orchestrate_agent.node_inputs = np.asarray(node_inputs)
    orchestrate_agent.scale_inputs = np.asarray(scale_inputs)
    adv = torch.tensor(adv)
    orchestrate_agent.gcn(node_inputs)
    
    # Map gcn_outputs and raw_inputs to action probabilities
    orchestrate_agent.node_act_probs, orchestrate_agent.scale_act_probs = orchestrate_agent.ocn_net((node_inputs, scale_inputs, orchestrate_agent.gcn.outputs) )#
    #
    orchestrate_agent.adv_loss, orchestrate_agent.entropy_loss, orchestrate_agent.act_loss_ = act_loss(orchestrate_agent.scale_act_probs, orchestrate_agent.node_act_probs, 
                                                                                                       orchestrate_agent.eps, adv, orchestrate_agent.entropy_weight)
    #old
    #loss = orchestrate_agent.act_loss(
    #    node_inputs, scale_inputs, adv)

    return orchestrate_agent.act_loss_


def decrease_var(var, min_var, decay_rate):
    """Function to decrease Variable generally entropy

    Args:
        var ([float]): [Variable]
        min_var ([float]): [min_allowed value of variable]
        decay_rate ([float]): [Decay rate for the variable]

    Returns:
        [var]: [Variable with reduced value]
    """

    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def train_orchestrate_agent(orchestrate_agent, exp, cum_reward_across_episodes =[0]):
    
    """[Train the orchestration agent]

    Args:
        orchestrate_agent ([Orchestrate Agent Class]): [Orchestrate Agent]
        exp ([dictionary]): [Experience]
        entropy_weight ([float]): [Entropy weight]
        entropy_weight_min ([float]): [Minimum Entropy Weight]
        entropy_weight_decay ([type]): [Entropy Weight Decay rate]

    Returns:
        [Tensors]: [Entropy weight and calculated loss]
    """
    all_cum_reward = []
    all_rewards = exp['reward']
    batch_time = exp['wall_time']

    rewards = np.array([r for (r, t) in zip(all_rewards, batch_time)])
    cum_reward = discount(rewards, 1)
    
    all_cum_reward.append(cum_reward)
    #cum_reward_across_episodes=[x.append('a') or x for x in a]
    #cum_reward_across_episodes.append(cum_reward)
    #for elem in cum_reward:
    #print('cum_reward_across_episodes : ', cum_reward_across_episodes)
    #orchestrate_agent.entropy_weight = entropy_weight
    
    # Compute baseline
    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [batch_time])
    average_baseline = statistics.mean(cum_reward_across_episodes)
    # Calculate the advantage
    #batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = all_cum_reward[0] - average_baseline
    
    print('average_baseline : ', average_baseline)
    #print('all_cum_reward[0]: ', all_cum_reward[0] )
    #print('baselines[0]: ', baselines[0] )
    print('batch_adv : ', batch_adv)
    '''
    if len(batch_time)>1:
        for i in range(len(all_cum_reward)):
            print(f'At {i} point of time'.format(i))
            print('all_cum_reward[0]: ', all_cum_reward[i] )
            print('baselines[0]: ', baselines[i] )
            print('batch_adv : ', batch_adv[i])
            print('batch_time : ', batch_time[i])
    '''       
    
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
    
    
    #orchestrate_agent.entropy_weight = entropy_weight
    
    # Actual training of Orchestrate Net
    orchestrate_agent.optimizer.zero_grad()
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv)
    loss.backward()
    orchestrate_agent.optimizer.step()
    
    orchestrate_agent.entropy_weight = decrease_var(orchestrate_agent.entropy_weight, entropy_weight_min, entropy_weight_decay)
    #print('orchestrate_agent.entropy_weight : ', orchestrate_agent.entropy_weight)
    return orchestrate_agent.entropy_weight, loss, all_cum_reward
    '''
        def get_orchestration_reward(master_list, cur_time, check_queue):
            reward = []
            for mstr in master_list:
                for i, node in enumerate(mstr.node_list):
                    _, undone, _ = check_queue(node.task_queue, cur_time, len(master_list))
                    reward.append(len(undone))
            orchestration_reward = exp(-sum(reward))
            
            print('orchestration_reward : ', orchestration_reward)
            return orchestration_reward
    '''
def execute_orchestration(change_node, change_service,deploy_state, service_coefficient, POD_MEM, POD_CPU, cur_time, master_list):
    """Execute the orchestrated actions

    Args:
        vaild_node (int) : Number of valid nodes for execution of tasks
        MAX_TESK_TYPE (int) : Maximum number of task types
        deploy_state(list of lists): List containing the tasks running on all of the Nodes  
        service_coefficient(list): Service Coefficients
        POD_MEM: (float): Memory of POD
        POD_CPU (float): CPU of POD
        cur_time (float) : Time
        master_list (list) :  List of created Master Nodes

    """

    length_list = [0]
    last_length = length_list[0]
    for mstr in master_list:
        length_list.append(last_length + len(mstr.node_list))
        last_length = last_length + len(mstr.node_list)
    # Execute orchestration
    for i in range(len(change_node)):
        if change_service[i] < 0:
            # Delete docker and free memory
            service_index = -1 * change_service[i] - 1
            
            for iter in range(len(length_list)-1):
                if change_node[i] < length_list[iter+1] and change_node[i] >= length_list[iter]:
                    node_index = change_node[i] - length_list[iter]
                    remove_docker_from_master_node(master_list[iter], node_index, service_index, deploy_state[iter])
        else:
            # Add docker and tack up memory
            service_index = change_service[i] - 1
            
            for iter in range(len(length_list)-1):
                if change_node[i] < length_list[iter+1] and change_node[i] >= length_list[iter]:
                    node_index = change_node[i] - length_list[iter]
                    deploy_new_docker(master_list[iter], POD_MEM, POD_CPU, cur_time, node_index, service_coefficient, service_index, deploy_state[iter])