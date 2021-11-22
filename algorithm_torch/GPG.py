# GNN-based Learning for Service Orchestration
from math import exp
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import bisect
from algorithm_torch.gcn import GraphCNN
from algorithm_torch.Orchestration_Agent import *
from algorithm_torch.ReplayMemory import ReplayMemory
from algorithm_torch.policyReplayMemory import policyReplayMemory

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



def invoke_model(orchestrate_agent, obs, exp, MAX_TESK_TYPE):
    """[Invoke model (propagate the observation through the orchestration model) 
    and return the chosen node, chosen services and the appended experience, after the orchestration step]

    Args:
        orchestrate_agent ([OrchestrateAgent Type]): [Instance of Orchestrate Agent]
        obs ([list]): [Observations containing done tasks, undone tasks, current tasks in queue, deploy state]
        exp ([dict]): [Experience]

    Returns:
        [list, list , dictionary]: [chosen node, chosen services and the appended experience]
    """
    
    # Propagate the observation of the environment and produces
    node_act, cluster_act, node_act_probs, cluster_act_probs, node_inputs, cluster_inputs = \
        orchestrate_agent.invoke_model(obs)
    node_choice = [x for x in node_act[0]]# nodes chosen for deployment
    server_choice = [] # Server choice here is chosen services

    for x in cluster_act[0][0]: 
        if x >= MAX_TESK_TYPE:
            server_choice.append(x - MAX_TESK_TYPE - 1)
        else:
            server_choice.append(x - MAX_TESK_TYPE)
    
    # For storing node index        
    node_act_vec = np.ones(node_act_probs.shape)
    # For storing cluster index
    cluster_act_vec = np.ones(cluster_act_probs.shape)
    # Both of them are always one just used to allow matrix multiplication
    # Store experience
    exp['node_inputs'].append(node_inputs)
    exp['cluster_inputs'].append(cluster_inputs)
    exp['node_act_vec'].append(node_act_vec)
    exp['cluster_act_vec'].append(cluster_act_vec)
    return node_choice, server_choice, exp
    


def act_offload_agent(orchestrate_agent, exp, done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state, MAX_TESK_TYPE):
    """Chooses action using the invocation (propagation through) of Orchestrate Agent model
    Args:
        orchestrate_agent ([OrchestrateAgent Type]): [Instance of Orchestrate Agent]
        exp ([dictionary]): [Experience dictionary]
        done_tasks ([list]): [list of done tasks]
        undone_tasks ([list]): [lists of undone(not done) tasks]
        curr_tasks_in_queue ([list]): [list of tasks in queue]
        deploy_state ([list]): [List of lists containing the deployment of nodes]
        MAX_TESK_TYPE([int]): Maximum number of tasks in queue
    Returns:
        [node, use_exec, exp]: [chosen node, chosen service and the appended experience]
    """
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state]
    # Invokes model (propagate the observation through the orchestration model) 
    #and return the chosen node, chosen services and the appended experience, after the orchestration step]
    node, use_exec, exp = invoke_model(orchestrate_agent, obs, exp, MAX_TESK_TYPE)
    return node, use_exec, exp


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    """Generate a piecewise linear fit for the given reward function along with time
        this is done to generate Q Value targets so that advantage i.e Q_Value_target - Q_Value_Predicted can be calculated
        
    Args:
        all_cum_rewards ([list of floats]): [All Cumulative Rewards] (oldest reward the first)
        all_wall_time ([list of floats]): [Time]

    Returns:
        [baselines]: [returns a list of piecewise linear data extrapolation]
    """

    assert len(all_cum_rewards) == len(all_wall_time)
    # All time
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    # Find baseline value
    baseline_values = {}
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

        baseline_values[t] = baseline / float(len(all_wall_time))
    # Output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)
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
    loss = 0
    batch_adv = np.array(batch_adv)

    node_inputs = exp['node_inputs']
    cluster_inputs = exp['cluster_inputs']
    node_act_vec = exp['node_act_vec']
    cluster_act_vec = exp['cluster_act_vec']
    adv = batch_adv

    # Convert to numpy array
    node_inputs = np.array(node_inputs)
    cluster_inputs = np.array(cluster_inputs)
    node_act_vec = np.array(node_act_vec)
    cluster_act_vec = np.array(cluster_act_vec)
    
    loss = orchestrate_agent.act_loss(
        node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv)

    return loss


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


def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    
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
    orchestrate_agent.entropy_weight = entropy_weight
    
    # Compute baseline
    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [batch_time])
    # Calculate the advantage
    batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
    orchestrate_agent.entropy_weight = entropy_weight
    
    # Actual training of Orchestrate Net
    orchestrate_agent.optimizer.zero_grad()
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv)
    loss.backward()
    orchestrate_agent.optimizer.step()
    entropy_weight = decrease_var(entropy_weight,
                                  entropy_weight_min, entropy_weight_decay)
    return entropy_weight, loss
 
def get_orchestration_reward(master_list, cur_time, check_queue):
    reward = []
    for mstr in master_list:
        for i, node in enumerate(mstr.node_list):
            _, undone, undone_kind = check_queue(node.task_queue, cur_time, len(master_list))
            reward.append(len(undone))
    orchestration_reward = exp(-sum(reward))
    return orchestration_reward