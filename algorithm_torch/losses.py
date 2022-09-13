
from typing import Tuple
from torch import Tensor, multiply, sum, mean
from torch.nn import SmoothL1Loss, MSELoss
import torch
import numpy as np
from torch.nn.functional import log_softmax
# Notes on ReplayMemory_list and policy_replay_list

# ReplayMemory_list - (previous state matrix, previous action matrix, targets, state_grid)
# policy_replay_list - (previous policy state matrix, previous action choosen matrix, advantage, previous curr_neighbor_mask)

def calculate_log_prob(act_probs:torch.Tensor, eps:float)->torch.Tensor:
        return torch.log(act_probs + eps)
def mul_act_probs(act_probs:torch.Tensor, adv_or_torch_log:torch.Tensor)->torch.Tensor:
    return -torch.mul(act_probs, adv_or_torch_log)
def sum_normalise( torch_mul_log_prob:torch.Tensor)->torch.Tensor:
    return torch.sum(torch_mul_log_prob)/torch_mul_log_prob.shape[0]

def simple_policy_loss(log_prob: torch.Tensor, advantage:torch.Tensor)->torch.Tensor:
    return multiply(-log_prob, advantage)

def simple_value_loss_1(value:torch.Tensor, _return:torch.Tensor)->torch.Tensor:
    return SmoothL1Loss(value, _return)

def simple_square_loss(output:torch.Tensor,target:torch.Tensor)->torch.Tensor: # current value loss implementation
    val_loss = sum((target - output)**2)
    return val_loss

def _policy_net_loss(logsoftmaxprob:torch.Tensor, tfadv:torch.Tensor, ACTION:np.array)->torch.Tensor:
    log_prob = logsoftmaxprob * ACTION
    _policy_loss = simple_policy_loss(log_prob, tfadv)
    _policy_loss_sum = sum(_policy_loss, axis=1)
    mean_policy_loss_sum = mean(_policy_loss_sum) 
    return mean_policy_loss_sum# changing it

def _entropy_loss(softmaxprob: torch.Tensor, logsoftmaxprob:torch.Tensor)->torch.Tensor:
    raw_softmaxprob_logsoftmaxprob = softmaxprob * logsoftmaxprob
    return - mean(raw_softmaxprob_logsoftmaxprob)

def policy_net_loss(softmaxprob:torch.Tensor, tfadv:torch.Tensor, ACTION:np.array, entropy:float)->torch.Tensor:#TODO entropy here is redundant
    """[Method to calculate policy net loss]

    Args:
        softmaxprob ([Pytorch Tensor]): [Output of Policy Net]
        tfadv ([Numpy Array]): [difference between estmated Q value and the target Q value]
        ACTION ([Numpy Array]): [Actions for the given batch]

    Returns:
        [Pytorch Tensor]: [Policy loss]
    """
    logsoftmaxprob  = log_softmax(softmaxprob)
    actor_loss = _policy_net_loss(logsoftmaxprob, tfadv, ACTION)
    entropy = _entropy_loss(softmaxprob, logsoftmaxprob)
    policy_loss = actor_loss - 0.01 *  entropy
    return policy_loss

def act_loss(scale_act_probs:torch.Tensor, node_act_probs:torch.Tensor, eps:float, adv:np.array, entropy_weight:float)->Tuple[float, float, float]:
    """Calculation of Orchestration Loss

    Args:
        scale_act_probs ([Numpy array]): [Scale Action Probability inputs]
        node_act_probs ([Numpy array]): [Node Action Probability inputs]
        eps: experience
        adv ([Numpy array]): [Calcualted advantage (difference between Q value predicted and Q value targets)]
        entropy_weight : entropy_weight
    Returns:
        [Tensor]: [Calculated Loss]
    """
    # Scale segment
    #adv_loss
    torch_log_scale = calculate_log_prob(scale_act_probs , eps)
    torch_log_adv_mul_scale = mul_act_probs(torch_log_scale, adv)
    scale_adv_loss = sum_normalise(torch_log_adv_mul_scale)
    
    # Entropy loss
    torch_log_scale_prob = calculate_log_prob(scale_act_probs, eps)
    torch_mul_log_scale_prob = mul_act_probs(torch_log_scale_prob, scale_act_probs)
    scale_entropy_loss = sum_normalise(torch_mul_log_scale_prob)
    total_scale_loss = scale_adv_loss + entropy_weight* scale_entropy_loss

    # Node Segment
    # Adv Loss

    torch_log_node = calculate_log_prob(node_act_probs , eps)
    torch_log_adv_mul_node = mul_act_probs(torch_log_node, adv)
    node_adv_loss = sum_normalise(torch_log_adv_mul_node)

    # Entropy loss
    torch_log_node_prob = calculate_log_prob(node_act_probs, eps)
    torch_mul_log_node_prob = mul_act_probs(torch_log_node_prob, node_act_probs)
    node_entropy_loss = sum_normalise(torch_mul_log_node_prob)
    
    total_node_loss = node_adv_loss + entropy_weight* node_entropy_loss
    act_loss_ = total_node_loss + total_scale_loss
    # Problem is 
    return scale_adv_loss+ node_adv_loss , scale_entropy_loss + node_entropy_loss, act_loss_


'''

def act_loss(scale_act_probs, node_act_probs, eps, adv, entropy_weight): #original
    print('node_act_probs , scale_act_probs : ', node_act_probs.shape, scale_act_probs.shape )
    adv_loss = torch.sum(torch.mul(torch.log(node_act_probs*scale_act_probs+ eps), -adv))
    entropy_loss =torch.sum(torch.mul(node_act_probs, torch.log(node_act_probs+ eps)))
    act_loss = adv_loss+ entropy_weight * entropy_loss
    return act_loss

'''