import numpy as np
import torch.nn.functional as F
import torch
import time

from tensorboardX import SummaryWriter
from typing import Dict, Optional


def expected_action_distribution(node_num:int, use_mask:bool=True) -> np.array:
    """Method to calculate the expected action distribution for a random scheduler

    If we dont use the mask, we expected a uniform distribution across nodes

    If we do use a mask, we expected a distribution where the first nodes get more tasks then the later ones.
    This is with the expectation that we use the ordered_valid_action_mask() in the environment

    e.g. for 4 nodes we expected following distribution:

    node 1: 40%
    node 2: 30%
    node 3: 20%
    node 4: 10%

    Args:
        node_num (int): Total number of nodes
        use_mask (bool, optional): If we use the action mask. Defaults to True.

    Returns:
        np.array: Returns the expected distribution
    """
    if use_mask:
        frac = np.sum(list(range(1, node_num+1)))
        dist = np.array(list(range(1, node_num+1))[::-1]) / frac
    else:
        dist = np.ones(node_num) / node_num

    return dist


def action_distribution_kl_div(input_dist:np.array, target_dist:np.array) -> float:
    """Helper method to calculate the kullback leiber divergence between two distributions as numpy arrays

    Args:
        input_dist (np.array): Input distribution
        target_dist (np.array): Target distribution

    Returns:
        float: Kullback Leiber Divergence
    """
    input_dist  = torch.Tensor(input_dist / sum(input_dist))
    target_dist = torch.Tensor(target_dist)
    # May be better to not take log softmax as it standardises our distribution
    div = F.kl_div(input_dist, target_dist, reduction="batchmean", log_target=False)
    return div

