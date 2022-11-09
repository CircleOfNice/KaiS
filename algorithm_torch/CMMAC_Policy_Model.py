from typing import Callable, Tuple
import torch.nn as nn
import torch
import numpy as np
from torch import from_numpy, log
from algorithm_torch.cMMAC_model import createmodel
from torch.nn.functional import softmax
from algorithm_torch.losses import policy_net_loss
import torch.optim as optim
from algorithm_torch.helpers_main_pytorch import device

class Policy_Model(nn.Module):
    """Class for defining Policy model (Actor part) of the Actor Critic Model
    """
    def __init__(self, state_dim:int, action_dim:int, inp_sizes: Tuple = [128, 64, 32], act:nn = nn.ReLU(), loss:Callable = policy_net_loss):
        """Initialisation arguments for class

        Args:
            state_dim (int): Dimensions of the input state
            action_dim (int): Dimensions of the output (actions)
            inp_sizes (list, optional): Dimensions for hidden state. Defaults to [128, 64, 32].
            act (Pytorch Activation layer type, optional): Desired Activation function for all layers. Defaults to nn.ReLU().
            loss (func): Loss function for the Policys model
        """
        super().__init__()
        
        self.policy_state = state_dim
        self.action_dim = action_dim
        self.inp_sizes = inp_sizes
        
        self.pm_criterion = loss
        self.act = act
        self.model = createmodel(self.inp_sizes, self.policy_state, self.action_dim, self.act)
        self.pm_criterion = loss

    def forward(self, x:np.array)->torch.Tensor:
        x = self.model(x.float())
        return x 
        
    
def build_policy_model(state_dim:int, action_dim:int, inp_sizes: Tuple = [128, 64, 32], act:nn = nn.ReLU(), loss:nn= policy_net_loss)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[Method to build the Policy model and assign its loss and optimizers]
    Args:
        state_dim: State Dimensions
        action_dim: Action Dimension
        inp_sizes: list of layer width of model
        act: Activation loss
        loss: Loss Function
    Returns:
        pm: Policy Model
        pm_optimizer:Policy Model optimizer
    """
    pm = Policy_Model(state_dim, action_dim, inp_sizes, act, loss)
    pm_optimizer = optim.Adam(pm.parameters(), lr=0.001)
    return pm, pm_optimizer

def sm_prob( policy_net_output:torch.Tensor, neighbor_mask:np.array)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[Method to apply policy filtering of edge nodes using neighbor mask and policy output]

    Args:
        policy_net_output ([Pytorch Tensor]): [Output of Policy Network]
        neighbor_mask ([Numpy array]): [Mask to determine available network]

    Returns:
        [Pytorch Tensors]: [Torch Tensor respectively containing softmax probabilities, logits and valid_logits]
    """
    neighbor_mask = from_numpy(neighbor_mask)
    neighbor_mask = neighbor_mask.to(device)
    
    logits = policy_net_output +1 
    valid_logits = logits * neighbor_mask
    softmaxprob = softmax(log(valid_logits + 1e-8))
    neighbor_mask = neighbor_mask.to('cpu')
    softmaxprob = softmaxprob.to('cpu')
    return softmaxprob, logits, valid_logits
    
def update_policy(q_estim:np.array, policy_state:np.array, advantage:np.array, action_choosen_mat:np.array, curr_neighbor_mask:np.array)-> torch.nn:
    """[Optimize Policy net]

    Args:
        policy_state ([Numpy Array]): [State Policy]
        advantage ([Numpy Array]): [Calcualted Advantage]
        action_choosen_mat ([Numpy Array]): [Choose action matrix]
        curr_neighbor_mask ([Numpy Array]): [Current Neighbor mask]
        learning_rate ([float]): [Learning Rate]
    """
    q_estim.pm.to(device)
    policy_state = torch.from_numpy(policy_state)
    policy_state = policy_state.to(device)
    q_estim.pm_optimizer.zero_grad()
    policy_net_output = q_estim.pm(policy_state)
    
    softmaxprob, logits, valid_logits = sm_prob(policy_net_output, curr_neighbor_mask)
    action_choosen_mat = torch.tensor(action_choosen_mat)
    adv = torch.tensor(advantage)
    loss = q_estim.pm.pm_criterion(softmaxprob, adv, action_choosen_mat , q_estim.entropy)
    loss.backward()
    q_estim.pm_optimizer.step()
    q_estim.pm.to("cpu")
    policy_state= policy_state.to("cpu")
    return loss