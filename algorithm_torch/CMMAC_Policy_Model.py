import torch.nn as nn
import torch
from torch import from_numpy, log
from algorithm_torch.CMMAC_fc_layer import fc
from torch.nn.functional import softmax
from algorithm_torch.losses import policy_net_loss
import torch.optim as optim
#from algorithm_torch.helpers_main_pytorch import set_lr

class Policy_Model(nn.Module):
    """Class for defining Policy model (Actor part) of the Actor Critic Model
    """
    def __init__(self, state_dim, action_dim, inp_sizes = [128, 64, 32], act = nn.ReLU(), loss = policy_net_loss):
        """Initialisation arguments for class

        Args:
            state_dim (int): Dimensions of the input state
            action_dim (int): Dimensions of the output (actions)
            inp_sizes (list, optional): Dimensions for hidden state. Defaults to [128, 64, 32].
            act (Pytorch Activation layer type, optional): Desired Activation function for all layers. Defaults to nn.ReLU().
        """
        super().__init__()
        self.policy_state = state_dim
        self.action_dim = action_dim
        self.fc1 = fc(self.policy_state, inp_sizes[0], act=act)
        self.fc2 = fc(inp_sizes[0], inp_sizes[1], act=act)
        self.fc3 = fc(inp_sizes[1], inp_sizes[2], act=act)
        self.fc4 = fc(inp_sizes[2], self.action_dim, act=act)
        self.pm_criterion = loss
    def forward(self, x):
        x = torch.from_numpy(x)
        
        x = self.fc1(x.float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x) 
        return x 
    
    
def build_policy_model(state_dim, action_dim, inp_sizes = [128, 64, 32], act = nn.ReLU(), loss= policy_net_loss):
    """[Method to build the Policy model and assign its loss and optimizers]
    """
    pm = Policy_Model(state_dim, action_dim, inp_sizes, act, loss)
    pm_optimizer = optim.Adam(pm.parameters(), lr=0.001)
    return pm, pm_optimizer

def sm_prob( policy_net_output, neighbor_mask):
    """[Method to apply policy filtering of edge nodes using neighbor mask and policy output]

    Args:
        policy_net_output ([Pytorch Tensor]): [Output of Policy Network]
        neighbor_mask ([Numpy array]): [Mask to determine available network]

    Returns:
        [Pytorch Tensors]: [Torch Tensor respectively containing softmax probabilities, logits and valid_logits]
    """
    neighbor_mask = from_numpy(neighbor_mask)
    
    logits = policy_net_output +1 
    valid_logits = logits * neighbor_mask
    softmaxprob = softmax(log(valid_logits + 1e-8))
    return softmaxprob, logits, valid_logits
    
def update_policy(q_estim, policy_state, advantage, action_choosen_mat, curr_neighbor_mask):
    """[Optimize Policy net]

    Args:
        policy_state ([Numpy Array]): [State Policy]
        advantage ([Numpy Array]): [Calcualted Advantage]
        action_choosen_mat ([Numpy Array]): [Choose action matrix]
        curr_neighbor_mask ([Numpy Array]): [Current Neighbor mask]
        learning_rate ([float]): [Learning Rate]
    """

    q_estim.pm_optimizer.zero_grad()
    policy_net_output = q_estim.pm(policy_state)
    
    softmaxprob, logits, valid_logits = sm_prob(policy_net_output, curr_neighbor_mask)
    action_choosen_mat = torch.tensor(action_choosen_mat)
    adv = torch.tensor(advantage)

    loss = q_estim.pm.pm_criterion(softmaxprob, adv, action_choosen_mat , q_estim.entropy)
    loss.backward()
    q_estim.pm_optimizer.step()
    return loss