import sys
sys.path.append("..") 

import torch.nn as nn
import torch

from algorithm_torch.CMMAC_fc_layer import fc
import torch.optim as optim
from algorithm_torch.helpers_main_pytorch import set_lr
from algorithm_torch.losses import simple_square_loss

class Value_Model(nn.Module):
    """Class for defining the value model (Critic part) of the Actor critic Model
    """
    def __init__(self, state_dim, inp_sizes = [128, 64, 32], act = nn.Softsign(), loss = simple_square_loss):
        """Initialisation arguments for class

        Args:
            state_dim (int): Dimensions of the input state
            inp_sizes (list, optional): Dimensions for hidden state. Defaults to [128, 64, 32].
            act (Pytorch Activation layer type, optional): Desired Activation function for all layers. Defaults to nn.ReLU().
        """
        super().__init__()
        self.fc1 = fc(state_dim, inp_sizes[0], act=act)
        self.fc2 = fc(inp_sizes[0], inp_sizes[1], act=act)
        self.fc3 = fc(inp_sizes[1], inp_sizes[2], act=act)
        self.fc4 = fc(inp_sizes[2], 1, act=act)
        self.vm_criterion = loss
        

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.fc1(x.float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
def build_value_model(state_dim, loss = simple_square_loss):
    """[Method to build the value model and assign its loss and optimizers]
    """
    vm = Value_Model(state_dim, inp_sizes = [128, 64, 32], loss = loss)
    vm_optimizer = optim.Adam(vm.parameters(), lr=0.001)
    return vm, vm_optimizer
    
    
def update_value( s, y, learning_rate, vm, vm_optimizer):
    """[Method to optimize the Value net]

    Args:
        s ([Numpy array]): [state]
        y ([Numpy array]): [target]
        learning_rate ([float]): [learning rate]
    """
    vm_optimizer.zero_grad()
    value_output = vm(s)
    y = torch.tensor(y)
    loss = vm.vm_criterion(y, value_output)
    
    set_lr(vm_optimizer, learning_rate)
    loss.backward()
    vm_optimizer.step()
    return loss
    
