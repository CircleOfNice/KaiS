import sys
from typing import Callable, Tuple, Type
sys.path.append("..") 
import numpy as np
import torch.nn as nn
import torch
from algorithm_torch.helpers_main_pytorch import device
from algorithm_torch.cMMAC_model import createmodel
import torch.optim as optim
from algorithm_torch.helpers_main_pytorch import set_lr
from algorithm_torch.losses import simple_square_loss

class Value_Model(nn.Module):
    """Class for defining the value model (Critic part) of the Actor critic Model
    """
    def __init__(self, state_dim:int, inp_sizes: Tuple = [128, 64, 32], act:torch.nn = nn.Softsign(), loss:Callable = simple_square_loss):
        """Initialisation arguments for class

        Args:
            state_dim (int): Dimensions of the input state
            inp_sizes (list, optional): Dimensions for hidden state. Defaults to [128, 64, 32].
            act (Pytorch Activation layer type, optional): Desired Activation function for all layers. Defaults to nn.ReLU().
        """
        super().__init__()
        self.inp_sizes = inp_sizes
        self.state_dim = state_dim
        self.act  = act
        self.model = createmodel(self.inp_sizes, self.state_dim, 1, self.act)
        self.vm_criterion = loss
        
    def forward(self, x:np.array)->torch.Tensor:
        x = self.model(x.float())
        return x 
    
def build_value_model(state_dim:int, inp_sizes, loss:Callable= simple_square_loss)-> Tuple[Type[Value_Model], Callable]:
    """[Method to build the value model and assign its loss and optimizers]
    """
    vm = Value_Model(state_dim, inp_sizes = inp_sizes, loss = loss)
    vm_optimizer = optim.Adam(vm.parameters(), lr=0.001)
    return vm, vm_optimizer
    
    
def update_value( s:np.array, y:np.array, learning_rate:float, vm:Type[Value_Model], vm_optimizer:torch.optim)->torch.Tensor:
    """[Method to optimize the Value net]

    Args:
        s ([Numpy array]): [state]
        y ([Numpy array]): [target]
        learning_rate ([float]): [learning rate]
    """
    vm.to(device)
    s = torch.from_numpy(s)
    s = s.to(device)
    vm_optimizer.zero_grad()
    value_output = vm(s.float())
    y = torch.tensor(y)
    value_output, y = value_output.to(device), y.to(device)
    loss = vm.vm_criterion(y, value_output)
    
    set_lr(vm_optimizer, learning_rate)
    loss.backward()
    vm_optimizer.step()
    vm.to("cpu")
    value_output, y = value_output.to("cpu"), y.to("cpu")
    return loss
    
