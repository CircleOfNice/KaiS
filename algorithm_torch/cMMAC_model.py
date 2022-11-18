from typing import Callable, Tuple
import torch.nn as nn

from algorithm_torch.CMMAC_fc_layer import fc
def createmodel(inp_sizes: list, state_dim:int,action_dim:int, act:nn)->nn.Sequential:
    """Create Model: 

    Args:
        inp_sizes (int) : Number of Master Node data required  (Number of task lists available)
        state_dim (int): Number of Edge Node data required 
        action_dim (int): 
        act :
    Returns: 
        s_grid_len : list of size of state for corresponding all_task_list
    """
    list_model = []
    for i, inp_size in enumerate(inp_sizes):
        if i == 0:
            list_model.append(fc(state_dim, inp_size, act=act))
        else:
            list_model.append(fc(inp_sizes[i-1], inp_size, act=act))

    list_model.append(fc(inp_size, action_dim, act=act))
        
    model = nn.Sequential(*list_model)
    return model