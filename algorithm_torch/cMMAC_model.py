from typing import Callable, Tuple
import torch.nn as nn

from algorithm_torch.CMMAC_fc_layer import fc
def createmodel(inp_sizes, policy_state,action_dim, act):
        list_model = []
        for i, inp_size in enumerate(inp_sizes):
            if i == 0:
                list_model.append(fc(policy_state, inp_size, act=act))
            else:
                list_model.append(fc(inp_sizes[i-1], inp_size, act=act))

        list_model.append(fc(inp_size, action_dim, act=act))
            
        model = nn.Sequential(*list_model)
        return model