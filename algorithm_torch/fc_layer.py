import torch
import torch.nn as nn

def fc(inp_dim, output_dim, act=False):
    """Fully Connected Layer block

    Args:
        inp_dim (int): Input Dimension for the given layer
        output_dim (int): Input Dimension for the given layer
        act (Pytorch Activation Layer, optional): Pytorch Activation Layer. Defaults to nn.ReLU().

    Returns:
    
        Sequential Model: A sequential model which can be used as layer in Functional model
    """
    linear = nn.Linear(inp_dim, output_dim)
    if act:
        fc_out = nn.Sequential(linear, act)
    else:
        fc_out = nn.Sequential(linear)
        
    return fc_out 

def expand_act_on_state(state, sub_acts):
    """Function to concatenate states with sub_acts (Basically a list of values between 0.02 to 0.96) with multiple repetition over the tiling range (24)
    # Repeated state is appended with a vector of values havings different weightage across the last dimension likely to generate varied kind of action activations
    
    # You may not get this explanation as I only have it as an inference
    Args:
        state ([Pytorch Tensor]): [State Matrix]
        sub_acts ([list]): [list for tiling operation]

    Returns:
        [Pytorch Tensor]: [Concatenated State Tensor]
    """
    # Expand the state
    batch_size = state.shape[0]
    num_nodes = state.shape[1]
    num_features = state.shape[2]#.value
    
    expand_dim = len(sub_acts)

    # Replicate the state
    state = torch.tile(state, [1, 1, expand_dim])
    state = state.view([batch_size, num_nodes * expand_dim, num_features])

    # Prepare the appended actions
    sub_acts = torch.FloatTensor(sub_acts)
    sub_acts = sub_acts.view([1, 1, expand_dim])

    sub_acts = torch.tile(sub_acts, [1, 1, num_nodes])

    sub_acts = sub_acts.view([1, num_nodes * expand_dim, 1])
    sub_acts = torch.tile(sub_acts, [batch_size, 1, 1])
    # Concatenate expanded state with sub-action features
    concat_state = torch.cat([state, sub_acts], axis=2)

    return concat_state