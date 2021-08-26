import torch.nn as nn

def fc(inp_dim, output_dim, act=nn.ReLU()):
    """Function to define a fully connected block

    Args:
        inp_dim (int): Input dimension of the layer
        output_dim ([int]): Output dimension of the layer
        act ([Pytorch Activation layer type], optional): [Desired Activation Layer for the FC Unit]. Defaults to nn.ReLU().

    Returns:
        Sequential Model: Fully connected layer block
    """
    linear = nn.Linear(inp_dim, output_dim)
    nn.init.xavier_uniform_(linear.weight)
    linear.bias.data.fill_(0)
    fc_out = nn.Sequential(linear, act)
    return fc_out 