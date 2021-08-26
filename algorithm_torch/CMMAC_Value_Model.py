import torch.nn as nn
import torch
from algorithm_torch.CMMAC_fc_layer import fc
class Value_Model(nn.Module):
    """Class for defining the value model (Critic part) of the Actor critic Model
    """
    def __init__(self, state_dim, inp_sizes = [128, 64, 32], act = nn.ReLU()):
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

    def forward(self, x):
        x = torch.from_numpy(x)
        x = self.fc1(x.float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x