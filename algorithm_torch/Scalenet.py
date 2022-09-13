import torch.nn as nn
from algorithm_torch.fc_layer import fc


class ScaleNet(nn.Module):
    """Service Scaling part of the orchestrate neural network
    """
    def __init__(self, expanded_state:int, scale_inp_sizes:list = [32, 16, 8 ,1], act:nn = nn.ReLU()):
        """Initialisation

        Args:
            expanded_state ([int]): [Input dimensions] # scale merged reshape input shape + executor levels shape 24
            scale_inp_sizes (list, optional): [Hidden Dimensions]. Defaults to [32, 16, 8 ,1].
            act ([Python Activation Layer], optional): [Python Activation Layer]. Defaults to nn.ReLU().
        """
        super().__init__()
        self.fc1 = fc(expanded_state, scale_inp_sizes[0], act=act)
        self.fc2 = fc(scale_inp_sizes[0], scale_inp_sizes[1], act=act)
        self.fc3 = fc(scale_inp_sizes[1], scale_inp_sizes[2], act=act)
        self.fc4 = fc(scale_inp_sizes[2], scale_inp_sizes[3])#, act=None)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        scale_outputs = self.fc4(x)

        return scale_outputs