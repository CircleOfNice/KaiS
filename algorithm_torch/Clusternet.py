import torch.nn as nn
from algorithm_torch.fc_layer import fc


class ClusterNet(nn.Module):
    """Cluster part of the orchestrate neural network
    """
    def __init__(self, expanded_state, cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU()):
        """Initialisation

        Args:
            expanded_state ([int]): [Input dimensions] # cluster merged reshape input shape + executor levels shape 24
            cluster_inp_sizes (list, optional): [Hidden Dimensions]. Defaults to [32, 16, 8 ,1].
            act ([Python Activation Layer], optional): [Python Activation Layer]. Defaults to nn.ReLU().
        """
        super().__init__()
        self.fc1 = fc(expanded_state, cluster_inp_sizes[0], act=act)
        self.fc2 = fc(cluster_inp_sizes[0], cluster_inp_sizes[1], act=act)
        self.fc3 = fc(cluster_inp_sizes[1], cluster_inp_sizes[2], act=act)
        self.fc4 = fc(cluster_inp_sizes[2], cluster_inp_sizes[3])#, act=None)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        cluster_outputs = self.fc4(x)

        return cluster_outputs