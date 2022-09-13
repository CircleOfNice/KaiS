import torch.nn as nn
from algorithm_torch.fc_layer import fc

class NodeNet(nn.Module):
    """Node part of the orchestrate neural network
    """
    def __init__(self, merge_node:int, node_inp_sizes:list = [32, 16, 8 ,1], act:nn = nn.ReLU()):
        """Initialisation of attributes

        Args:
            merge_node ([int]): [Input dimensions]
            node_inp_sizes (list, optional): [Hidden Dimensions]. Defaults to [32, 16, 8 ,1].
            act ([Python Activation Layer], optional): [Python Activation Layer]. Defaults to nn.ReLU().
        """
        super().__init__()
        self.fc1 = fc(merge_node, node_inp_sizes[0], act=act)
        self.fc2 = fc(node_inp_sizes[0], node_inp_sizes[1], act=act)
        self.fc3 = fc(node_inp_sizes[1], node_inp_sizes[2], act=act)
        self.fc4 = fc(node_inp_sizes[2], node_inp_sizes[3])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        node_outputs = self.fc4(x)

        return node_outputs