import torch
import torch.nn as nn
import numpy as np 

class GraphSNN(nn.Module):
    """GraphCNN Class
    """
    def __init__(self, input_dim, hid_dims, output_dim, act_fn):
        """[summary]

        Args:
            input_dim ([int]): [Dimension of input dimensions]
            hid_dims ([list]): [List of hidden dimension]
            output_dim ([int]): [output dimension]
            act_fn ([Pytorch Action]): [Activation Function]
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims
        self.act_fn = act_fn

        self.sum_levels = 2
    # initialize summarization parameters
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.global_weights, self.global_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        # graph summarization operation

    def glorot(self, shape):
        """[summary]

        Args:
            shape ([int]): [Shape required for initialization]

        Returns:
            [Pytorch Layer]: [Glorot initialization Layer]
        """
        init = nn.init.xavier_uniform_(torch.empty(shape))
        return init
    
    def init(self, input_dim, hid_dims, output_dim):
        """Initialization of layers

        Args:
            input_dim ([int]): [Input Dimension]
            hid_dims ([list]): [Hidden Dimensions]
            output_dim ([int]): [Output Dimensions]

        Returns:
            [lists]: [list of weights and biases]
        """
        weights = []
        bias = []
        curr_in_dim = input_dim
        
        # Hidden Layers
        for hid_dim in hid_dims:
            
            weights.append(self.glorot([curr_in_dim, hid_dim]))
            bias.append(torch.zeros(hid_dim))
            curr_in_dim = hid_dim
        # Output layer
        weights.append(self.glorot([curr_in_dim, output_dim],))
        bias.append(torch.zeros([output_dim]))
        return weights, bias
        
    def forward(self, s):
        # summarize information
        summaries = []
        for i in range(len(self.dag_weights)):
            s = torch.matmul(s.float(), self.dag_weights[i])
            s += self.dag_bias[i]
            s = self.act_fn(s)
        summaries.append(s)
        # global level summary
        for i in range(len(self.global_weights)):
            s = torch.matmul(s, self.global_weights[i])
            s += self.global_bias[i]
            s = self.act_fn(s)
        summaries.append(s)
        self.summaries = summaries
        return summaries