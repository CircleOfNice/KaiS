import numpy as np
import torch
import torch.nn as nn

class GraphCNN(nn.Module):
    """GraphCNN Class
    """
    def __init__(self, input_dim, hid_dims, output_dim,
    max_depth, act_fn):
        """[summary]

        Args:
            input_dim ([int]): [Dimension of input dimensions]
            hid_dims ([list]): [List of hidden dimension]
            output_dim ([int]): [output dimension]
            max_depth ([int]): [Depth to which the features are needed to be aggregated. Though output_dim and max_depth are of same value]
            act_fn ([Pytorch Action]): [Activation Function]
        """
        super().__init__()
        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.act_fn = act_fn
        # initialize message passing transformation parameters
        self.prep_weights, self.prep_bias = self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.proc_weights, self.proc_bias = self.init(self.output_dim, self.hid_dims, self.output_dim)
        self.agg_weights, self.agg_bias = self.init(self.output_dim, self.hid_dims, self.output_dim)

        
    
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
        
    def forward(self, x):
        
        x = torch.from_numpy(x)
        #print('input shape : ', x.shape)
        # Raise x into higher dimension
            
        for l in range(len(self.prep_weights)):

            x = torch.matmul(x.float(), self.prep_weights[l])
            x += self.prep_bias[l]
            x = self.act_fn(x)
            
        for d in range(self.max_depth):
            y = x
            # Process the features
            for l in range(len(self.proc_weights)):
                y = torch.matmul(y, self.proc_weights[l])
                y += self.proc_bias[l]
                y = self.act_fn(y)
            # Aggregate features
            for l in range(len(self.agg_weights)):
                y = torch.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = self.act_fn(y)

            # assemble neighboring information
            x = x + y
            
        self.outputs = x
        return x