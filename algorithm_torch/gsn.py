import torch
import torch.nn as nn
import numpy as np 

class GraphSNN(object):
    def __init__(self, input_dim, hid_dims, output_dim, act_fn):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dims = hid_dims
        self.act_fn = act_fn
        #self.scope = scope
        self.sum_levels = 2
    # initialize summarization parameters
        self.dag_weights, self.dag_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.global_weights, self.global_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        # graph summarization operation
        #self.summaries = self.summarize()
    def glorot(self, shape):
        init = nn.init.xavier_uniform_(torch.empty(shape))
        return init
    
    def init(self, input_dim, hid_dims, output_dim):
        weights = []
        bias = []
        curr_in_dim = input_dim
        
        # Hidden Layers
        
        for hid_dim in hid_dims:
            weights.append(self.glorot([curr_in_dim, hid_dim]))
            bias.append(torch.zeros(hid_dim))
            
        # Output layer
        weights.append(self.glorot([curr_in_dim, output_dim],))
        bias.append(torch.zeros([output_dim]))
        return weights, bias
        
    def forward(self, s):
        # summarize information
        summaries = []
        for i in range(len(self.dag_weights)):
            s = torch.matmul(s, self.dag_weights[i])
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