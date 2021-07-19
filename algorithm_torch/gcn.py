import numpy as np
import torch
import torch.nn as nn

class GraphCNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim,
    max_depth, act_fn):
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
        #self.outputs = self.forward()
        
    
    def glorot(self, shape):
        init = nn.init.xavier_uniform_(torch.empty(shape))
        return init
    
        
    def init(self, input_dim, hid_dims, output_dim):
        weights = []
        bias = []
        curr_in_dim = input_dim
        
        
        # Hidden Layers
        #print('start :  ',input_dim, hid_dims, output_dim)
        for hid_dim in hid_dims:
            #print('Hid Dim : ', curr_in_dim, hid_dim)
            
            weights.append(self.glorot([curr_in_dim, hid_dim]))
            bias.append(torch.zeros(hid_dim))
            curr_in_dim = hid_dim
            
        # Output layer
        weights.append(self.glorot([curr_in_dim, output_dim],))
        bias.append(torch.zeros([output_dim]))
        '''
        print('weights : ' , output_dim)
        for l in range(len(weights)):
            print(weights[l].shape)
        
        print('bias : ', )
        for l in range(len(bias)):
            print(bias[l].shape)
            
        '''
        return weights, bias
        
    def forward(self, x):
        x = torch.from_numpy(x)
        # Raise x into higher dimension
        #print(self.prep_weights)
        #for l in range(len(self.prep_weights)):
        #    print(self.prep_weights[l].shape) 
            
        for l in range(len(self.prep_weights)):
            #print(type(x), type(self.prep_weights[l]), x.shape, self.prep_weights[l].shape)
            #print()
            #print()
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