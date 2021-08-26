import torch
import torch.nn as nn
from algorithm_torch.Nodenet import *
from algorithm_torch.Clusternet import *


class OCN(nn.Module):
    """Class for definition for Orchestration Network
    """
    def __init__(self, merge_node_dim, expanded_state_dim, node_input_dim, 
    cluster_input_dim, output_dim, expand_act_on_state, executor_levels, 
    node_inp_sizes = [32, 16, 8 ,1], cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU(), batch_size = 1):
        """Creation of the cumulative 

        Args:
            merge_node_dim ([int]): [Input Dimension for Node Net] # node_inputs_reshape.shape + gcn_outputs_reshape.shape (along axis 2)
            expanded_state_dim ([int]): [Input Dimension for Cluster Net] # node_inputs_reshaped.shape + executor levels length (along axis 2)
            node_input_dim ([int]): [Reshape Node Inputs Dimension ]
            cluster_input_dim ([int]): [Reshape Cluster Inputs Dimension ]
            output_dim ([int]): [Output Dimensions of Graph Neural Networks]
            expand_act_on_state ([Function]): [Function to concatenate the cluster]
            executor_levels ([type]): [Executor Levels (Tiling Length for Cluster Input)] 
            node_inp_sizes (list, optional): [Node net hidden dims]. Defaults to [32, 16, 8 ,1].
            cluster_inp_sizes (list, optional): [Cluster net hidden dims]. Defaults to [32, 16, 8 ,1].
            act ([Python Activation Layer], optional): [Python Activation Layer]. Defaults to nn.ReLU().
            batch_size (int, optional): [Batch Size]. Defaults to 1.
        """
        super().__init__()
        
        self.merge_node_dim = merge_node_dim
        self.expanded_state_dim = expanded_state_dim
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.output_dim = output_dim
        self.expand_act_on_state = expand_act_on_state
        self.batch_size = batch_size
        self.executor_levels = executor_levels
        self.cluster_inp_sizes = cluster_inp_sizes
        self.nodenet = NodeNet(merge_node_dim, node_inp_sizes = node_inp_sizes, act = nn.ReLU())
        self.clusterenet = ClusterNet(expanded_state_dim, cluster_inp_sizes = self.cluster_inp_sizes, act = nn.ReLU())

    def propagate(self, x):
        """Common function to propagate the input through the OCN network

        Args:
            x ([tuple ]): [Tuple containing node inputs, cluster inputs and outputs of GCN Network]

        Returns:
            [tuple]: [Tuple containing node outputs, cluster outputs]
        """
        
        node_inputs, cluster_inputs, gcn_outputs = x

        node_inputs = torch.from_numpy(node_inputs).float()
        cluster_inputs = torch.from_numpy(cluster_inputs).float()

        node_inputs_reshape = node_inputs.view(self.batch_size, -1, self.node_input_dim)
        cluster_inputs_reshape = cluster_inputs.view(self.batch_size, -1, self.cluster_input_dim)
        gcn_outputs_reshape = gcn_outputs.view(self.batch_size, -1, self.output_dim)
        
        merge_node = torch.cat((node_inputs_reshape, gcn_outputs_reshape), axis=2)
        node_outputs = self.nodenet(merge_node)
        node_outputs = node_outputs.view(self.batch_size, -1)
        node_outputs = nn.functional.softmax(node_outputs)
        
        merge_cluster = torch.cat([cluster_inputs_reshape, ], axis=2)
        expanded_state = self.expand_act_on_state(
                merge_cluster, [l / 50.0 for l in self.executor_levels])
        cluster_outputs = self.clusterenet(expanded_state)
            
        cluster_outputs = cluster_outputs.view(self.batch_size, -1)
        cluster_outputs = cluster_outputs.view(self.batch_size, -1, len(self.executor_levels))

        # Do softmax
        cluster_outputs = nn.functional.softmax(cluster_outputs)
        return node_outputs, cluster_outputs
        
    def predict(self, x):
        """ Function to predict the output given inputs

        Args:
            x ([tuple ]): [Tuple containing node inputs, cluster inputs and outputs of GCN Network]

        Returns:
            [tuple]: [Tuple containing node outputs, cluster outputs]
        """
        self.batch_size = 1
        node_outputs, cluster_outputs = self.propagate(x)
        return node_outputs, cluster_outputs  
    
    def forward(self, x):
        node_inputs, _, _ = x
        self.batch_size = node_inputs.shape[0]
        
        node_outputs, cluster_outputs = self.propagate(x)
        return node_outputs, cluster_outputs        