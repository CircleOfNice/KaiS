import torch
import numpy as np
import random
import torch.nn as nn
from algorithm_torch.fc_layer import expand_act_on_state
from algorithm_torch.Nodenet import *
from algorithm_torch.Scalenet import *
from algorithm_torch.Ocn_net import *
from algorithm_torch.gcn import GraphCNN
from algorithm_torch.Orchestration_Agent import *
from algorithm_torch.Agent import Agent
from algorithm_torch.helpers_main_pytorch import high_value_edge_nodes, flatten, output_dim
class OrchestrateAgent(Agent):
    def __init__(self,node_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, MAX_TASK_TYPE, entropy_weight,eps, act_fn,optimizer):
        """Orchestration Agent initialisation

        Args:
            node_input_dim ([int]): [Input dimension of the node part of orchestration net]
            scale_input_dim ([int]): [Input dimension of the service scaling part of orchestration net]
            hid_dims ([list]): [int]
            output_dim ([int]): [Output dimensions of OCN Net (also for the inbuilt )]
            max_depth ([int]): [description]
            executor_levels ([range]): [Levels of Execution (for Tiling)]
            eps ([float]): [Epsilon value to avoid numerical instabilities]
            act_fn ([Pytorch Activation function]): [Pytorch Activation ]
            optimizer ([Pytorch Optimizer]): [Pytorch Optimizer]
        """
        Agent.__init__(self)
        self.node_input_dim =  node_input_dim
        self.scale_input_dim =2*MAX_TASK_TYPE# scale_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth

        self.merge_node_dim_ = self.node_input_dim +  output_dim # node_inputs_reshape.shape + gcn_outputs_reshape.shape (along axis 2)
        self.expanded_state_dim_ = 1 + 24 # node_inputs_reshaped.shape + executor levels range (along axis 2)
        self.executor_levels = executor_levels
        self.eps = eps #=1e-6
        self.act_fn = act_fn
        self.entropy_weight = entropy_weight
        self.MAX_TASK_TYPE = MAX_TASK_TYPE
        self.gcn = GraphCNN(
            self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn)
        
        self.orchestrate_network( act_fn = self.act_fn)
        self.optimizer = optimizer(self.ocn_net.parameters(), lr = 0.001)
        
    def orchestrate_network(self, act_fn):
        """Initialize and orchestrate the agent

        Args:
            act_fn ([Pytorch Activation Function]): [Pytorch Activation function]
        """
        batch_size = 1
    
        self.ocn_net = OCN(self.merge_node_dim_, self.expanded_state_dim_, self.node_input_dim, 
    self.scale_input_dim, self.output_dim, expand_act_on_state, self.executor_levels,
    node_inp_sizes = [32, 16, 8 ,1], scale_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU(), batch_size = batch_size)
    
    def save_model(self, file_path):
        """Saving the model at desired path

        Args:
            file_path ([str]): [Path for saving the model]
        """
        torch.save(self.ocn_net, file_path)
        
    def translate_state(self, obs):
        """Translates the state (from observation environment returns Node and scale inputs)

        Args:
            obs ([list]): [Observation of environment]

        Returns:
            [Numpy arrays]: [Node Inputs, scale inputs for next state]
        """
        done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state, cpu_lists, mem_lists, task_lists, gcnn_list = obs

        g_out_list = []
        for i, gcnn in enumerate(gcnn_list):
            node_input =[]
            node_input.append(cpu_lists[i][:])
            node_input.append(mem_lists[i][:])
            node_input.append(task_lists[i][:])
            node_input = flatten(node_input)

            node_input = np.asarray(node_input)
            node_input = np.expand_dims(node_input, axis=0)

            gcnn(node_input)
            g_out_list.append(gcnn.outputs)
  
        done_tasks = np.array(done_tasks)
        undone_tasks = np.array(undone_tasks)
        curr_tasks_in_queue = np.array(curr_tasks_in_queue)
        deploy_state = np.array(deploy_state)
        
        # Compute total number of nodes
        total_num_nodes = len(curr_tasks_in_queue)
        # Inputs to feed

        # Add values to the node inputs task_list, mem_list, cpu_list etc
        node_inputs = np.zeros([total_num_nodes, 2*self.MAX_TASK_TYPE+ len(gcnn_list)*output_dim])
        scale_inputs = np.zeros([1, self.scale_input_dim])
        
        new_deploy_state = []
        for element in deploy_state:
            for elem in element:
                new_deploy_state.append(elem)

        deploy_state = new_deploy_state

        for i in range(len(node_inputs)):
            ds_int_list = [int(x) for x in deploy_state[i][:][:]] 
            node_inputs[i, :self.MAX_TASK_TYPE] = ds_int_list
            node_inputs[i, self.MAX_TASK_TYPE: 2*self.MAX_TASK_TYPE] = deploy_state[i]
            for j in range(len(gcnn_list)):
                node_inputs[i, 2*self.MAX_TASK_TYPE + (j)*output_dim: 2*self.MAX_TASK_TYPE+(j+1)*output_dim] = gcnn_list[j].outputs.numpy()
            
        scale_inputs[0, :self.MAX_TASK_TYPE] = done_tasks[:self.MAX_TASK_TYPE]
        scale_inputs[0, self.MAX_TASK_TYPE:] = undone_tasks[:self.MAX_TASK_TYPE]
        
        return node_inputs, scale_inputs
    

    def predict(self, x, epsilon_exploration):
        """Function to make predictions

        Args:
            x ([list]): [list containing scale and node inputs and gcn outputs]

        Returns:
            [list]: [list of Tensors]
        """

        self.node_inputs, self.scale_inputs, self.gcn.outputs = x
        
        self.optimizer.zero_grad()
        self.gcn(self.node_inputs)
        
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.scale_act_probs = self.ocn_net.predict((self.node_inputs, self.scale_inputs, self.gcn.outputs))#

        # Draw action based on the probability
        logits = torch.log(self.node_act_probs)
        noise = torch.rand(logits.shape)
        node_val = logits 
        if epsilon_exploration:
                if random.uniform(0, 1)< 0.05:
                    node_val = logits

        self.node_acts = torch.topk(node_val, k=high_value_edge_nodes).indices
        
        # scale_acts
        logits = torch.log(self.scale_act_probs)
        noise = torch.rand(logits.shape)
        
        scale_val = logits 
        if epsilon_exploration:
                if random.uniform(0, 1)< 0.05:
                    scale_val = logits - torch.log(-torch.log(noise))

        self.scale_acts = torch.topk(scale_val, k=high_value_edge_nodes).indices
        return [self.node_act_probs, self.scale_act_probs, self.node_acts, self.scale_acts]    

    def invoke_model(self, obs, epsilon_exploration):
        """[Propagates the model inputs]

        Args:
            obs ([list]): [list of observations]

        Returns:
            [type]: [List of predictions containing, node and scale net outputs and also the new node and scale inputs ]
        """

        node_inputs, scale_inputs = self.translate_state(obs)
        
        self.gcn(node_inputs)

        node_act_probs, scale_act_probs, node_acts, scale_acts = \
            self.predict((node_inputs, scale_inputs, self.gcn.outputs), epsilon_exploration)
        return node_acts, scale_acts, \
               node_act_probs, scale_act_probs, \
               node_inputs, scale_inputs   