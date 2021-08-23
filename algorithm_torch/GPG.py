# GNN-based Learning for Service Orchestration

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import bisect
from algorithm_torch.gcn import GraphCNN
from algorithm_torch.gsn import GraphSNN


def discount(x, gamma):
    """Calculate the discounted cumulative reward
        Cumulative Reward = r_t + gamma * r_t+1 + gamma ^2 * r_t+2 + ________

    Args:
        x (Numpy array): numpy array of rewards over time
        gamma (float): Discount factor

    Returns:
        numpy array: Calculated cumulative discounted reward
    """
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out

def fc(inp_dim, output_dim, act=nn.ReLU()):
    """Fully Connected Layer block

    Args:
        inp_dim (int): Input Dimension for the given layer
        output_dim (int): Input Dimension for the given layer
        act (Pytorch Activation Layer, optional): Pytorch Activation Layer. Defaults to nn.ReLU().

    Returns:
    
        Sequential Model: A sequential model which can be used as layer in Functional model
    """
    linear = nn.Linear(inp_dim, output_dim)
    fc_out = nn.Sequential(linear, act)
    return fc_out 

def invoke_model(orchestrate_agent, obs, exp):
    """[Invoke model (propagate the observation through the orchestration model) and return choice of nodes given the observation]

    Args:
        orchestrate_agent ([OrchestrateAgent Type]): [Instance of Orchestrate Agent]
        obs ([list]): [Observations containning done tasks, undone tasks, current tasks in queue, deploy state]
        exp ([dict]): [Experience]

    Returns:
        [list, list , dictionary]: [chosen node, chosen services and the appended experience]
    """
    
    node_act, cluster_act, node_act_probs, cluster_act_probs, node_inputs, cluster_inputs = \
        orchestrate_agent.invoke_model(obs)
    node_choice = [x for x in node_act[0]]
    server_choice = []

    for x in cluster_act[0][0]: # Server choice here is chosen services
        if x >= 12:
            server_choice.append(x - 11)
        else:
            server_choice.append(x - 12)
    node_act_vec = np.ones(node_act_probs.shape)
    # For storing cluster index
    cluster_act_vec = np.ones(cluster_act_probs.shape)

    # Store experience
    exp['node_inputs'].append(node_inputs)
    exp['cluster_inputs'].append(cluster_inputs)
    exp['node_act_vec'].append(node_act_vec)
    exp['cluster_act_vec'].append(cluster_act_vec)
    return node_choice, server_choice, exp
    
def expand_act_on_state(state, sub_acts):
    """Function to concatenate states with sub_acts with multiple repetition over the tiling range (24)
    # Repeated state is appended with a vector of values havings different weightage likely to activate different actions for different state
    
    Args:
        state ([Pytorch Tensor]): [State Matrix]
        sub_acts ([list]): [list for tiling operation]

    Returns:
        [Pytorch Tensor]: [Concatenated State Tensor]
    """
    # Expand the state
    batch_size = state.shape[0]
    num_nodes = state.shape[1]
    num_features = state.shape[2]#.value
    
    expand_dim = len(sub_acts)

    # Replicate the state
    state = torch.tile(state, [1, 1, expand_dim])
    state = state.view([batch_size, num_nodes * expand_dim, num_features])

    # Prepare the appended actions
    sub_acts = torch.FloatTensor(sub_acts)
    sub_acts = sub_acts.view([1, 1, expand_dim])

    sub_acts = torch.tile(sub_acts, [1, 1, num_nodes])

    sub_acts = sub_acts.view([1, num_nodes * expand_dim, 1])
    sub_acts = torch.tile(sub_acts, [batch_size, 1, 1])
    # Concatenate expanded state with sub-action features
    concat_state = torch.cat([state, sub_acts], axis=2)

    return concat_state

def act_offload_agent(orchestrate_agent, exp, done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state):
    """Action choice using the invocation (propagation through) of Orchestrate Agent model
    
    

    Args:
        orchestrate_agent ([OrchestrateAgent Type]): [Instance of Orchestrate Agent]
        exp ([dictionary]): [Experience dictionary]
        done_tasks ([list]): [list of done tasks]
        undone_tasks ([list]): [lists of undone(not done) tasks]
        curr_tasks_in_queue ([list]): [list of tasks in queue]
        deploy_state ([list]): [List of lists containing the deployment of nodes]

    Returns:
        [node, use_exec, exp]: [chosen node, chosen service and the appended experience]
    """
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state]

    node, use_exec, exp = invoke_model(orchestrate_agent, obs, exp)
    return node, use_exec, exp


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    """Generate a piecewise linear fit

    Args:
        all_cum_rewards ([list]): [All Cumulative Rewards]
        all_wall_time ([list]): [Time]

    Returns:
        [baselines]: [returns a list of piecewise linear data extrapolation]
    """

    assert len(all_cum_rewards) == len(all_wall_time)
    # All time
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    # Find baseline value
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_wall_time))
    # Output n baselines
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)
    return baselines



def compute_orchestrate_loss(orchestrate_agent, exp, batch_adv, entropy_weight):
    """[Computation of orchestration loss]

    Args:
        orchestrate_agent ([Orchestrate Agent Class]): [Orchestrate Agent]
        exp ([dictionary]): [Experience]
        batch_adv ([numpy array]): [difference between qvalue target and q value predicted]
        entropy_weight ([int]): [Entropy weight]

    Returns:
        [Tensor]: [Computed Loss]
    """
    loss = 0
    batch_adv = np.array(batch_adv)
    # Use a piece of experience
    node_inputs = exp['node_inputs']
    cluster_inputs = exp['cluster_inputs']
    node_act_vec = exp['node_act_vec']
    cluster_act_vec = exp['cluster_act_vec']
    adv = batch_adv

    node_inputs = np.array(node_inputs)

    cluster_inputs = np.array(cluster_inputs)
    

    node_act_vec = np.array(node_act_vec)

    cluster_act_vec = np.array(cluster_act_vec)
    
    loss = orchestrate_agent.act_loss(
        node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv)

    return loss


def decrease_var(var, min_var, decay_rate):
    """Function to decrease Variable generally entropy

    Args:
        var ([float]): [Variable]
        min_var ([float]): [min_allowed value of variable]
        decay_rate ([float]): [Decay rate for the variable]

    Returns:
        [var]: [Variable with reduced value]
    """

    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    
    """[Train the orchestration agent]

    Args:
        orchestrate_agent ([Orchestrate Agent Class]): [Orchestrate Agent]
        exp ([dictionary]): [Experience]
        entropy_weight ([float]): [Entropy weight]
        entropy_weight_min ([float]): [Minimum Entropy Weight]
        entropy_weight_decay ([type]): [Entropy Weight Decay rate]

    Returns:
        [Tensors]: [Entropy weight and calculated loss]
    """
    all_cum_reward = []
    all_rewards = exp['reward']
    batch_time = exp['wall_time']

    rewards = np.array([r for (r, t) in zip(all_rewards, batch_time)])
    cum_reward = discount(rewards, 1)
    all_cum_reward.append(cum_reward)
    orchestrate_agent.entropy_weight = entropy_weight
    
    # Compute baseline
    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [batch_time])
    # Calculate the advantage
    batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
    orchestrate_agent.entropy_weight = entropy_weight
    
    # Actual training of Orchestrate Net
    orchestrate_agent.optimizer.zero_grad()
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv, entropy_weight)
    loss.backward()
    orchestrate_agent.optimizer.step()
    entropy_weight = decrease_var(entropy_weight,
                                  entropy_weight_min, entropy_weight_decay)
    return entropy_weight, loss

class NodeNet(nn.Module):
    """Node part of the orchestrate neural network
    """
    def __init__(self, merge_node, node_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU()):
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
        self.fc4 = fc(node_inp_sizes[2], node_inp_sizes[3])# act=None)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        node_outputs = self.fc4(x)

        return node_outputs

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


class Agent(object):
    # Abstraction class (Not really)
    def __init__(self):
        pass
        
class OrchestrateAgent(Agent):
    def __init__(self, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, MAX_TESK_TYPE, eps, act_fn,optimizer):
        """Orchestration Agent initialisation

        Args:
            node_input_dim ([int]): [Input dimension of the node part of orchestration net]
            cluster_input_dim ([int]): [Input dimension of the cluster part of orchestration net]
            hid_dims ([list]): [int]
            output_dim ([int]): [Output dimensions of OCN Net (also for the inbuilt )]
            max_depth ([int]): [description]
            executor_levels ([range]): [Levels of Execution (for Tiling)]
            eps ([float]): [Epsilon value to avoid numerical instabilities]
            act_fn ([Pytorch Activation function]): [Pytorch Activation ]
            optimizer ([Pytorch Optimizer]): [Pytorch Optimizer]
        """
        Agent.__init__(self)
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim




        self.max_depth = max_depth
        self.merge_node_dim_ = 24+ 8 # node_inputs_reshape.shape + gcn_outputs_reshape.shape (along axis 2)
        self.expanded_state_dim_ = 1 + 24 # node_inputs_reshaped.shape + executor levels range (along axis 2)
        self.executor_levels = executor_levels
        self.eps = eps #=1e-6
        self.act_fn = act_fn
        self.entropy_weight = 1
        self.MAX_TESK_TYPE = MAX_TESK_TYPE
        self.gcn = GraphCNN(
            self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn)
        self.gsn = GraphSNN(
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn)
        
        self.orchestrate_network( act_fn = self.act_fn)
        self.optimizer = optimizer(self.ocn_net.parameters(), lr = 0.001)
        
    def orchestrate_network(self, act_fn):
        """Initialize and orchestrate the agent

        Args:
            act_fn ([Pytorch Activation Function]): [Pytorch Activation function]
        """
        batch_size = 1
    
        self.ocn_net = OCN(self.merge_node_dim_, self.expanded_state_dim_, self.node_input_dim, 
    self.cluster_input_dim, self.output_dim, expand_act_on_state, self.executor_levels,
    node_inp_sizes = [32, 16, 8 ,1], cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU(), batch_size = batch_size)
    
    def save_model(self, file_path):
        """Saving the model at desired path

        Args:
            file_path ([str]): [Path for saving the model]
        """
        torch.save(self.ocn_net, file_path)
        
    def act_loss(self, node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv):
        """act_loss

        Args:
            node_inputs ([Numpy array]): [Node inputs]
            cluster_inputs ([Numpy array]): [Cluster inputs]
            node_act_vec ([Numpy array]): [Node Activation Vectors]
            cluster_act_vec ([Numpy array]): [Cluster Activation Vectors]
            adv ([Numpy array]): [Calcualted advantage (difference between Q value predicted and Q value targets)]

        Returns:
            [Tensor]: [Calculated Loss]
        """
        
        self.node_inputs = np.asarray(node_inputs)
        self.cluster_inputs = np.asarray(cluster_inputs)
        
        node_act_vec = np.asarray(node_act_vec)
        cluster_act_vec = np.asarray(cluster_act_vec)

        self.gcn(self.node_inputs)
        
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.cluster_act_probs = self.ocn_net((self.node_inputs, self.cluster_inputs, self.gcn.outputs) )#

        # Draw action based on the probability
        logits = torch.log(self.node_act_probs)

        noise = torch.rand(logits.shape)

        self.node_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Cluster_acts
        logits = torch.log(self.cluster_act_probs)

        noise = torch.rand(logits.shape)

        self.cluster_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices
        
        node_act_probs = torch.tensor(self.node_act_probs)
        node_act_vec = torch.tensor(node_act_vec)
        node_act_vec = torch.squeeze(node_act_vec)

        node_prod = torch.mul(
            node_act_probs, node_act_vec)

        cluster_act_probs, cluster_act_vec = torch.tensor(self.cluster_act_probs), torch.tensor(cluster_act_vec)
        cluster_act_vec = torch.squeeze(cluster_act_vec, dim= 1)
        selected_node_prob = torch.sum(node_prod,
            dim=(1,), keepdim=True)

        select_cluster_prod = torch.mul( self.cluster_act_probs, cluster_act_vec)
        sum_cluster_1 = torch.sum(select_cluster_prod, dim=2)
        
        selected_cluster_prob = torch.sum(sum_cluster_1, dim=1, keepdim=True)

        torch_log = torch.log(selected_node_prob * selected_cluster_prob + \
                   self.eps)

        adv = torch.tensor(adv)
        
        torch_log_adv_mul = torch.mul(torch_log, -adv)

        # Orchestrate loss due to advantge
        self.adv_loss = torch.sum(torch_log_adv_mul)

        # Node_entropy
        torch_log_entropy = torch.log(node_act_probs + self.eps)

        torch_mul_dimension = torch.mul(self.node_act_probs, torch_log_entropy)
        self.node_entropy = torch.sum(torch_mul_dimension)

        # Entropy loss
        self.entropy_loss = self.node_entropy  # + self.cluster_entropy

        # Normalize entropy
        len_ex = float(len(self.executor_levels))
        len_ex = torch.tensor(len_ex)
        node_act_probs_shape = torch.tensor(node_act_probs.shape[1])
        torch_log_norm = torch.log(len_ex)

        denom = (torch.log(node_act_probs_shape) + \
             torch_log_norm)
        
        denom = denom.type(torch.FloatTensor)
        self.entropy_loss /= denom

        # Define combined loss
        self.act_loss_ = self.adv_loss + self.entropy_weight * self.entropy_loss

        return self.act_loss_
        
    def translate_state(self, obs):
        """Translates the state (from observation environment returns Node and cluster inputs)

        Args:
            obs ([list]): [Observation of environment]

        Returns:
            [Numpy arrays]: [Node Inputs, Cluster inputs for next state]
        """
        done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state = obs
        done_tasks = np.array(done_tasks)
        undone_tasks = np.array(undone_tasks)
        curr_tasks_in_queue = np.array(curr_tasks_in_queue)
        deploy_state = np.array(deploy_state)

        # Compute total number of nodes
        total_num_nodes = len(curr_tasks_in_queue)

        # Inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        cluster_inputs = np.zeros([1, self.cluster_input_dim])

        for i in range(len(node_inputs)):
            node_inputs[i, :self.MAX_TESK_TYPE] = curr_tasks_in_queue[i, :self.MAX_TESK_TYPE]
            node_inputs[i, self.MAX_TESK_TYPE:] = deploy_state[i, :self.MAX_TESK_TYPE]
        cluster_inputs[0, :self.MAX_TESK_TYPE] = done_tasks[:self.MAX_TESK_TYPE]
        cluster_inputs[0, self.MAX_TESK_TYPE:] = undone_tasks[:self.MAX_TESK_TYPE]
        
        #))
        return node_inputs, cluster_inputs
    

    def predict(self, x):
        """Function to make predictions

        Args:
            x ([list]): [list containing cluster and node inputs and gcn outputs]

        Returns:
            [list]: [list of Tensors]
        """

        self.node_inputs, self.cluster_inputs, self.gcn.outputs = x
        
        self.optimizer.zero_grad()
        self.gcn(self.node_inputs)
        self.gsn(torch.cat((torch.tensor(self.node_inputs), self.gcn.outputs), axis=1))
        
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.cluster_act_probs = self.ocn_net.predict((self.node_inputs, self.cluster_inputs, self.gcn.outputs) )#
        
        # Draw action based on the probability
        logits = torch.log(self.node_act_probs)
        noise = torch.rand(logits.shape)
        self.node_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Cluster_acts
        logits = torch.log(self.cluster_act_probs)
        noise = torch.rand(logits.shape)
        self.cluster_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        return [self.node_act_probs, self.cluster_act_probs, self.node_acts, self.cluster_acts]    

    def invoke_model(self, obs):
        """[Propagates the model inputs]

        Args:
            obs ([list]): [list of observations]

        Returns:
            [type]: [List of predictions containing, node and cluster net outputs and also the new node and cluster inputs ]
        """

        node_inputs, cluster_inputs = self.translate_state(obs)
        self.gcn(node_inputs)
        node_act_probs, cluster_act_probs, node_acts, cluster_acts = \
            self.predict((node_inputs, cluster_inputs, self.gcn.outputs))
        return node_acts, cluster_acts, \
               node_act_probs, cluster_act_probs, \
               node_inputs, cluster_inputs    
               