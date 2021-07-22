import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import bisect
from algorithm_torch.gcn import GraphCNN
from algorithm_torch.gsn import GraphSNN


def discount(x, gamma):
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out

def fc(inp_dim, output_dim, act=nn.ReLU()):
    linear = nn.Linear(inp_dim, output_dim)
    #nn.init.xavier_uniform_(linear.weight)
    #linear.bias.data.fill_(0)
    fc_out = nn.Sequential(linear, act)
    return fc_out 

def invoke_model(orchestrate_agent, obs, exp):
    node_act, cluster_act, node_act_probs, cluster_act_probs, node_inputs, cluster_inputs = \
        orchestrate_agent.invoke_model(obs)
    node_choice = [x for x in node_act[0]]
    server_choice = []
    for x in cluster_act[0][0]:
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
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state]
    node, use_exec, exp = invoke_model(orchestrate_agent, obs, exp)
    return node, use_exec, exp


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
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
    batch_points = 2
    loss = 0
    for b in range(batch_points - 1):
        ba_start = 0
        ba_end = -1
        # Use a piece of experience
        node_inputs = exp['node_inputs']
        cluster_inputs = exp['cluster_inputs']
        node_act_vec = exp['node_act_vec']
        cluster_act_vec = exp['cluster_act_vec']
        adv = batch_adv[ba_start: ba_end, :]
        #print('cluster_act_vec : ', cluster_act_vec) 
        #print('node_act_vec', node_act_vec)
        #print('cluster_inputs', cluster_inputs)
        loss = orchestrate_agent.act_loss(
            node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv)
    return loss


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    all_cum_reward = []
    all_rewards = exp['reward']
    batch_time = exp['wall_time']
    all_times = batch_time[1:]
    all_diff_times = np.array(batch_time[1:]) - np.array(batch_time[:-1])
    rewards = np.array([r for (r, t) in zip(all_rewards, all_diff_times)])
    cum_reward = discount(rewards, 1)
    all_cum_reward.append(cum_reward)

    # Compute baseline
    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [all_times])

    # Back the advantage
    batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
    #print('Inside train_orchestrate_agent')
    #print('orchestrate_agent '
    # Compute gradients
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv, entropy_weight)
    entropy_weight = decrease_var(entropy_weight,
                                  entropy_weight_min, entropy_weight_decay)
    return entropy_weight, loss

class NodeNet(nn.Module):
    def __init__(self, merge_node, node_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU()):
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
    def __init__(self, expanded_state, cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU()):
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
    def __init__(self, merge_node_dim, expanded_state_dim, node_input_dim, 
    cluster_input_dim, output_dim, expand_act_on_state, executor_levels, 
    node_inp_sizes = [32, 16, 8 ,1], cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU(), batch_size = 1):
        super().__init__()
        
        self.merge_node_dim = merge_node_dim
        self.expanded_state_dim = expanded_state_dim
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.output_dim = output_dim
        self.expand_act_on_state = expand_act_on_state
        self.batch_size = batch_size
        self.executor_levels = executor_levels
        self.nodenet = NodeNet(merge_node_dim, node_inp_sizes = node_inp_sizes, act = nn.ReLU())
        self.clusterenet = ClusterNet(expanded_state_dim, cluster_inp_sizes = cluster_inp_sizes, act = nn.ReLU())

    def forward(self, x):
        node_inputs, cluster_inputs, gcn_outputs = x
        node_inputs = torch.from_numpy(node_inputs).float()
        cluster_inputs = torch.from_numpy(cluster_inputs).float()
        #node_inputs = torch.from_numpy(node_inputs).float()
        node_inputs_reshape = node_inputs.view(self.batch_size, -1, self.node_input_dim)
        cluster_inputs_reshape = cluster_inputs.view(self.batch_size, -1, self.cluster_input_dim)
        gcn_outputs_reshape = gcn_outputs.view(self.batch_size, -1, self.output_dim)
        
        
        merge_node = torch.cat((node_inputs_reshape, gcn_outputs_reshape), axis=2)
        
        node_outputs = self.nodenet(merge_node)
        
        #print('node_outputs before: ', node_outputs.shape)
        
        node_outputs = node_outputs.view(self.batch_size, -1)
        node_outputs = nn.functional.softmax(node_outputs)
        
        merge_cluster = torch.cat([cluster_inputs_reshape, ], axis=2)
        expanded_state = self.expand_act_on_state(
                merge_cluster, [l / 50.0 for l in self.executor_levels])
        cluster_outputs = self.clusterenet(expanded_state)
        
            
            
        cluster_outputs = cluster_outputs.view(self.batch_size, -1)
        cluster_outputs = cluster_outputs.view(self.batch_size, -1, len(self.executor_levels))

        # Do softmax
        cluster_outputs = nn.functional.softmax(cluster_outputs)#, dim=-1)
        #print('cluster_outputs, node_outputs : ', cluster_outputs.shape, node_outputs.shape)
        return node_outputs, cluster_outputs        


class Agent(object):
    def __init__(self):
        pass
        
class OrchestrateAgent(Agent):
    def __init__(self, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps, act_fn,optimizer):
        Agent.__init__(self)
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps #=1e-6
        self.act_fn = act_fn
        
        # TODO: Relook at the follows again
        self.gcn = GraphCNN(
            self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn)
        self.gsn = GraphSNN(
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn)
            
            
            
        # skipping few of the init steps for now, will fill it again
        self.orchestrate_network( act_fn = self.act_fn)
        self.optimizer = optimizer(self.ocn_net.parameters(), lr = 0.001)
        
    def orchestrate_network(self, act_fn):    
        batch_size = 1
        
        # expanded state dim is  to be checked along with merge_node_dim #Can be done on the fly
    #    self.ocn_net = OCN(self.node_input_dim, 
    #self.cluster_input_dim, self.output_dim, expand_act_on_state, merge_node_dim= 5, expanded_state_dim= 50,
    #node_inp_sizes = [32, 16, 8 ,1], cluster_inp_sizes = [32, 16, 8 ,1], act = act_fn, batch_size = batch_size)
    
        self.ocn_net = OCN(32, 25, self.node_input_dim, 
    self.cluster_input_dim, self.output_dim, expand_act_on_state, self.executor_levels,
    node_inp_sizes = [32, 16, 8 ,1], cluster_inp_sizes = [32, 16, 8 ,1], act = nn.ReLU(), batch_size = batch_size)
        #self.optimizer(self.ocn_net)
        #return self.ocn_net
        
        
        
    # Functions apply_gradient, define_params_op, gcn_forward  get_params ,seems to be not required
    
    def save_model(self, file_path):
        torch.save(self.ocn_net, file_path)
        
    def act_loss(self, node_act_probs, node_act_vec, cluster_act_probs, cluster_act_vec, adv):
        
        node_act_probs = torch.tensor(node_act_probs)
        node_act_vec = torch.tensor(node_act_vec)
        
        prod = torch.mul(
            node_act_probs, node_act_vec)
            
        # Action probability
        
        print('cluster_act_probs : ', len(cluster_act_probs)) 
        print()
        print()
        print('cluster_act_vec :', len(cluster_act_vec))
        print()
        print()
        print(type(cluster_act_probs), type(cluster_act_vec))
        print()
        print()
        cluster_act_probs, cluster_act_vec = torch.tensor(cluster_act_probs), torch.tensor(cluster_act_vec)
        
        print(type(cluster_act_probs), type(cluster_act_vec))
        print()
        print()
        print(cluster_act_probs.shape, cluster_act_vec.shape)
        selected_node_prob = torch.sum(prod,
            dim=(1,), keepdim=True)
        selected_cluster_prob = torch.sum(torch.sum(torch.mul(
            cluster_act_probs, cluster_act_vec),
            reduction_indices=2), reduction_indices=1, keepdim=True)

        # Orchestrate loss due to advantge
        self.adv_loss = tf.sum(torch.mul(
            torch.log(selected_node_prob * selected_cluster_prob + \
                   self.eps), -adv))

        # Node_entropy
        self.node_entropy = tf.sum(torch.mul(
            self.node_act_probs, torch.log(node_act_probs + self.eps)))

        # Entropy loss
        self.entropy_loss = self.node_entropy  # + self.cluster_entropy

        # Normalize entropy
        denom = (torch.log(tf.shape(node_act_probs)[1]) + \
             torch.log(float(len(self.executor_levels))))
        denom = denom.type(torch.FloatTensor)
        self.entropy_loss /= denom

        # Define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss
        
        return self.loss
        
    def translate_state(self, obs):
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
            node_inputs[i, :12] = curr_tasks_in_queue[i, :12]
            node_inputs[i, 12:] = deploy_state[i, :12]
        cluster_inputs[0, :12] = done_tasks[:12]
        cluster_inputs[0, 12:] = undone_tasks[:12]
        return node_inputs, cluster_inputs
    
    def get_valid_masks(self, cluster_states, frontier_nodes,
                        source_cluster, num_source_exec, exec_map, action_map):
        cluster_valid_mask = \
            np.zeros([1, len(cluster_states) * len(self.executor_levels)])
        cluster_valid = {}
        base = 0
        for cluster_state in cluster_states:
            if cluster_state is source_cluster:
                least_exec_amount = \
                    exec_map[cluster_state] - num_source_exec + 1
            else:
                least_exec_amount = exec_map[cluster_state] + 1
            assert least_exec_amount > 0
            assert least_exec_amount <= self.executor_levels[-1] + 1
            # Find the index
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)
            if exec_level_idx >= len(self.executor_levels):
                cluster_valid[cluster_state] = False
            else:
                cluster_valid[cluster_state] = True
            for l in range(exec_level_idx, len(self.executor_levels)):
                cluster_valid_mask[0, base + l] = 1
            base += self.executor_levels[-1]
        total_num_nodes = int(np.sum(
            cluster_state.num_nodes for cluster_state in cluster_states))
        node_valid_mask = np.zeros([1, total_num_nodes])
        for node in frontier_nodes:
            if cluster_valid[node.cluster_state]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1

        return node_valid_mask, cluster_valid_mask

    def predict(self, x):
        # here I have to define the functioning of net (baiscally init)
        self.node_inputs, self.cluster_inputs, self.gcn.outputs = x
        
        self.optimizer.zero_grad()
        self.gcn(self.node_inputs)
        self.gsn(torch.cat((torch.tensor(self.node_inputs), self.gcn.outputs), axis=1))
        
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.cluster_act_probs = self.ocn_net((self.node_inputs, self.cluster_inputs, self.gcn.outputs) )#
            #self.gsn.summaries[0], self.gsn.summaries[1], self.act_fn)
        
        # Draw action based on the probability
        logits = torch.log(self.node_act_probs)
        noise = torch.rand(logits.shape)
        self.node_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Cluster_acts
        logits = torch.log(self.cluster_act_probs)
        noise = torch.rand(logits.shape)
        self.cluster_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Define combined loss
        #loss =act_loss(self.node_act_probs, self.node_act_vec, self.cluster_act_probs, self.cluster_act_vec, self.adv) # get loss

        #loss.backward()
        #self.optimizer.step()
        
        return [self.node_act_probs, self.cluster_act_probs, self.node_acts, self.cluster_acts]    
        
    def optimize_net(self, x):
        # here I have to define the functioning of net (baiscally init)
        self.node_inputs, self.cluster_inputs, self.node_act_vec, self.cluster_act_vec, self.adv, self.entropy_weight  = x
        self.optimizer.zero_grad()
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.cluster_act_probs = self.orchestrate_network(
            self.node_inputs, self.gcn.outputs, self.cluster_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1], self.act_fn)

        # Draw action based on the probability
        logits = torch.log(self.node_act_probs)
        noise = torch.rand(logits.shape)
        self.node_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Cluster_acts
        logits = torch.log(self.cluster_act_probs)
        noise = torch.rand(logits.shape)
        self.cluster_acts = torch.topk(logits - torch.log(-torch.log(noise)), k=3).indices

        # Define combined loss
        loss =self.act_loss(self.node_act_probs, self.node_act_vec, self.cluster_act_probs, self.cluster_act_vec, self.adv) # get loss
        self.loss.backward()
        self.optimizer.step()
        return [self.node_act_probs, self.cluster_act_probs, self.node_acts, self.cluster_acts]
    def invoke_model(self, obs):
        # Invoke learning model
        node_inputs, cluster_inputs = self.translate_state(obs)
        self.gcn(node_inputs)
        node_act_probs, cluster_act_probs, node_acts, cluster_acts = \
            self.predict((node_inputs, cluster_inputs, self.gcn.outputs))
        
        return node_acts, cluster_acts, \
               node_act_probs, cluster_act_probs, \
               node_inputs, cluster_inputs    
               
               
    def get_action(self, obs):
        # Parse observation
        cluster_states, source_cluster, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs
        if len(frontier_nodes) == 0:
            return None, num_source_exec

        # Invoking the learning model
        node_act, cluster_act, \
        node_act_probs, cluster_act_probs, \
        node_inputs, cluster_inputs, \
        node_valid_mask, cluster_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_states_mat, state_summ_backward_map, \
        exec_map, cluster_states_changed = self.invoke_model(obs)

        if sum(node_valid_mask[0, :]) == 0:
            return None, num_source_exec

        # Should be valid
        assert node_valid_mask[0, node_act[0]] == 1

        # Parse node action
        node = action_map[node_act[0]]
        cluster_idx = cluster_states.index(node.cluster_state)

        # Should be valid
        assert cluster_valid_mask[0, cluster_act[0, cluster_idx] +
                                  len(self.executor_levels) * cluster_idx] == 1
        if node.cluster_state is source_cluster:
            agent_exec_act = self.executor_levels[
                                 cluster_act[0, cluster_idx]] - \
                             exec_map[node.cluster_state] + \
                             num_source_exec
        else:
            agent_exec_act = self.executor_levels[
                                 cluster_act[0, cluster_idx]] - exec_map[node.cluster_state]

        # Parse  action
        use_exec = min(
            node.num_tasks - node.next_task_idx -
            exec_commit.node_commit[node] -
            moving_executors.count(node),
            agent_exec_act, num_source_exec)
        return node, use_exec