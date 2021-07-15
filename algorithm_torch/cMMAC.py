import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from copy import deepcopy


def fc(inp_dim, output_dim, act=nn.ReLU()):
    linear = nn.Linear(inp_dim, output_dim)
    nn.init.xavier_uniform_(linear.weight)
    linear.bias.data.fill_(0)
    fc_out = nn.Sequential(linear, act)
    return fc_out 


class Value_Model(nn.Module):
    def __init__(self, state_dim, inp_sizes = [128, 64, 32], act = nn.ReLU()):
        super().__init__()
        self.fc1 = fc(state_dim, inp_sizes[0], act=act)
        self.fc2 = fc(inp_sizes[0], inp_sizes[1], act=act)
        self.fc3 = fc(inp_sizes[1], inp_sizes[2], act=act)
        self.fc4 = fc(inp_sizes[2], 1, act=act)

    def forward(self, x):
        x = torch.from_numpy(x)
        #print(type(x), x.shape)
        x = self.fc1(x.float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim, inp_sizes = [128, 64, 32], act = nn.ReLU()):
        super().__init__()
        self.policy_state = state_dim
        self.action_dim = action_dim
        self.fc1 = fc(self.policy_state, inp_sizes[0], act=act)
        self.fc2 = fc(inp_sizes[0], inp_sizes[1], act=act)
        self.fc3 = fc(inp_sizes[1], inp_sizes[2], act=act)
        self.fc4 = fc(inp_sizes[2], self.action_dim, act=act)

    def forward(self, x):
        x = torch.from_numpy(x)
        
        x = self.fc1(x.float())
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x) 
        
        #self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)
        #print('x.shape : ',x.shape, x)
        return x 
        
class Estimator:
    def __init__(self, action_dim, state_dim, n_valid_node, summaries_dir=None):

        #self.sess = sess 
        self.n_valid_node = n_valid_node
        self.action_dim = action_dim
        self.state_dim = state_dim
        #self.scope = scope
        
        # Initial value for losses
        self.actor_loss = 0
        self.value_loss = 0
        self.entropy = 0
        self._build_value_model()
        self._build_policy()
        
        self.loss = self.actor_loss + .5 * self.value_loss - 10 * self.entropy
        
        self.neighbors_list = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
        
        
        # skipping tensorboard summaries_dir
        
    def _build_value_model(self):
        self.vm = Value_Model(self.state_dim, inp_sizes = [128, 64, 32])
        
        self.vm_criterion = nn.MSELoss()
        self.vm_optimizer = optim.Adam(self.vm.parameters(), lr=0.001)
        
        #return self.vm, self.vm_criterion
        
    def sm_prob(self, policy_net_output, neighbor_mask):
        neighbor_mask = torch.from_numpy(neighbor_mask)
        #print(policy_net_output, neighbor_mask)
        
        logits = policy_net_output +1 
        
        #print(type(logits), type(neighbor_mask))
        valid_logits = logits * neighbor_mask
        
        #softmaxprob = nn.Softmax(torch.log(valid_logits + 1e-8))
        softmaxprob = nn.Softmax()
        softmaxprob = softmaxprob(torch.log(valid_logits + 1e-8))
        #print(softmaxprob)
        return softmaxprob, logits, valid_logits
        
    def policy_net_loss(self, policy_net_output, neighbor_mask, tfadv ):
        softmaxprob, logits, valid_logits = self.sm_prob(policy_net_output, neighbor_mask)
        logsoftmaxprob = nn.functional.log_softmax(softmaxprob)

        neglogprob = - logsoftmaxprob * ACTION

        self.actor_loss = torch.mean(torch.sum(neglogprob * tfadv, axis=1))
        self.entropy = - torch.mean(softmaxprob * logsoftmaxprob)
        self.policy_loss = actor_loss - 0.01 * entropy
        
        return self.policy_loss 
        
    def _build_policy(self):
        self.pm = Policy_Model(self.state_dim, self.action_dim, inp_sizes = [128, 64, 32], act = nn.ReLU())
        self.pm_criterion = self.policy_net_loss
        self.pm_optimizer = optim.Adam(self.pm.parameters(), lr=0.001)
        #return self.pm, self.pm_criterion
        
    def action(self, s, ava_node, context, epsilon):
        #print('s.shape', s.shape)
        value_output = self.vm(s)#.flatten()
        #print('value_output.shape', value_output.shape)
        value_output = value_output.flatten()
        #print('value_output_new.shape', value_output.shape)
        action_tuple = []
        valid_prob = []

        # For training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []
        
        grid_ids = [x for x in range(self.n_valid_node)]
        #print('grid_ids.shape', len(grid_ids))
        self.valid_action_mask = np.zeros((self.n_valid_node, self.action_dim))
        
        #print('self.valid_action_mask.shape', self.valid_action_mask.shape)
        
        for i in range(len(ava_node)):
            for j in ava_node[i]:
                self.valid_action_mask[i][j] = 1
        curr_neighbor_mask = deepcopy(self.valid_action_mask)
        
        self.valid_neighbor_node_id = [[i for i in range(self.action_dim)], [i for i in range(self.action_dim)]]
        
        # compute policy probability.
        self.pm_out =self.pm(s)
        #print('s.shape', s.shape)
        #print(' self.pm_out.shape',  self.pm_out.shape)
        action_probs,_,_ = self.sm_prob( self.pm_out, curr_neighbor_mask) 
        #print(' action_probs.shape',  len(action_probs), action_probs.shape)
        #print()
        #print()
        #print(' action_probs : ',action_probs)
        curr_neighbor_mask_policy = []
        
        
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]
            #print(' action_prob.shape',  action_prob.shape, idx)
            #a=b
            # action probability for state value function
            action_prob = action_prob.detach().numpy()
            valid_prob.append(action_prob)
            if int(context[idx]) == 0:
                continue
                
            #print(self.action_dim, int(context[idx]), action_prob.shape)
            #print(action_prob / torch.sum(action_prob))
            #print(torch.sum(action_prob))
            #print(action_prob / np.sum(action_prob))
            
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob / np.sum(action_prob))
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            self.valid_neighbor_grid_id = self.valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)

                    # for training
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])
        #print(type(valid_prob))
        #print(valid_prob)
        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids
    
    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        # compute advantage
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.vm(next_state).flatten()
        for idx, next_state_id in enumerate(next_state_ids):
            temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[idx]
            advantage.append(temp_adv.detach().numpy())
        return advantage
    
    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        # compute targets
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.vm(next_state).flatten()
        
        #print(type(node_reward) , type(qvalue_next))
        for idx in np.arange(self.n_valid_node):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx] > 0]
            curr_grid_target = np.sum(
                grid_prob * (sum(node_reward) + gamma * sum(qvalue_next.detach().numpy())))
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])
        
        
        
# Don't see the point of initiallization , update_policy and update_value methods here
    
 
class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    # Take a batch of samples
    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0 
        
        
class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    # Put data in policy replay memory
    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    # Take a batch of samples
    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        
        
# Don't see the use of the class ModelParametersCopier
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0