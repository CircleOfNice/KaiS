import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import bisect
from algorithm.gcn import GraphCNN
from algorithm.gsn import GraphSNN
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def discount(x, gamma):
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out


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
        print()
        print()
        print()
        print('compute_orchestrate_loss :')
        node_inputs = exp['node_inputs']
        cluster_inputs = exp['cluster_inputs']
        node_act_vec = exp['node_act_vec']
        cluster_act_vec = exp['cluster_act_vec']
        adv = batch_adv[ba_start: ba_end, :]
        
        node_inputs = np.array(node_inputs)
        print('node_inputs', node_inputs.shape)
        cluster_inputs = np.array(cluster_inputs)
        
        print('cluster_inputs', cluster_inputs.shape)
        node_act_vec = np.array(node_act_vec)
        #
        # print('node_act_vec', node_act_vec.shape)
        cluster_act_vec = np.array(cluster_act_vec)
        #print('cluster_act_vec', cluster_act_vec.shape)
        print('cluster_inputs', cluster_inputs.shape)
        node_act_vec = np.array(node_act_vec)
        print('node_act_vec', node_act_vec.shape)
        cluster_act_vec = np.array(cluster_act_vec)
        print('cluster_act_vec', cluster_act_vec.shape)
        print()
        print()
        print()
        #m=k
        loss = orchestrate_agent.update_gradients(
            node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv, entropy_weight)
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

    # Compute gradients
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv, entropy_weight)
    entropy_weight = decrease_var(entropy_weight,
                                  entropy_weight_min, entropy_weight_decay)
    return entropy_weight, loss


class Agent(object):
    def __init__(self):
        pass


def expand_act_on_state(state, sub_acts):
    # Expand the state
    batch_size = tf.shape(state)[0]
    num_nodes = tf.shape(state)[1]
    num_features = state.shape[2].value
    expand_dim = len(sub_acts)

    # Replicate the state
    state = tf.tile(state, [1, 1, expand_dim])
    state = tf.reshape(state,
                       [batch_size, num_nodes * expand_dim, num_features])

    # Prepare the appended actions
    sub_acts = tf.constant(sub_acts, dtype=tf.float32)
    sub_acts = tf.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = tf.tile(sub_acts, [1, 1, num_nodes])
    sub_acts = tf.reshape(sub_acts, [1, num_nodes * expand_dim, 1])
    sub_acts = tf.tile(sub_acts, [batch_size, 1, 1])

    # Concatenate expanded state with sub-action features
    concat_state = tf.concat([state, sub_acts], axis=2)

    return concat_state


def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


class OrchestrateAgent(Agent):
    def __init__(self, sess, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='orchestrate_agent'):

        Agent.__init__(self)
        self.sess = sess
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])
        self.cluster_inputs = tf.placeholder(tf.float32, [None, self.cluster_input_dim])
        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)
        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)
        self.summary_n = 50
        # Map gcn_outputs and raw_inputs to action probabilities
        self.node_act_probs, self.cluster_act_probs = self.orchestrate_network(
            self.node_inputs, self.gcn.outputs, self.cluster_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1], self.act_fn)

        print('self.node_inputs,  self.cluster_inputs :, ',self.node_inputs.shape,  self.cluster_inputs.shape )
        
        print()
        print()
        print('self.node_act_probs.shape, cluster_act_probs.shape : ', self.node_act_probs.shape,  self.cluster_act_probs.shape)
        
        # Draw action based on the probability
        tf_node_act_probs = tf.Print(self.node_act_probs, [self.node_act_probs], 'self.node_act_probs : ', summarize=self.summary_n)
        
        logits = tf.log(tf_node_act_probs)
        
        tf_logits = tf.Print(logits, [logits], '1 logits  tf_logits: ', summarize=self.summary_n)
        print()
        print('logits.shape : ', logits.shape)
        noise = tf.random_uniform(tf.shape(tf_logits))
        
        tf_noise = tf.Print(noise, [noise], '1 noise tf_noise: ', summarize=self.summary_n)
        print()
        print('noise.shape : ', noise.shape)
        self.node_acts = tf.nn.top_k(logits - tf.log(-tf.log(tf_noise)), k=3).indices

        print()
        print('self.node_acts.shape : ', self.node_acts.shape)
        
        self.tf_node_acts = tf.Print(self.node_acts, [self.node_acts], 'self.tf_node_acts, self.node_acts : ', summarize=self.summary_n)
        
        # Cluster_acts
        
        tf_cluster_act_probs = tf.Print(self.cluster_act_probs, [self.cluster_act_probs], 'self.cluster_act_probs : ', summarize=self.summary_n)
        logits = tf.log(self.cluster_act_probs)
        print()
        print()
        print('logits.shape : ', logits.shape)
        
        tf_logits_2 = tf.Print(logits, [logits], '2 logits tf_logits_2 : ', summarize=self.summary_n)
        noise_2 = tf.random_uniform(tf.shape(tf_logits_2))
        print()
        print()
        print('noise.shape : ', noise_2.shape)
        
        tf_noise_2 = tf.Print(noise_2, [noise_2], '2 noise tf_noise_2 : ', summarize=self.summary_n)
        self.cluster_acts = tf.nn.top_k(logits - tf.log(-tf.log(tf_noise_2)), k=3).indices
        
        self.tf_cluster_acts = tf.Print(self.cluster_acts, [self.cluster_acts], 'self.cluster_acts tf_cluster_acts : ', summarize=self.summary_n)
        print()
        print()
        print('self.cluster_acts.shape : ', self.cluster_acts.shape)
        
        #print('self.node_acts,  self.cluster_acts :, ',self.node_acts.shape,  self.cluster_acts.shape )
        
        # Selected action
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        self.cluster_act_vec = tf.placeholder(tf.float32, [None, None, None])
        print('self.node_act_probs,  self.node_act_vec :, ',tf_node_act_probs.shape,  self.node_act_vec.shape )
        
        tf_node_act_vec = tf.Print(self.node_act_vec, [self.node_act_vec], 'self.node_act_vec tf_node_act_vec : ', summarize=self.summary_n)
        #print('self.node_act_vec : ', self.node_act_vec.get_shape())
        # ASdvantage
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # Decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())
        
        node_prod = tf.multiply(
            tf_node_act_probs, tf_node_act_vec)#self.node_act_vec)
        print()
        print()
        print('prod.shape : ', node_prod.shape)
        
        tf_node_prod = tf.Print(node_prod, [node_prod], 'node_prod tf_node_prod: ', summarize=self.summary_n)
        # Action probability
        self.selected_node_prob = tf.reduce_sum(tf_node_prod,
            reduction_indices=1, keep_dims=True)
        
        tf_selected_node_prob = tf.Print(self.selected_node_prob, [self.selected_node_prob], 'self.selected_node_prob tf_selected_node_prob: ', summarize=self.summary_n)
        print()
        print()
        print('selected_node_prob :  ', self.selected_node_prob.shape)
        #a=b
        
        print('self.cluster_act_probs, cluster_act_vec : ', self.cluster_act_probs.shape, self.cluster_act_vec.shape)
        select_cluster_prod = tf.multiply(
            tf_cluster_act_probs, self.cluster_act_vec)
        print('')
        print('')
        print('select_cluster_prod :', select_cluster_prod.shape)
        
        
        tf_select_cluster_prod = tf.Print(select_cluster_prod, [select_cluster_prod], 'select_cluster_prod tf_select_cluster_prod : ', summarize=self.summary_n)
        
        sum_cluster_1 = tf.reduce_sum(tf_select_cluster_prod,
            reduction_indices=2)
        
        print()
        print()
        print('sum_cluster_1 :', sum_cluster_1.shape)
        tf_sum_cluster_1 = tf.Print(sum_cluster_1, [sum_cluster_1], 'sum_cluster_1 tf_sum_cluster_1 : ', summarize=self.summary_n)
        
        self.selected_cluster_prob = tf.reduce_sum(tf_sum_cluster_1, reduction_indices=1, keep_dims=True)
        print()
        print()
        print('self.selected_cluster_prob :', self.selected_cluster_prob.shape)
        
        tf_selected_cluster_prob = tf.Print(self.selected_cluster_prob , [self.selected_cluster_prob ], 'self.selected_cluster_prob tf_selected_cluster_prob : ', summarize=self.summary_n)
        #print('self.selected_node_prob.shape, node_act_probs.shape, node_act_vec.shape : ', self.selected_node_prob.shape,  self.node_act_probs.shape, self.node_act_vec.shape)
        #a=b
        #print('node_act_probs.shape, node_act_vec.shape) :', self.node_act_probs.shape, self.node_act_vec.shape)
        #print('self.cluster_act_probs, cluster_act_vec : ', self.cluster_act_probs.shape, self.cluster_act_vec.shape)
        #print('selected_cluster_prob.shape, selected_node_prob.shape : ', self.selected_cluster_prob.shape, self.selected_node_prob.shape)
        # Orchestrate loss due to advantge
        
        
        torch_log = tf.log(tf_selected_node_prob * tf_selected_cluster_prob + \
                   self.eps)
        tf_torch_log = tf.Print(torch_log, [torch_log], 'torch_log tf_torch_log : ', summarize=self.summary_n)
        print()
        print()
        print('torch_log :', torch_log.shape)
        tf_adv = tf.Print(self.adv, [self.adv], 'self.adv tf_adv: ', summarize=self.summary_n)
        torch_log_adv_mul = tf.multiply(tf_torch_log, -tf_adv)

        print()
        print()
        print('torch_log_adv_mul :', torch_log_adv_mul.shape)
        tf_torch_log_adv_mul = tf.Print(torch_log_adv_mul, [torch_log_adv_mul], 'torch_log_adv_mul tf_torch_log_adv_mul: ', summarize=self.summary_n)
        
        self.adv_loss = tf.reduce_sum(tf_torch_log_adv_mul)

        print('')
        print('self.adv_loss :', self.adv_loss.shape)
        
        tf_adv_loss = tf.Print(self.adv_loss, [self.adv_loss], 'self.adv_loss tf_adv_loss: ', summarize=self.summary_n)

        # Node_entropy
        torch_log_entropy = tf.log(tf_node_act_probs + self.eps)
        print('')
        print('')
        print('torch_log_entropy :', torch_log_entropy.shape)
        
        tf_torch_log_entropy = tf.Print(torch_log_entropy, [torch_log_entropy], 'torch_log_entropy tf_torch_log_entropy: ', summarize=self.summary_n)
        
        torch_mul_dimension = tf.multiply(
            tf_node_act_probs, tf_torch_log_entropy)

        print('')
        print('')
        print('torch_mul_dimension :', torch_mul_dimension.shape)
        tf_torch_mul_dimension = tf.Print(torch_mul_dimension, [torch_mul_dimension], 'self.torch_mul_dimension tf_torch_mul_dimension : ', summarize=self.summary_n)
        
        
        self.node_entropy = tf.reduce_sum(tf_torch_mul_dimension)
        print('')
        print('')
        print('self.node_entropy :', self.node_entropy.shape)
        
        tf_node_entropy= tf.Print(self.node_entropy , [self.node_entropy ], 'self.node_entropy tf_node_entropy : ', summarize=self.summary_n)
        # Entropy loss
        self.entropy_loss = tf_node_entropy  # + self.cluster_entropy

        # Normalize entropy
        
        torch_log_norm = tf.log(float(len(self.executor_levels)))
        print('')
        print('')
        print('torch_log_norm :', torch_log_norm.shape)
        
        tf_torch_log_norm=  tf.Print(torch_log_norm , [torch_log_norm], 'torch_log_norm  tf_torch_log_norm : ', summarize=self.summary_n)
        
        denom = (tf.log(tf.cast(tf.shape(tf_node_act_probs)[1], tf.float32)) +tf_torch_log_norm)
        
        tf_denom=  tf.Print(denom , [denom], 'denom  : ', summarize=20)
        
        self.entropy_loss /= tf_denom
            #(tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)) + \
            # tf.log(float(len(self.executor_levels))))
        tf_entropy_loss = tf.Print(self.entropy_loss, [self.entropy_loss], 'self.entropy_loss tf_entropy_loss : ', summarize=self.summary_n)
            
        # Define combined loss
        self.act_loss = tf_adv_loss + self.entropy_weight * tf_entropy_loss

        # Get training parameters
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # Operations for setting network parameters
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # Orchestrate gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # Adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # Orchestrate optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # Apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate). \
            apply_gradients(zip(self.act_gradients, self.params))

        # Network paramter saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)
        self.sess.run(tf.global_variables_initializer())

    def orchestrate_network(self, node_inputs, gcn_outputs, cluster_inputs,
                            gsn_dag_summary, gsn_global_summary, act_fn):
        
        batch_size = 1
        node_inputs_reshape = tf.reshape(
            node_inputs, [batch_size, -1, self.node_input_dim])
        cluster_inputs_reshape = tf.reshape(
            cluster_inputs, [batch_size, -1, self.cluster_input_dim])
        gcn_outputs_reshape = tf.reshape(
            gcn_outputs, [batch_size, -1, self.output_dim])
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_cluster = tf.tile(
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])
        
        with tf.variable_scope(self.scope):
            merge_node = tf.concat([
                node_inputs_reshape, gcn_outputs_reshape
            ], axis=2)
            
            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)
            #print()
            #print() 
            #print('node_outputs before: ', node_outputs.shape)
            # Reshape the output dimension
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # Do softmax
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)
            merge_cluster = tf.concat([cluster_inputs_reshape, ], axis=2)
            #print()
            #print() 
            #print('merge_cluster.shape ', merge_cluster.shape)
            expanded_state = expand_act_on_state(
                merge_cluster, [l / 50.0 for l in self.executor_levels])
            cluster_hid_0 = tl.fully_connected(expanded_state, 32, activation_fn=act_fn)
            cluster_hid_1 = tl.fully_connected(cluster_hid_0, 16, activation_fn=act_fn)
            cluster_hid_2 = tl.fully_connected(cluster_hid_1, 8, activation_fn=act_fn)
            cluster_outputs = tl.fully_connected(cluster_hid_2, 1, activation_fn=None)
            
            #print()
            #print() 
            #print('cluster_outputs.shape as is', cluster_outputs.shape)
            
            cluster_outputs = tf.reshape(cluster_outputs, [batch_size, -1])
            cluster_outputs = tf.reshape(
                cluster_outputs, [batch_size, -1, len(self.executor_levels)])

            
            #print()
            #print() 
            #print('cluster_outputs.shape after reshaping : ', cluster_outputs.shape)  
            # Do softmax
            tf_cluster_outputs = tf.Print(cluster_outputs, [cluster_outputs], 'self.cluster_acts tf_cluster_acts : ', summarize=50)
            cluster_outputs = tf.nn.softmax(tf_cluster_outputs, dim=-1)
            #a=b
            #print()
            #print()
            print('node_outputs.shape, cluster_outputs.shape : ',node_outputs.shape, cluster_outputs.shape)
            return node_outputs, cluster_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(self.act_gradients + [self.lr_rate], gradients + [lr_rate])})

    def define_params_op(self):
        # Define operations
        input_params = []
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        return self.sess.run([self.gsn.summaries],
                             feed_dict={i: d for i, d in
                                        zip([self.node_inputs] + self.gsn.summ_mats, [node_inputs] + summ_mats)})

    def get_params(self):
        return self.sess.run(self.params)

    def save_model(self, file_path):
        self.saver.save(self.sess, file_path)

    def update_gradients(self, node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv, entropy_weight):
        node_inputs = node_inputs[0]
        cluster_inputs = cluster_inputs[0]
        node_act_vec = node_act_vec[0]
        cluster_act_vec = cluster_act_vec[0]
        
        print('update gradients node_inputs.shape, cluster_inputs.shape, node_act_vec.shape, cluster_act_vec.shape')
        print(node_inputs.shape, cluster_inputs.shape, node_act_vec.shape, cluster_act_vec.shape)
        #a=b
        entropy_weight = entropy_weight
        self.sess.run(self.act_opt, feed_dict={i: d for i, d in zip(
            [self.node_inputs] + [self.cluster_inputs] + [self.node_act_vec] + [
                self.cluster_act_vec] + [self.adv] + [self.entropy_weight] + [self.lr_rate],
            [node_inputs] + [cluster_inputs] + [node_act_vec] + [cluster_act_vec] + [
                adv] + [entropy_weight] + [0.001])})

        loss_ = self.sess.run(self.act_loss, feed_dict={i: d for i, d in zip(
            [self.node_inputs] + [self.cluster_inputs] + [self.node_act_vec] + [
                self.cluster_act_vec] + [self.adv] + [self.entropy_weight],
            [node_inputs] + [cluster_inputs] + [node_act_vec] + [cluster_act_vec] + [
                adv] + [entropy_weight])})
        
        #tvars = tf.trainable_variables()
        #tvars_vals = self.sess.run(tvars)

        #for var, val in zip(tvars, tvars_vals):
        #
        # print(var.name, val.shape)  # Prints the name of the variable alongside its value.
        #tf.print(' after update self.cluster_act_probs, cluster_act_vec : ', self.cluster_act_probs.shape, self.cluster_act_vec.shape)
        #a=b
        return loss_

    def predict(self, node_inputs, cluster_inputs):
        return self.sess.run([self.node_act_probs, self.cluster_act_probs, self.tf_node_acts, self.tf_cluster_acts],
                             feed_dict={i: d for i, d in zip([self.node_inputs] + [self.cluster_inputs],
                                                             [node_inputs] + [cluster_inputs])})

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

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

    def invoke_model(self, obs):
        # Invoke learning model
        node_inputs, cluster_inputs = self.translate_state(obs)
        node_act_probs, cluster_act_probs, node_acts, cluster_acts = \
            self.predict(node_inputs, cluster_inputs)
        #a=b
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

