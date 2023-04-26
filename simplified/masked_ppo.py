import gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from env_run import get_all_task_kubernetes
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from tensorboardX import SummaryWriter

#from new_gym_patching import CustomEnv
#from gym_env_patching_and_repeatable import CustomEnv
from new_gym_with_next_state_fixed import CustomEnv
import utils

class CustomLoggerCallback(BaseCallback):
    """Custom callback for logging different values from the environment.

    If we use multiple environments to train in parallel, we only log values of environment 0, assuming that the others will have similar values

    Currently logging the:
        - Action distribution
        - Number of invalid scheduling decisions
        - Number of times the cluster was at max capacity
    """

    def __init__(self, eval_env:gym.Env, verbose, log_freq:int, num_envs:int):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.num_envs = num_envs


    def _on_training_start(self) -> None:
        self.writer = SummaryWriter(logdir=self.logger.get_dir())
        self.action_dist_dict = {}
        self.master = self.eval_env.envs[0].master

        self.expected_action_distribution = utils.expected_action_distribution(node_num=total_nodes, use_mask=True)

        for idx in range(len(self.master.action_distribution)):
            self.action_dist_dict["a_" + str(idx)] = None

    def _on_step(self) -> bool:
        # Since self.num_timesteps is increased depending on the number of environments, we divide all values here by the num_envs
        if (self.num_timesteps // self.num_envs) % (self.log_freq // self.num_envs) == self.log_freq // self.num_envs - 1:

            # Logging the action distribution
            action_distribution = self.master.action_distribution

            if sum(action_distribution) > 0:
                for idx, value in enumerate(action_distribution):
                        self.action_dist_dict["a_" + str(idx)] = value / sum(action_distribution)
                self.writer.add_scalars("norm_action_distribution", self.action_dist_dict, global_step=self.num_timesteps)

            # KL div to the ideal load balanced approach
            kl_div = utils.action_distribution_kl_div(action_distribution, self.expected_action_distribution)
            self.writer.add_scalars("expected_action_dist_kl_div", {"expected_action_dist_kl_div": kl_div}, global_step=self.num_timesteps)

            # print(self.master.max_capacity_count)
            # Logging the relationship between the number of times the cluster was successfully brought to max capacity
            # vs the number of times a node get an invalid scheduling decision
            # Ideally this value should converge to a value close to 0
            invalid_decision_counter = self.master.max_capacity_count / self.master.invalid_decision_counter
            self.writer.add_scalars("max_capacity_pct", {"max_capacity_pct": invalid_decision_counter}, global_step=self.num_timesteps)

        return True
    
    
    def _on_rollout_end(self) -> None:
        # hist_values = self.eval_env.master.action_distribution
        # self.histogram_writer.add_histogram("action_distribution", hist_values, self.num_timesteps)
        return True


def mask_fn(env:CustomEnv) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    #return env.all_valid_action_mask()
    return env.ordered_valid_action_mask()
    # return env.valid_action_mask()


def create_custom_env(num_total_nodes:int, num_max_masked_nodes:int, data_list:list, train:bool=True):
    """Creator function for custom environments. Needed for stable baselines 3 wrappers

    Args:
        num_total_nodes (int): Total number of nodes in the system
        num_max_masked_nodes (int): Maximum allowed number of masked nodes. Could be one less than num_total_nodes.
        data_list (list): List with the data with tasks from kubernetes
        train (bool, optional): ?. Defaults to True.

    Returns:
        gym.Env: Returns an object implementing the gym interface
    """
    return CustomEnv(number_of_nodes=num_total_nodes, mask_nodes=num_max_masked_nodes, data=data_list, train=train)
    

path = os.path.join(os.getcwd(), 'Data', '2023_02_06_data', 'data_2.json')
result_list,_ = get_all_task_kubernetes(path)
total_nodes = 4
masked_nodes = total_nodes - 1

eval_freq = 50_000 # Number of timesteps after which to evaluate the models
num_envs = 16



# Need to first wrap the environment in all needed masks, and only then vectorize it
env_fn = lambda: ActionMasker(CustomEnv(total_nodes, masked_nodes, result_list), mask_fn)
custom_env = make_vec_env(env_fn, n_envs=num_envs)

# Use evaluation environment to calculate mean reward and find best model
eval_env = CustomEnv(total_nodes, masked_nodes, result_list)   # Initialize env
eval_env = ActionMasker(eval_env, mask_fn)  # Wrap to enable masking
eval_env = Monitor(eval_env)
eval_callback = EvalCallback(eval_env, best_model_save_path="best_model", log_path="logs",
                              eval_freq=eval_freq//num_envs, deterministic=True, render=False, n_eval_episodes=3, verbose=False)


action_dist_callback = CustomLoggerCallback(eval_env=custom_env, verbose=0, log_freq=eval_freq, num_envs=num_envs)

Episode_length = len(result_list[0])#[:5])
Episodes = 100 #5000
# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
total_reward_list = []

# policy_kwargs = dict(net_arch=[32, 64, 128, 256, 512, 1024])
#policy_kwargs = dict(net_arch=[16, 16])
policy_kwargs = None
model = MaskablePPO(MaskableActorCriticPolicy, custom_env, ent_coef=0.01, verbose=0, tensorboard_log="tensorboard_logs", policy_kwargs = policy_kwargs,
                    learning_rate=0.0003)#, verbose=True)

print(model.policy)
# Simple one shot training 
# custom_env.set_train_param(True)
# model.learn(total_timesteps=(Episode_length-1)*Episodes, progress_bar=True, callback=[eval_callback, custom_callback])
model.learn(total_timesteps=(Episode_length-1)*Episodes, progress_bar=True, callback=[eval_callback, action_dist_callback])
model.save(os.path.join('models','PPO2', "final_ppo_model.zip"))
# print(' len(customenv.reward_list) : ', sum(custom_env.reward_list ), len(custom_env.reward_list))
# sum_reward = sum(custom_env.reward_list )/ len(custom_env.reward_list)


# model.logger.record("histogram", eval_env.master.action_distribution)
# max_reward_model = 0
# for epi in tqdm(range(Episodes)):
#     customenv.set_train_param(True)
#     model.learn(total_timesteps=(Episode_length-1))
#     print('done PPO2 learn')
#     print(' len(customenv.reward_list) : ', sum(customenv.reward_list ), len(customenv.reward_list))
#     sum_reward = sum(customenv.reward_list )/ len(customenv.reward_list)
#     print(f'Episodes : {epi} average reward till episode {sum_reward}')
#     if sum_reward>max_reward_model:
#         model.save(os.path.join('models','PPO2', str(sum_reward)+'_'+str(total_nodes) +'_node_'+ str(masked_nodes)+'_mask.zip'))
#         max_reward_model = sum_reward
#     total_reward_list.append(sum_reward)
#     plt.plot(total_reward_list)
#     plt.title('Reward PPO2 Over time')
#     plt.ylabel('Reward Accumulated ')
#     plt.xlabel('Episodes')
#     #plt.show()
#     plt.savefig('current_progress2.png',  dpi=300)
   
# print('total_reward_list : ', total_reward_list)
# model.save("PPO2_Final_" +'_'+str(total_nodes) +'_node_'+ str(masked_nodes)+'_mask.zip')
# plt.plot(total_reward_list)
# plt.title('Final Reward PPO2 Over time')
# plt.ylabel('Reward Accumulated ')
# plt.xlabel('Episodes')
# plt.show()
