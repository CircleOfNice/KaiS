import gym
import joblib
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

            avg_mem_utilisation_ratios = sum(self.master.avg_mem_utilisation_ratios)/len(self.master.avg_mem_utilisation_ratios)
            avg_cpu_utilisation_ratios = sum(self.master.avg_cpu_utilisation_ratios)/len(self.master.avg_cpu_utilisation_ratios)
            avg_std_cpu = sum(self.master.avg_std_cpu)/len(self.master.avg_std_cpu)
            avg_std_mem = sum(self.master.avg_std_mem)/len(self.master.avg_std_mem)
            avg_ent_cpu = sum(self.master.avg_ent_cpu)/len(self.master.avg_ent_cpu)
            avg_ent_mem = sum(self.master.avg_ent_mem)/len(self.master.avg_ent_mem)
            avg_rel_entropy_per_node= sum(self.master.avg_rel_entropy_per_node)/len(self.master.avg_rel_entropy_per_node)
            avg_coeff_cpu = sum(self.master.avg_coeff_cpu)/len(self.master.avg_coeff_cpu)
            avg_coeff_mem = sum(self.master.avg_coeff_mem)/len(self.master.avg_coeff_mem)
            
            # statistics_dict = {"avg_mem_utilisation_ratios": avg_mem_utilisation_ratios, "avg_cpu_utilisation_ratios": avg_cpu_utilisation_ratios,
            #                    "avg_std_cpu": avg_std_cpu, "avg_std_mem": avg_std_mem,
            #                    "avg_ent_cpu": avg_ent_cpu, "avg_ent_mem": avg_ent_mem,
            #                    "avg_coeff_cpu": avg_coeff_cpu, "avg_coeff_mem": avg_coeff_mem,
            #                    "avg_rel_entropy_per_node": avg_rel_entropy_per_node
            #                    }
            # self.writer.add_scalars("Statistics", statistics_dict, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_cpu_utilisation_ratios", {"avg_cpu_utilisation_ratios": avg_cpu_utilisation_ratios}, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_std_cpu", {"avg_std_cpu": avg_std_cpu}, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_std_mem", {"avg_std_mem": avg_std_mem}, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_ent_cpu", {"avg_ent_cpu": avg_ent_cpu}, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_ent_mem", {"avg_ent_mem": avg_ent_mem}, global_step=self.num_timesteps)
            # # self.writer.add_scalars("avg_rel_entropy_per_node", {"avg_rel_entropy_per_node": avg_rel_entropy_per_node}, global_step=self.num_timesteps)
            
            
            # statistics_dict = {"avg_mem_utilisation_ratios": avg_mem_utilisation_ratios, "avg_cpu_utilisation_ratios": avg_cpu_utilisation_ratios,
            #                    "avg_std_cpu": avg_std_cpu, "avg_std_mem": avg_std_mem,
            #                    "avg_ent_cpu": avg_ent_cpu, "avg_ent_mem": avg_ent_mem,
            #                    "avg_coeff_cpu": avg_coeff_cpu, "avg_coeff_mem": avg_coeff_mem,
            #                    "avg_rel_entropy_per_node": avg_rel_entropy_per_node
            #                    }
            
            
            self.writer.add_scalars("Utilisation Ratio/Average Memory Utilisation Ratio", {"avg_mem_utilisation_ratios": avg_mem_utilisation_ratios}, global_step=self.num_timesteps)
            self.writer.add_scalars("Utilisation Ratio/Average CPU Utilisation Ratio", {"avg_cpu_utilisation_ratios": avg_cpu_utilisation_ratios}, global_step=self.num_timesteps)
            
            self.writer.add_scalars("Standard Deviation/Average Standard Deviation of CPU Usage", {"avg_std_cpu": avg_std_cpu}, global_step=self.num_timesteps)
            self.writer.add_scalars("Standard Deviation/Average Standard Deviation of Memory Usage", {"avg_std_mem": avg_std_mem}, global_step=self.num_timesteps)
            
            self.writer.add_scalars("Entropy/Average Entropy of CPU Usage", {"avg_ent_cpu": avg_ent_cpu}, global_step=self.num_timesteps)
            self.writer.add_scalars("Entropy/Average Entropy of Memory Usage", {"avg_ent_mem": avg_ent_mem}, global_step=self.num_timesteps)
            self.writer.add_scalars("Entropy/avg_rel_entropy_per_node", {"avg_rel_entropy_per_node": avg_rel_entropy_per_node}, global_step=self.num_timesteps)
            
            self.writer.add_scalars("Coefficient of Correlation/Average Coefficient of Correlation for CPU Usage", {"avg_coeff_cpu": avg_coeff_cpu}, global_step=self.num_timesteps)
            self.writer.add_scalars("Coefficient of Correlation/Average Coefficient of Correlation for Memory Usage", {"avg_coeff_mem": avg_coeff_mem}, global_step=self.num_timesteps)
            
            
            
            
            # print(self.master.max_capacity_count)
            # Logging the relationship between the number of times the cluster was successfully brought to max capacity
            # vs the number of times a node get an invalid scheduling decision
            # Ideally this value should converge to a value close to 0
            invalid_decision_counter = self.master.max_capacity_count / self.master.invalid_decision_counter

            dict_ = {
                "max_capacity_pct": invalid_decision_counter,
            }
            self.writer.add_scalars("max_capacity_pct", dict_ , global_step=self.num_timesteps)

        return True
    
    
    def _on_rollout_end(self) -> None:
        # hist_values = self.eval_env.master.action_distribution
        # self.histogram_writer.add_histogram("action_distribution", hist_values, self.num_timesteps)
        return True


def mask_fn(env:CustomEnv) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.all_valid_action_mask()
    # return env.ordered_valid_action_mask()
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
custom_env = VecNormalize(custom_env)

eval_env = make_vec_env(env_fn, n_envs=1)
eval_env = Monitor(eval_env)
eval_env = VecNormalize(eval_env)

eval_callback = EvalCallback(eval_env, best_model_save_path="best_model", log_path="logs",
                              eval_freq=eval_freq//num_envs, deterministic=True, render=False, n_eval_episodes=5, verbose=False)


action_dist_callback = CustomLoggerCallback(eval_env=custom_env, verbose=0, log_freq=eval_freq, num_envs=num_envs)

episode_length = len(result_list[0])
num_episodes = 100

total_reward_list = []

# policy_kwargs = dict(net_arch=[32, 64, 128, 256, 512, 1024])
# policy_kwargs = dict(net_arch=[32, 32])
policy_kwargs = None
lr = 0.0003
enf_coef = 0.01
model = MaskablePPO(MaskableActorCriticPolicy, custom_env, ent_coef=enf_coef, verbose=0, tensorboard_log="tensorboard_logs", policy_kwargs = policy_kwargs,
                    learning_rate=lr)#, verbose=True)

print(model.policy)
model.learn(total_timesteps=(episode_length-1)*num_episodes, progress_bar=True, callback=[eval_callback, action_dist_callback])
model.save(os.path.join('models','PPO2', "final_ppo_model.zip"))
custom_env.save(os.path.join("models", "PPO2", "final_env.zip"))

# Logging hyperparameters
# Metric dict must not be empty
hparams = {
    "num_episodes": num_episodes,
    "episode_length": episode_length,
    "num_nodes": total_nodes,
    "num_masked_nodes": masked_nodes,
    "num_envs": num_envs,
}
if policy_kwargs:
    for idx, layer_n in enumerate(policy_kwargs["net_arch"]):
        hparams["layer_" + str(idx)] = layer_n

writer = SummaryWriter(logdir=action_dist_callback.logger.get_dir())
# Metric dict must not be empty
writer.add_hparams(hparam_dict=hparams, metric_dict={"empty": 1})
