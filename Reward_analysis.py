Node rewards
def get_gpg_reward(master_list):
    task_len = 0
    for master in master_list:
        for node in master.node_list:
            task_len += len(node.task_queue)
    reward = np.exp(-(task_len))
    return reward
	
	
	
rewards = np.array([r for (r, t) in zip(all_rewards, batch_time)])
cum_reward = discount(rewards, 1)

all_cum_reward.append(cum_reward)


def calculate_reward(master_list, cur_done, cur_undone):
    """
    Tailored MARDL for Decentralised request dispatch - Reward : Improve the longterm throughput while ensuring the load balancing at the edge
    
    [Function that returns rewards from environment given master nodes and the current tasks]

    Args:
        master_list ([Master Object list]): [Edge Access Point list containing nodes]
        cur_done ([list]): [list containing two elements for tasks done on both master nodes]
        cur_undone ([list]): [list containing two elements for tasks not done yet on both master nodes]

    Returns:
        reward [list]: [list of rewards for both master nodes]
    """
    weight = 1.0
    all_task = []
    fail_task = []
    for i in range(len(master_list)):
        all_task.append(float(cur_done[i] + cur_undone[i]))
        fail_task.append(float(cur_undone[i]))
 
    reward = []
    # The ratio of requests that violate delay requirements
    task_fail_rate = []
    
    for i in range(len(master_list)):
        if all_task[i] != 0:
            task_fail_rate.append(fail_task[i] / all_task[i])
        else:
            task_fail_rate.append(0)
    # The standard deviation of the CPU and memory usage
    
    use_rate_dict = {}
    for i in range(len(master_list)):
        use_rate_dict[i] = []
    
    for i, mstr in enumerate(master_list):
        for j in range(len(mstr.node_list)):
            use_rate_dict[i].append(mstr.node_list[j].cpu / mstr.node_list[j].cpu_max)
            use_rate_dict[i].append(mstr.node_list[j].mem / mstr.node_list[j].mem_max)

    standard_list_dict = {}
    for i in range(len(master_list)):
        standard_list_dict[i] = np.std(use_rate_dict[i], ddof=1)
    reward_dict = {}
    for i in range(len(master_list)):
        reward_dict[i] = math.exp(-task_fail_rate[i]) + weight * math.exp(-standard_list_dict[i])

    for r in range(len(master_list)):
        reward.append(reward_dict[r])
    # Immediate reward   e^(-lambda - weight_of_load_balancing *standard_deviation_of_cpu_memory)
    return reward