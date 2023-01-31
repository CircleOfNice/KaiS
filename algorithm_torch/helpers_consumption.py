
import matplotlib.pyplot as plt
def plot_list(data_list:list, title:str, x_label:str, y_label:str)->None:
    """Plot the given list

    Args:
        data_list (list) : CPU Parameters
        mem_list (list) : Memory Parameters
        task_list (list) :  Task Lists

    """

    plt.figure(figsize=(15,10))

    plt.plot(data_list)#throughput_list)
    plt.title(title)
    plt.xlabel(x_label)#"Number of Episodes")
    plt.ylabel(y_label)#"Throughput rate")
    plt.savefig('./plots/'+title + '.png')


def get_consumption_dictionary(master_list: list)-> dict:
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
    
    master_list_conusmption_dict = {}
    for i, mstr in enumerate(master_list):
        master_list_conusmption_dict[i] = {}
        node_consumption_dict = {}
        for j in range(len(mstr.node_list)):
            node_consumption_dict[j] = []
        master_list_conusmption_dict[i] = node_consumption_dict
    
    
    for i, mstr in enumerate(master_list):
        node_consumption_dict = master_list_conusmption_dict[i]
        for j in range(len(mstr.node_list)):
            node_consumption_dict[j].append(mstr.node_list[j].cpu)
            node_consumption_dict[j].append(mstr.node_list[j].cpu_max)
            node_consumption_dict[j].append(mstr.node_list[j].mem)
            node_consumption_dict[j].append(mstr.node_list[j].mem_max)
    return master_list_conusmption_dict

def create_base_info_dictionary(instance_dict):
    master_dict = {}

    for key in  instance_dict.keys():
        master_dict[key] = {}
        sub_node_dict = instance_dict[key]
        node_dict ={} 
        for sub_key in sub_node_dict.keys():
            
            node_dict[sub_key] = {}
            node_dict[sub_key]['cpu_load_dict'] =[] 
            node_dict[sub_key]['cpu_max_load_dict'] =[] 
            node_dict[sub_key]['mem_load_dict'] =[] 
            node_dict[sub_key]['mem_max_load_dict'] =[] 
        master_dict[key] = node_dict
    return master_dict


def get_data(consumption_list_overall, master_list):
    episode_list = []
    for episode, consumption_list_episode in enumerate(consumption_list_overall):

        instance = consumption_list_episode[0]
        master_dict = create_base_info_dictionary(instance)
        def append_to_base_dict(time_instance_dict, base_dict):

            for key in  time_instance_dict.keys():
                sub_node_dict = time_instance_dict[key]
                for sub_key in sub_node_dict.keys():
                    base_dict[key][sub_key]['cpu_load_dict'].append(time_instance_dict[key][sub_key][0])
                    base_dict[key][sub_key]['cpu_max_load_dict'].append(time_instance_dict[key][sub_key][1])
                    base_dict[key][sub_key]['mem_load_dict'].append(time_instance_dict[key][sub_key][2])
                    base_dict[key][sub_key]['mem_max_load_dict'].append(time_instance_dict[key][sub_key][3])
        for i in range(len(consumption_list_episode)):
            time_dict = consumption_list_episode[i]
            append_to_base_dict(time_dict, master_dict)
        episode_list.append(master_dict)
    return episode_list
    
def plot_episodes_individually(episode_data):
    for episode_number , data in enumerate(episode_data):
        for master_key in data.keys():
            for node_key in data[master_key].keys():
                for dat_key in data[master_key][node_key].keys():
                    title = 'episode_number_'+str(episode_number)+'_master_'+str(master_key)+'_node_'+str(node_key)+'_data_type_'+str(dat_key)
                    plot_list(data[master_key][node_key][dat_key], title = title, x_label= 'time', y_label = dat_key)
def get_base_aggregation_dict():
    aggregrate_data = {'cpu_load_dict': {}, 'cpu_max_load_dict': {}, 
    'mem_load_dict': {}, 'mem_max_load_dict': {},}
    return aggregrate_data
def aggregrate_data(consumption_list_as_is):
    
    def get_master_dict (consumption_instance):
        master_dict = {}
        episode_dict = consumption_instance[0]
        for master_key in episode_dict.keys():
            master_dict[master_key] = {}
            for node_key in episode_dict[master_key].keys():
                master_dict[master_key][node_key] = get_base_aggregation_dict()
        return master_dict
    list_of_master_dicts = []
    for episode_number , data in enumerate(consumption_list_as_is):
        master_dict = get_master_dict(consumption_list_as_is[0])   
        for time_slot in range(len(data)):
            consumpution_master_dict = data[time_slot]
            for consumption_master_key in consumpution_master_dict.keys():
                for consumption_node_key in consumpution_master_dict[consumption_master_key].keys():
                    consumption_node_data= consumpution_master_dict[consumption_master_key][consumption_node_key]
                    master_dict[consumption_master_key][consumption_node_key]['cpu_load_dict'][time_slot] = consumption_node_data[0]
                    master_dict[consumption_master_key][consumption_node_key]['cpu_max_load_dict'][time_slot]= consumption_node_data[1]
                    master_dict[consumption_master_key][consumption_node_key]['mem_load_dict'][time_slot]= consumption_node_data[2]
                    master_dict[consumption_master_key][consumption_node_key]['mem_max_load_dict'][time_slot]= consumption_node_data[3]
        list_of_master_dicts.append(master_dict)
    return list_of_master_dicts
