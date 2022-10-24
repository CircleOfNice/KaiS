import pickle, gzip
import matplotlib.pyplot as plt
import numpy as np
import os

name = 'torch_out_timefull_randomisation_orchestration_no_randomisation_1633093889.9293108'
with gzip.open("result/torch_out_timefull_randomisation_orchestration_no_randomisation_1633093889.9293108.obj", 'rb') as fp:
        record = pickle.load(fp)


with gzip.open("result/throughputfull_randomisation_orchestration_no_randomisation_1633093889.9293108.obj", 'rb') as fp:
        throughput = pickle.load(fp)     
        
           
print(len(throughput))

print(record[0])

print()
num_episodes = 50
len_episodes = 5000
imd_rewards_1 = []
imd_rewards_2 = []
cur_done_1 = []
cur_done_2 = []
cur_undone_1 = []
cur_undone_2 = []

num_nodes = 3
num_eaps = 2

node_1_eap_1_cpu = []
node_2_eap_1_cpu = []
node_3_eap_1_cpu = []

node_1_eap_1_mem = []
node_2_eap_1_mem = []
node_3_eap_1_mem = []

node_1_eap_2_cpu = []
node_2_eap_2_cpu = []
node_3_eap_2_cpu = []

node_1_eap_2_mem = []
node_2_eap_2_mem = []
node_3_eap_2_mem = []


for i in range(len(record)):
    
    node_1_eap_1_cpu.append(record[i][0].node_list[0].cpu/record[i][0].node_list[0].cpu_max)
    node_2_eap_1_cpu.append(record[i][0].node_list[1].cpu/record[i][0].node_list[1].cpu_max)
    node_3_eap_1_cpu.append(record[i][0].node_list[2].cpu/record[i][0].node_list[2].cpu_max)
    
    node_1_eap_1_mem.append(record[i][0].node_list[0].mem/record[i][0].node_list[0].mem_max)
    node_2_eap_1_mem.append(record[i][0].node_list[1].mem/record[i][0].node_list[1].mem_max)
    node_3_eap_1_mem.append(record[i][0].node_list[2].mem/record[i][0].node_list[2].mem_max)
    
    node_1_eap_2_cpu.append(record[i][1].node_list[0].cpu/record[i][1].node_list[0].cpu_max)
    node_2_eap_2_cpu.append(record[i][1].node_list[1].cpu/record[i][1].node_list[1].cpu_max)
    node_3_eap_2_cpu.append(record[i][1].node_list[2].cpu/record[i][1].node_list[2].cpu_max)
    
    node_1_eap_2_mem.append(record[i][1].node_list[0].mem/record[i][1].node_list[0].mem_max)
    node_2_eap_2_mem.append(record[i][1].node_list[1].mem/record[i][1].node_list[1].mem_max)
    node_3_eap_2_mem.append(record[i][1].node_list[2].mem/record[i][1].node_list[2].mem_max)
    
    imd_rewards_1.append(record[i][4][0])
    imd_rewards_2.append(record[i][4][1])
    
    cur_done_1.append(record[i][2][0])
    cur_done_2.append(record[i][2][1])
    
    cur_undone_1.append(record[i][3][0])
    cur_undone_2.append(record[i][3][1])
    
print(len(imd_rewards_1), len(cur_done_1), len(cur_undone_1))


ep_node_1_eap_1_cpu = []
ep_node_2_eap_1_cpu = []
ep_node_3_eap_1_cpu = []

ep_node_1_eap_1_mem = []
ep_node_2_eap_1_mem = []
ep_node_3_eap_1_mem = []

ep_node_1_eap_2_cpu = []
ep_node_2_eap_2_cpu = []
ep_node_3_eap_2_cpu = []

ep_node_1_eap_2_mem = []
ep_node_2_eap_2_mem = []
ep_node_3_eap_2_mem = []

ep_imd_rewards_1 = []
ep_imd_rewards_2 = []
ep_cur_done_1 = []
ep_cur_done_2 = []
ep_cur_undone_1 = []
ep_cur_undone_2 = []

for i in range(num_episodes):
    ep_imd_rewards_1.append(imd_rewards_1[i*len_episodes:(i+1)*len_episodes])
    ep_imd_rewards_2.append(imd_rewards_2[i*len_episodes:(i+1)*len_episodes])
    
    ep_cur_done_1.append(cur_done_1[i*len_episodes:(i+1)*len_episodes])
    ep_cur_done_2.append(cur_done_2[i*len_episodes:(i+1)*len_episodes])
    
    ep_cur_undone_1.append(cur_undone_1[i*len_episodes:(i+1)*len_episodes])
    ep_cur_undone_2.append(cur_undone_2[i*len_episodes:(i+1)*len_episodes])
    
    
    
    ep_node_1_eap_1_cpu.append(node_1_eap_1_cpu[i*len_episodes:(i+1)*len_episodes])
    ep_node_2_eap_1_cpu.append(node_2_eap_1_cpu[i*len_episodes:(i+1)*len_episodes])
    ep_node_3_eap_1_cpu.append(node_3_eap_1_cpu[i*len_episodes:(i+1)*len_episodes])
    
    ep_node_1_eap_1_mem.append(node_1_eap_1_mem[i*len_episodes:(i+1)*len_episodes])
    ep_node_2_eap_1_mem.append(node_2_eap_1_mem[i*len_episodes:(i+1)*len_episodes])
    ep_node_3_eap_1_mem.append(node_3_eap_1_mem[i*len_episodes:(i+1)*len_episodes])
    
    ep_node_1_eap_2_cpu.append(node_1_eap_2_cpu[i*len_episodes:(i+1)*len_episodes])
    ep_node_2_eap_2_cpu.append(node_2_eap_2_cpu[i*len_episodes:(i+1)*len_episodes])
    ep_node_3_eap_2_cpu.append(node_3_eap_2_cpu[i*len_episodes:(i+1)*len_episodes])
    
    ep_node_1_eap_2_mem.append(node_1_eap_2_mem[i*len_episodes:(i+1)*len_episodes])
    ep_node_2_eap_2_mem.append(node_2_eap_2_mem[i*len_episodes:(i+1)*len_episodes])
    ep_node_3_eap_2_mem.append(node_3_eap_2_mem[i*len_episodes:(i+1)*len_episodes])
    
    
print(len(ep_imd_rewards_1), len(ep_cur_done_1), len(ep_cur_undone_1))

#for re in ep_imd_rewards_1:
#    print(len(re))

ep_imd_rewards_1 = np.stack(ep_imd_rewards_1)
ep_imd_rewards_2 = np.stack(ep_imd_rewards_2)

ep_cur_done_1 = np.stack(ep_cur_done_1)
ep_cur_done_2 = np.stack(ep_cur_done_2)

ep_cur_undone_1 = np.stack(ep_cur_undone_1)
ep_cur_undone_2 = np.stack(ep_cur_undone_2)


ep_node_1_eap_1_cpu = np.stack(ep_node_1_eap_1_cpu)
ep_node_2_eap_1_cpu = np.stack(ep_node_2_eap_1_cpu)
ep_node_3_eap_1_cpu = np.stack(ep_node_3_eap_1_cpu)

ep_node_1_eap_1_mem = np.stack(ep_node_1_eap_1_mem)
ep_node_2_eap_1_mem = np.stack(ep_node_2_eap_1_mem)
ep_node_3_eap_1_mem = np.stack(ep_node_3_eap_1_mem)

ep_node_1_eap_2_cpu = np.stack(ep_node_1_eap_2_cpu)
ep_node_2_eap_2_cpu = np.stack(ep_node_2_eap_2_cpu)
ep_node_3_eap_2_cpu = np.stack(ep_node_3_eap_2_cpu)

ep_node_1_eap_2_mem = np.stack(ep_node_1_eap_2_mem)
ep_node_2_eap_2_mem = np.stack(ep_node_2_eap_2_mem)
ep_node_3_eap_2_mem = np.stack(ep_node_3_eap_2_mem)



'''
print(ep_imd_rewards_1.shape, ep_cur_done_1.shape, ep_cur_undone_1.shape)

print(ep_imd_rewards_1.shape, ep_cur_done_1.shape, ep_cur_undone_1.shape)


print(np.std(ep_imd_rewards_1, axis=0), np.std(ep_cur_done_1, axis=0), np.std(ep_cur_undone_1, axis=0))

'''

std_rewards_1 = np.std(ep_imd_rewards_1, axis=0)
std_rewards_2 = np.std(ep_imd_rewards_2, axis=0)

std_cur_done_1 = np.std(ep_cur_done_1, axis=0)
std_cur_done_2 = np.std(ep_cur_done_2, axis=0)

std_cur_undone_1 = np.std(ep_cur_undone_1, axis=0)
std_cur_undone_2 = np.std(ep_cur_undone_2, axis=0)


std_node_1_eap_1_cpu = np.std(ep_node_1_eap_1_cpu, axis=0)
std_node_2_eap_1_cpu = np.std(ep_node_2_eap_1_cpu, axis=0)
std_node_3_eap_1_cpu = np.std(ep_node_3_eap_1_cpu, axis=0)

std_node_1_eap_1_mem = np.std(ep_node_1_eap_1_mem, axis=0)
std_node_2_eap_1_mem = np.std(ep_node_2_eap_1_mem, axis=0)
std_node_3_eap_1_mem = np.std(ep_node_3_eap_1_mem, axis=0)

std_node_1_eap_2_cpu = np.std(ep_node_1_eap_2_cpu, axis=0)
std_node_2_eap_2_cpu = np.std(ep_node_2_eap_2_cpu, axis=0)
std_node_3_eap_2_cpu = np.std(ep_node_3_eap_2_cpu, axis=0)

std_node_1_eap_2_mem = np.std(ep_node_1_eap_2_mem, axis=0)
std_node_2_eap_2_mem = np.std(ep_node_2_eap_2_mem, axis=0)
std_node_3_eap_2_mem = np.std(ep_node_3_eap_2_mem, axis=0)




mean_rewards_1 = np.mean(ep_imd_rewards_1, axis=0)
mean_rewards_2 = np.mean(ep_imd_rewards_2, axis=0)

mean_cur_done_1 = np.mean(ep_cur_done_1, axis=0)
mean_cur_done_2 = np.mean(ep_cur_done_2, axis=0)

mean_cur_undone_1 = np.mean(ep_cur_undone_1, axis=0)
mean_cur_undone_2 = np.mean(ep_cur_undone_2, axis=0)


mean_node_1_eap_1_cpu = np.mean(ep_node_1_eap_1_cpu, axis=0)
mean_node_2_eap_1_cpu = np.mean(ep_node_2_eap_1_cpu, axis=0)
mean_node_3_eap_1_cpu = np.mean(ep_node_3_eap_1_cpu, axis=0)

mean_node_1_eap_1_mem = np.mean(ep_node_1_eap_1_mem, axis=0)
mean_node_2_eap_1_mem = np.mean(ep_node_2_eap_1_mem, axis=0)
mean_node_3_eap_1_mem = np.mean(ep_node_3_eap_1_mem, axis=0)

mean_node_1_eap_2_cpu = np.mean(ep_node_1_eap_2_cpu, axis=0)
mean_node_2_eap_2_cpu = np.mean(ep_node_2_eap_2_cpu, axis=0)
mean_node_3_eap_2_cpu = np.mean(ep_node_3_eap_2_cpu, axis=0)

mean_node_1_eap_2_mem = np.mean(ep_node_1_eap_2_mem, axis=0)
mean_node_2_eap_2_mem = np.mean(ep_node_2_eap_2_mem, axis=0)
mean_node_3_eap_2_mem = np.mean(ep_node_3_eap_2_mem, axis=0)

plt.plot(throughput, label='throughput_' + name)
plt.legend(loc="lower right")
plt.show()
#plt.savefig(os.path.join(os.getcwd(), 'plots', name + 'throughput' +'.png'), dpi=300)
'''
print(range(len_episodes))



plt.errorbar(range(len_episodes), mean_rewards_1, std_rewards_1, linestyle='None', marker='^', capsize=3, label="Immediate rewards 1 with std_deviation")
plt.legend(loc="lower right")
plt.show()

plt.plot(throughput, label='throughput')
plt.legend(loc="lower right")
plt.show()





def plot_mean_std(mean, std, attribute):
    plt.style.use('ggplot') #Change/Remove This If you Want

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.plot(range(len_episodes), mean, alpha=0.5, color='red', label='cv', linewidth = 1.0)
    ax.fill_between(range(len_episodes), mean - std, mean + std, color='#888888', alpha=0.4)
    ax.fill_between(range(len_episodes), mean - 2*std, mean + 2*std, color='#888888', alpha=0.2)
    ax.legend(loc='best')
    #ax.set_ylim([0.88,1.02])
    ax.set_ylabel(attribute)
    ax.set_xlabel('Length of Episode')
    plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'plots', name + attribute +'.png'), dpi=fig.dpi)
    plt.close()


plot_mean_std(mean_rewards_1, std_rewards_1, 'rewards_1')
plot_mean_std(mean_rewards_2, std_rewards_2, 'rewards_2')


plot_mean_std(mean_cur_done_1 , std_cur_done_1, 'cur_done_1')
plot_mean_std(mean_cur_done_2 ,std_cur_done_2 , 'cur_done_2')
plot_mean_std(mean_cur_undone_1 , std_cur_undone_1, 'cur_undone_1')
plot_mean_std(mean_cur_undone_2 ,std_cur_undone_2 , 'cur_undone_2')

plot_mean_std(mean_node_1_eap_1_cpu, std_node_1_eap_1_cpu, 'mean_node_1_eap_1_cpu')
plot_mean_std( mean_node_2_eap_1_cpu, std_node_2_eap_1_cpu, 'mean_node_2_eap_1_cpu')
plot_mean_std(mean_node_3_eap_1_cpu, std_node_3_eap_1_cpu, 'mean_node_3_eap_1_cpu')


plot_mean_std(mean_node_1_eap_2_cpu, std_node_1_eap_2_cpu, 'mean_node_1_eap_2_cpu')
plot_mean_std(mean_node_2_eap_2_cpu, std_node_2_eap_2_cpu, 'mean_node_2_eap_2_cpu')
plot_mean_std(mean_node_3_eap_2_cpu, std_node_3_eap_2_cpu, 'mean_node_3_eap_2_cpu')


plot_mean_std(mean_node_1_eap_1_mem, std_node_1_eap_1_mem, 'mean_node_1_eap_1_mem')
plot_mean_std(mean_node_2_eap_1_mem, std_node_2_eap_1_mem, 'mean_node_2_eap_1_mem')
plot_mean_std(mean_node_3_eap_1_mem, std_node_3_eap_1_mem, 'mean_node_3_eap_1_mem')

plot_mean_std(mean_node_1_eap_2_mem, std_node_1_eap_2_mem, 'mean_node_1_eap_2_mem')
plot_mean_std(mean_node_2_eap_2_mem, std_node_2_eap_2_mem, 'mean_node_2_eap_2_mem')
plot_mean_std(mean_node_3_eap_2_mem, std_node_3_eap_2_mem, 'mean_node_3_eap_2_mem')


plot_mean_std(mean_node_3_eap_2_mem, std_node_3_eap_2_mem, 'mean_node_3_eap_2_mem')

'''