import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
def get_all_task(path):
    """Get Processed data from the file given in path

    Args:
        path ([str]): [path to the file]

    Returns:
        [list]: [lists containing type of task , start time & end time of tasks, cpu and memory]
    """
    type_list = []
    start_time = []
    end_time = []
    cpu_list = []
    mem_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            type_list.append(row[3])
            start_time.append(row[5])
            end_time.append(row[6])
            cpu_list.append(row[7])
            mem_list.append(row[8])

    init_time = int(start_time[0])
    #removed_indices = []
    new_type_list = []
    new_start_time = []
    new_end_time = []
    new_cpu_list = []
    new_mem_list = []
    time_required_list = []
    for i in range(len(start_time)):
        
        
        start_time_ = (int(start_time[i]) - init_time)
        end_time_ = (int(end_time[i]) - init_time) #+ random.randint(-9, 9)
        type_list_ = (int(type_list[i]) - 1)
        cpu_list_ = (int(cpu_list[i]) / 100.0)
        mem_list_ = (float(mem_list[i]))
        time_diff = int(end_time_ - start_time_)
        
        type_list[i] = type_list_
        start_time[i] = start_time_
        end_time[i] = end_time_
        cpu_list[i] = cpu_list_
        mem_list[i] = mem_list_
        
        if time_diff<0:
            continue
        
        #if type(time_diff) != int:
            
        #    print('type time diff : ', type(time_diff))
        
        new_start_time.append(start_time_)
        new_end_time.append(end_time_)
        new_cpu_list.append(cpu_list_)
        new_mem_list.append(mem_list_)
        new_type_list.append(type_list_)
        time_required_list.append(int(time_diff))
        
    print('len(time_required_list) : ', len(time_required_list))
    print('len(new_mem_list) : ', len(new_mem_list))
    #a=b
    name_list = ['type_list', 'start_time', 'end_time', 'cpu_list', 'mem_list']
    new_name_list = ['type_list', 'start_time', 'end_time', 'cpu_list', 'mem_list', 'time_diff_list']
    all_task = [type_list, start_time, end_time, cpu_list, mem_list]
    new_all_task = [new_type_list, new_start_time, new_end_time, new_cpu_list, new_mem_list, time_required_list]
    
    all_task_dict = {}
    for i, name in enumerate(name_list):
        all_task_dict[name] = all_task[i]
    
    new_all_task_dict = {}
    for i, name in enumerate(new_name_list):
        new_all_task_dict[name] = new_all_task[i]
    all_task_df = pd.DataFrame(all_task_dict)    
    new_all_task_df = pd.DataFrame(new_all_task_dict)    
    return all_task, all_task_df, new_all_task,new_all_task_df, max(type_list)

all_task1, all_task_df1, new_all_task1, new_all_task_df1, max_task_type1 = get_all_task('./data/Task_1.csv')# processed data [type_list, start_time, end_time, cpu_list, mem_list] fed to eAP 1
all_task2, all_task_df2, new_all_task2, new_all_task_df2, max_task_type2 = get_all_task('./data/Task_2.csv')# processed data fed to eAP 2

'''
print('all_task1 : ', len(all_task1[0]), len(all_task1[1]), len(all_task1[2]), len(all_task1[3]), len(all_task1[4]))
print('new_all_task1 : ', len(new_all_task1[0]) , len(new_all_task1[1]) , len(new_all_task1[2]) , len(new_all_task1[3]), len(new_all_task1[4]))

print('all_task_df1 : ', all_task_df1.shape)
print('all_task_df1 : ', all_task_df1.head)
'''
#all_task_df1['type_list'].value_counts().plot(kind='bar')
#plt.show()


print(all_task_df1['type_list'].value_counts())
print(new_all_task_df1['type_list'].value_counts())


#all_task_df1['type_list'].value_counts().plot(kind='bar')
#plt.show()


#print(new_all_task_df1['time_diff_list'].value_counts())

#new_all_task1['time_diff_list'].value_counts().plot(kind='bar')
#plt.show()

#print('new_all_task_df1[time_diff_list] : ', new_all_task_df1['time_diff_list'])

#result = new_all_task_df1.dtypes

#print(result)

#new_all_task_df1['time_diff_list'].value_counts().plot(kind='bar')


'''
sub_df = new_all_task_df1['time_diff_list']
print()
print()
print(sub_df.head())
print(sub_df.shape)

#result = sub_df.dtypes

#print(result)
print()
print()
print(sub_df.columns)

print(sub_df["time_diff_list"])

print()
print()
'''
#sub_df = new_all_task_df1.loc[new_all_task_df1["time_diff_list"] >= 30 ]
counts = new_all_task_df1['time_diff_list'].value_counts()
sub_df = new_all_task_df1.loc[new_all_task_df1['time_diff_list'].isin(counts.index[counts > 50])]
print()
print()
print(sub_df.shape)

print(sub_df['time_diff_list'].value_counts())

sub_df['time_diff_list'].value_counts().plot(kind='bar')
plt.show()