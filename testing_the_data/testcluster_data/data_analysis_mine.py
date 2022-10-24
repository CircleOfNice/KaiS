import logging
import numpy as np
import pandas as pd
import os
import json
import logging
import time
import regex as re
from typing import Dict, List
from multiprocessing import Process, Lock
import argparse


FILENAME = ['taskdata_2.json', 'taskdata_3.json', 'taskdata_4.json', 'taskdata_5.json', 'taskdata_6.json', 'taskdata_1.json'] 
DATABASE = ['cpu', 'memory']
DATAVAR = ['util', 'request', 'limit', 'alloc']
JUNKSIZE = 50
# -> out (cpuUtil, cpuReq, cpuLim, cpuAlloc, memUtil, memReq, memLim, memAlloc, )
def nodesUsage(node_data: Dict[str, str]) -> np.ndarray:
    out = np.ndarray((1,8))
    for index_1, data_b in enumerate(DATABASE):
        resource = f'{data_b.capitalize()}Data'
        for index_2, data_v in enumerate(DATAVAR):
            data_type = f'{data_b}{data_v.capitalize()}'
            data_point = node_data[resource][data_type]
            out[0][index_2+(index_1*4)] = data_point
    print('Out : ', out)
    return out

def producePlot(filename: str, make_np: bool, data_path: str=None) -> None:
    file_dir = os.getcwd()
    plot_dir = os.path.join(file_dir, 'Plot')
    task_file = os.path.join(file_dir, filename)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    ex = r'[.]'
    pic_dir = re.split(ex, filename)
    pic_dir = os.path.join(plot_dir, pic_dir[0])

    junk_dir = os.path.join(pic_dir, "Junks")

    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)

    if not os.path.isdir(junk_dir):
        os.mkdir(junk_dir)
    print('task_file : ', task_file)
    #a=b
    data = pd.read_csv(task_file, sep=";")
    print('Analyse starting ...')
    start = time.time()
    try:
        node_1 = np.zeros((1,8))
        node_2 = np.zeros((1,8)) 
        node_3 = np.zeros((1,8))
        node_4 = np.zeros((1,8))
        node_5 = np.zeros((1,8))
        test_array = np.zeros((12,3))
        data_array = []

        scheduled_list = []
        task_cpu_req = []
        task_cpu_lim = []
        task_mem_lim = []
        task_mem_req = []
        nodes = [node_1, node_2, node_3, node_4, node_5]


        for index, row in enumerate(data.values):
            sys_array = np.zeros((1,8))
            text = f'Currently at row: {index} of {len(data.values)}'
            print(text)

            task_data = json.loads(row[1])
            node_data = json.loads(row[2])
            for i in range(5):
                print()

            print('task_data : ', type(task_data))
            print('node_data : ', type(node_data))
            #a=b
            scheduled_node = row[3]          

            scheduled_list.append(scheduled_node)
            task_cpu_req.append(task_data['cpuTaskRequest'])
            task_cpu_lim.append(task_data['cpuTaskLimit'])
            task_mem_lim.append(task_data['memoryTaskRequest'])
            task_mem_req.append(task_data['memoryTaskLimit'])

            for index, node in enumerate(node_data):
                #print(index, node)
                val = nodesUsage(node_data[node])
                tmp = np.delete(val, [3,7], 1).copy()
                
                test_array[index] = tmp[:,:3]
                test_array[index+len(node_data)+1] = tmp[:,3:6]
                print('test_array[index] ', test_array[index])
                print('test_array[index+len(node_data)+1 ', test_array[index+len(node_data)+1])
                sys_array += val
                nodes[index] = np.vstack((nodes[index], val))

            if make_np:
                # -> out (cpuUtil, cpuReq, cpuLim, cpuAlloc, memUtil, memReq, memLim, memAlloc, )
                test_array[5] = [0, task_data['cpuTaskRequest'], task_data['cpuTaskLimit']]
                test_array[11] = [0, task_data['memoryTaskRequest'], task_data['memoryTaskLimit'] ]
                data_array.append(test_array.copy())
                # target is scheduled node.....
                
        task_data = ((task_cpu_req, task_cpu_lim), (task_mem_req, task_mem_lim))       

        if make_np:
            ex = r'[_.]+'
            split = re.split(ex, filename)
            data_loc = os.path.join(data_path, f'data-{split[1]}')
            label_loc = os.path.join(data_path, f'label-{split[1]}')
            np.save(label_loc, np.array(scheduled_list))
            np.save(data_loc, np.array(data_array))
            
        stop = time.time()
        print("Time spend Analyzin: ", stop - start)
    except:
        logging.ERROR("System crashed")

def main(args:argparse) -> None:
    start = time.time()
    pro = []
    data_path = ""
    if args.make_np:
        data_path = os.path.join(os.getcwd(), 'data')
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
    
    for index, file in enumerate(FILENAME):
        print(f'starting analyse with file {file}')
        if args.debug:
            producePlot(file, args.make_np, data_path)
        else:
            print('inside debug')
            pro.append(Process(target=producePlot, args=(file, args.make_np, data_path,)))
            pro[index].start()
    
    if args.debug:
        for p in pro:
            p.join()

    print('finished..... \n')

    stopp = time.time()
    print('all files done in : \n', stopp-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data analysis for crcl")
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--make-np', default=True, action='store_true')
    args = parser.parse_args()
    main(args)
