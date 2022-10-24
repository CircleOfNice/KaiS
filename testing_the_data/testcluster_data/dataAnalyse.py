#from asyncio import as_completed
#from audioop import mul
#from cProfile import label
#from cgi import test
#from concurrent.futures import process
#import enum
import logging
#from platform import node, system
#import sys
#from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import os
import json
import logging
#import matplotlib.pyplot as plt
import time
import regex as re
import math
from typing import Dict, List
import plotClass as pc 
from multiprocessing import Process, Lock
import argparse


"""
'taskdata_19.json',
'taskdata_18_min.json', 'taskdata_17_min.json',
'taskdata_15.json', 'taskdata_16.json',
'taskdata_13.json', 'taskdata_14.json',
'taskdata_11.json', 'taskdata_10.json', 
'taskdata_9.json', 'taskdata_8.json', 
"""

FILENAME = [
            
            'taskdata_2.json', 'taskdata_3.json', 
            'taskdata_4.json', 'taskdata_5.json', 
            'taskdata_6.json', 'taskdata_1.json'
            ] 

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
            
            #print('resource :', resource)
            #print('data_type :', data_type)
            data_point = node_data[resource][data_type]
            out[0][index_2+(index_1*4)] = data_point
    return out

def producePlot(filename: str, make_np: bool, data_path: str=None) -> None:

    file_dir = os.getcwd()
    plot_dir = os.path.join(file_dir, 'Plot')
    task_file = os.path.join(file_dir, filename)
    #print('task_file: ', task_file)
    #a=b
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

    data = pd.read_csv(task_file, sep=";")
    #print(data.head())
    #a=b
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
        system = np.zeros((1,8))

        for index, row in enumerate(data.values):
            sys_array = np.zeros((1,8))
            text = f'Currently at row: {index} of {len(data.values)}'
            print(text)

            task_data = json.loads(row[1])
            node_data = json.loads(row[2])
            for i in range(5):
                print()
            print('index, row  : ', index, row )
            print(task_data)
            print(node_data)
            #a=b
            scheduled_node = row[3]          

            scheduled_list.append(scheduled_node)
            task_cpu_req.append(task_data['cpuTaskRequest'])
            task_cpu_lim.append(task_data['cpuTaskLimit'])
            task_mem_lim.append(task_data['memoryTaskRequest'])
            task_mem_req.append(task_data['memoryTaskLimit'])

            for index, node in enumerate(node_data):
                print(index, node)
                val = nodesUsage(node_data[node])
                tmp = np.delete(val, [3,7], 1).copy()
                test_array[index] = tmp[:,:3]
                test_array[index+len(node_data)+1] = tmp[:,3:6]
                sys_array += val
                nodes[index] = np.vstack((nodes[index], val))

            system = np.vstack((system, sys_array))

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
        '''
        start = time.time()
        print("Start Ploting")

        systemPlot = pc.PlotWrapper(4)
        title = {
            'subtitle' : 'cpu usage in cluster',
            'ax-0' : 'cpu utilization',
            'ax-1' : 'cpu requested',
            'ax-2' : 'cpu limit',
            'ax-3' : 'cpu alloc',
        }
        systemPlot.add_title(title)
        for i in range (4):
            systemPlot.add_plot(i, system[1:,i])
        systemPlot.set_grid()
        systemPlot.set_legend()
        systemPlot.save_2_dir(pic_dir, 'systemUsage')

        title = {
            'subtitle' : 'scheduling of the pods in cluster',
            'ax-0' : 'pods scheduled',
            'ax-1' : 'task cpu requested/limits',
            'ax-2' : 'nodes cpu request',
            'ax-3' : 'nodes cpu limit',
        }

        for index, data_b in enumerate(DATABASE):
            podPlot = pc.PlotWrapper(4)

            podPlot.add_title(title)
            podPlot.add_scatter(0, scheduled_list)
            podPlot.add_scatter(1, task_data[index][1], label=f'{data_b} limit', marker='^')
            podPlot.add_scatter(1, task_data[index][0], label=f'{data_b} request', marker='o')
            for i in range(2):
                for subindex, node in enumerate(nodes):
                    data = node[1:, 1+i+(index*4)]
                    label_text = f'node-{subindex} {data_b} {DATAVAR[1+i]}'
                    podPlot.add_plot(2+i, data, label=label_text)

            podPlot.set_grid()
            podPlot.set_legend()
            podPlot.save_2_dir(pic_dir, f'{data_b}-podsScheduled')

        for index, data_b in enumerate(DATABASE):
            # make 50 er junks and save to junks folder
            lenght = math.ceil(len(scheduled_list)/JUNKSIZE)
            for i in range(lenght):
                start = 0 + (i*JUNKSIZE)
                stop = 49 + (i*JUNKSIZE)
                junkPlot = pc.PlotWrapper(4)
                junkPlot.add_title(title) 
                junkPlot.add_scatter(0, scheduled_list[start:stop])
                junkPlot.add_scatter(1, task_data[index][1][start:stop], label=f'{data_b} limit', marker='^')
                junkPlot.add_scatter(1, task_data[index][0][start:stop], label=f'{data_b} request', marker='o')
                for i in range(2):
                    for subindex, node in enumerate(nodes):
                        data = node[start+1:stop, 1+i+(index*4)]
                        label_text = f'node-{subindex} {data_b} {DATAVAR[1+i]}'
                        junkPlot.add_plot(2+i, data, label=label_text, start=start+1)

                junkPlot.set_grid()
                junkPlot.set_legend()
                junkPlot.save_2_dir(os.path.join(pic_dir, junk_dir), f'{data_b}-podsScheduled-{start}-{stop}')
    
        #generate plots
        for index, data_b in enumerate(DATABASE):
            for sub_index_1, node in enumerate(nodes):
                nodePlot = pc.PlotWrapper(4)
                title = {
                    'subtitle' : f'node-{sub_index_1}, {data_b}-data',
                    'ax-0' : f'{data_b} utilization',
                    'ax-1' : f'{data_b} requested',
                    'ax-2' : f'{data_b} limit',
                    'ax-3' : f'{data_b} alloc',
                }
                nodePlot.add_title(title)
                for sub_index_2, data_v in enumerate(DATAVAR):
                    data = node[1:, sub_index_2+(index*4)]
                    maxVal = np.ones((1, len(data))) * data.max()
                    labelMax = f'max value: {data.max()}' 
                    nodePlot.add_plot(sub_index_2, data)
                    nodePlot.add_plot(sub_index_2, maxVal[0], label=labelMax, linestyle='--', linewidth=0.5)    
                nodePlot.set_legend('lower right')
                nodePlot.set_grid()
                nodePlot.save_2_dir(pic_dir, f'{data_b}-node{sub_index_1}')

        stop = time.time()
        print("finished plots in: ", stop - start)
        '''
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
