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


FILENAME = ['taskdata_1.json','taskdata_2.json', 'taskdata_3.json', 'taskdata_4.json', 'taskdata_5.json', 'taskdata_6.json'] 
DATABASE = ['cpu', 'memory']
DATAVAR = ['util', 'request', 'limit', 'alloc']
NODE_VAR = ['nodeName', 'nodeID']
len_task_data = 0.0
len_node_data = 0.0
def get_node_key_graph_dict(nodes_data):
    nodes = {}
    for node_data_overview_key, node_data_overview in nodes_data.items():
        node_overview_dict = {}
        for i, (node_overview_key, node_data) in enumerate(node_data_overview.items()):
            if isinstance(node_data, str):
                node_overview_dict[node_overview_key] =  node_data    
            else:
                node_elem_dict = {}
                for node_elem_key, node_elem_value in node_data.items():
                    node_elem_dict[node_elem_key] = node_elem_value
                node_overview_dict[node_overview_key] =  node_elem_dict     
        nodes[node_data_overview_key] = node_overview_dict
    return nodes

def get_df_dict_from_node_graph(nodes_key_graph):
    node_df_dict= {}
    for node_key in nodes_key_graph.keys():
        column_list = []
        for node_overview_key in nodes_key_graph[node_key].keys():
            if node_overview_key in NODE_VAR:
                column_list.append(node_overview_key)
            elif isinstance(nodes_key_graph[node_key][node_overview_key], dict):
                for node_elem_key in nodes_key_graph[node_key][node_overview_key].keys():
                    column_list.append(node_elem_key)
        node_df_dict[node_key] = pd.DataFrame(columns=column_list)
    return node_df_dict

def get_dict_data(dictionary, column):
    for key in dictionary.keys():
        if key == column:
            return dictionary[key]
        
def add_data_to_df_dict(node_df_dict, nodes_data):
    for key in node_df_dict.keys():
        node_overview_data = nodes_data[key]
        df = node_df_dict[key]
        column_names = df.columns.values.tolist()
        column_data_dict = {}
        for node_overview_key in node_overview_data.keys():
            
            if node_overview_key in column_names:
                column_data_dict[node_overview_key] = node_overview_data[node_overview_key] 
            else:
                for node_elem_key in node_overview_data[node_overview_key].keys():
                    if node_elem_key in column_names:
                        column_data_dict[node_elem_key] = node_overview_data[node_overview_key][node_elem_key]
        df = df.append(column_data_dict, ignore_index=True)
        node_df_dict[key] = df
    return node_df_dict
def get_path(file):
    path = os.path.join('data', file)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path
def save_nodes_df_dict_to_respective_csv(node_df_dict, file):
    path = get_path(file)
    for key in node_df_dict:
        df = node_df_dict[key]
        df.to_csv(os.path.join(path, key+'.csv' ))
    
def get_task_df(tasks_data):
    cols = list(tasks_data.keys())
    df = pd.DataFrame(columns=cols)    
    return df

def save_df(df, file):
    path = os.path.join('data', file)
    df.to_csv(os.path.join(path, 'task.csv' ))
    
def complete_dfs_for_file(data, node_df_dict, task_df):
    
    for index, row in enumerate(data.values):
        tasks_data = json.loads(row[1])
        nodes_data = json.loads(row[2])
        scheduled_node = row[3]
        tasks_data.update({'scheduled_node': scheduled_node})
        node_df_dict = add_data_to_df_dict(node_df_dict, nodes_data)
        #print(node_df_dict)
        task_df = task_df.append(tasks_data, ignore_index = True)
        
    return task_df, node_df_dict
for file in FILENAME:
    
    data = pd.read_csv(file, sep=";")
    row = data.values[0]

    tasks_data = json.loads(row[1])
    nodes_data = json.loads(row[2])
    scheduled_node = row[3]

    tasks_data.update({'scheduled_node': scheduled_node})
    
    task_df = get_task_df(tasks_data)
    #a=b
    nodes_key_graph_dict = get_node_key_graph_dict(nodes_data)
    node_df_dict = get_df_dict_from_node_graph(nodes_key_graph_dict)
    task_df, node_df_dict = complete_dfs_for_file(data, node_df_dict, task_df)
    save_df(task_df, file)
    save_nodes_df_dict_to_respective_csv(node_df_dict, file)
    print('Json file dumped as csv files: ', file)