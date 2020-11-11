# -*- coding: utf-8 -*-
from networkx.classes import graph
from ConuForecast.src.graph_utils import DBconnector, GraphManager
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Dataset, Data
import networkx as nx
import os
import numpy as np
import os.path as osp
import pickle
from collections import defaultdict

# https://gist.github.com/YannDubs/3550259636987a7b460a200efbd6acf3

class ConuGraphDataset(Dataset):
    
    def __init__(self, root:str, time_step:int, graph_manager:GraphManager, attrs_dict:dict, N:int=1000, clean:bool=True, transform=None, pre_transform=None):
        # self.elapsed_time = elapsed_time
        self.time_step = time_step
        self.target_dict = defaultdict(int)
        self.graph_manager = graph_manager
        self.clean = clean
        self.attrs_dict = attrs_dict
        self.N = N
        super(ConuGraphDataset, self).__init__(root, transform, pre_transform)
        self.process()
        self.data = None
 

    @property
    def raw_file_names(self):
        return os.listdir(osp.join(self.root, 'raw'))

    @property
    def processed_file_names(self):
        return os.listdir(osp.join(self.root, 'processed'))
        

    def download(self):
        if self.clean:
            [os.remove(f'{self.raw_dir}/{file}') for file in self.raw_file_names]    

        graphs = self.graph_manager
        
        time_steps = (sorted(self.graph_manager.get_time_steps()[1])[::self.time_step])
        
        #samples by ET
        n = self.N / len(time_steps)

        for time in time_steps:
            a_graph = graphs.build_digraph(time, self.attrs_dict, in_place=False)
            
            # stratified sampling
            nodes_int_dict = {i:j['node_id'] for i,j in nx.convert_node_labels_to_integers(a_graph).nodes(True)}
            node_int_target = np.array([j['target'] for i,j in nx.convert_node_labels_to_integers(a_graph).nodes(True)])
            node_int_arr = np.array([i for i,j in nx.convert_node_labels_to_integers(a_graph).nodes(True)])

            sampled_nodes = train_test_split(node_int_arr, node_int_target, test_size= n/len(node_int_arr))

            for node in [nodes_int_dict[i] for i in sampled_nodes[1]]:
                graphs.subgraphs_to_torch_tensors(time, node, self.attrs_dict, self.raw_dir, to_pickle=True)


    def process(self):

        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with open(raw_path, 'rb') as pickle_file:
                torch_data_dict = pickle.load(pickle_file)            
                torch_data = Data.from_dict(torch_data_dict)


            # if self.pre_filter is not None and not self.pre_filter(torch_data):
            #     continue

            # if self.pre_transform is not None:
            #     torch_data = self.pre_transform(torch_data)

            torch.save(torch_data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

            self.target_dict[f'data_{i}'] = torch_data['y']

            i += 1

    
    def len(self):
        return len(self.processed_file_names) - 2

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        return len(set(j.item() for i,j in self.target_dict.items()))