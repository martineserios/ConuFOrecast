# -*- coding: utf-8 -*-
import os
import os.path as osp
from posix import listdir
from tqdm.std import tqdm
import networkx as nx
import numpy as np
import pickle
from collections import defaultdict
from ConuForecast.src.graph_utils import GraphEngine
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Dataset, Data


class ConuSubGraphDataset(Dataset):
    def __init__(
        self, root:str, 
        time_step:int, 
        graph_manager:GraphEngine, 
        N:int, 
        attrs_dict:dict, 
        nodes_to_sample:list=[], 
        time_recurrence=False, 
        clean:bool=True, 
        transform=None, 
        pre_transform=None
        ):

        self.root_dir = root
        self.time_step = time_step
        self.graph_name_dict= defaultdict(str)
        self.target_dict = defaultdict(int)
        self.graph_manager = graph_manager
        self.clean = clean
        self.attrs_dict = attrs_dict
        self.N = N
        self.graph_name_dict = defaultdict(str)
        self.time_recurrence = time_recurrence
        if len(nodes_to_sample) != 0:
            self.sampled_nodes = nodes_to_sample
        else:
            self.sampled_nodes = []                        
        if self.clean:  
            [os.remove(f'{self.root_dir}/raw/{file}') for file in os.listdir(f'{self.root_dir}/raw/')]
            [os.remove(f'{self.root_dir}/processed/{file}') for file in os.listdir(f'{self.root_dir}/processed/')]
        super(ConuSubGraphDataset, self).__init__(root, transform, pre_transform)
        self.process()
        self.data = None
 

    @property
    def raw_file_names(self):
        return os.listdir(osp.join(self.root, 'raw'))

    @property
    def processed_file_names(self):
        return os.listdir(osp.join(self.root, 'processed'))
        

    def download(self):
        graphs = self.graph_manager
        time_steps = (sorted(self.graph_manager.get_time_steps()[1])[::self.time_step])
        #samples by ET
        n = self.N / len(time_steps)

        if len(self.sampled_nodes) == 0:
            if self.time_recurrence:
                # stratified sampling
                reference_graph = graphs.build_digraph(time_steps[2], self.attrs_dict, persist=False)            
                nodes_int_dict = {i:j['node_id'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)}
                node_int_target = [j['target'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)]
                node_int_arr = [i for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)]
                sampled_nodes,_,_,_ = train_test_split(node_int_arr, node_int_target, train_size= n/len(node_int_arr))
                self.sampled_nodes = [nodes_int_dict[i] for i in sampled_nodes] 

                for time in tqdm(time_steps):
                    # graphs.build_digraph(time, self.attrs_dict, persist=False)
                    for node in [nodes_int_dict[i] for i in sampled_nodes]:
                        graphs.subgraphs_to_torch_tensors(time, node, self.attrs_dict, self.raw_dir, to_pickle=True)
            else:
                
                #samples by ET
                n = self.N / len(time_steps)

                for time in tqdm(time_steps):
                    # stratified sampling
                    reference_graph = graphs.build_digraph(time, self.attrs_dict, persist=False)
                    nodes_int_dict = {i:j['node_id'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)}
                    node_int_target = np.array([j['target'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)])
                    node_int_arr = np.array([i for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)])
                    sampled_nodes,_,_,_ = train_test_split(node_int_arr, node_int_target, train_size= n/len(node_int_arr))
                    
                    for node in [nodes_int_dict[i] for i in sampled_nodes]:
                        graphs.subgraphs_to_torch_tensors(time, node, self.attrs_dict, self.raw_dir, to_pickle=True)

        else:
            if self.time_recurrence:
                reference_graph = graphs.build_digraph(time_steps[2], self.attrs_dict, persist=False)
                nodes_int_dict = {i:j['node_id'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)}

                for time in tqdm(time_steps):
                    # graphs.build_digraph(time, self.attrs_dict, persist=False)
                    for node in [nodo for nodo in self.sampled_nodes]:
                        graphs.subgraphs_to_torch_tensors(time, node, self.attrs_dict, self.raw_dir, to_pickle=True)
            else:
                
                #samples by ET
                n = self.N / len(time_steps)

                for time in tqdm(time_steps):
                    # stratified sampling
                    reference_graph = graphs.build_digraph(time, self.attrs_dict, persist=False)
                    nodes_int_dict = {i:j['node_id'] for i,j in nx.convert_node_labels_to_integers(reference_graph).nodes(True)}
                    
                    for node in [nodo for nodo in self.sampled_nodes]:
                        graphs.subgraphs_to_torch_tensors(time, node, self.attrs_dict, self.raw_dir, to_pickle=True)


    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with open(raw_path, 'rb') as pickle_file:
                torch_data_dict = pickle.load(pickle_file)            
                torch_data = Data.from_dict(torch_data_dict)

            filename = raw_path.split('/')[-1].split('.')[0]
            if len(filename) > 1:
                self.graph_name_dict[i] = f'{i}_{filename}'

                # if self.pre_filter is not None and not self.pre_filter(torch_data):
                #     continue

                # if self.pre_transform is not None:
                #     torch_data = self.pre_transform(torch_data)

                torch.save(torch_data, osp.join(self.processed_dir, f'{i}_{filename}.pt'))

                self.target_dict[f'{i}_{filename}'] = torch_data['y']
                # self.target_dict[f'data_{i}'] = torch_data['y']

                i += 1


    def len(self):
        return len(self.graph_name_dict)


    def get(self, idx):
        graph_key = self.graph_name_dict[idx]
        data = torch.load(osp.join(self.processed_dir, f'{graph_key}.pt'))
        return data


    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        return len(set(j.item() for i,j in self.target_dict.items()))


    def train_test_split(self, test_size:float):
        node_int_target = [self.get(i).y.item() for i in range(self.len())]
        node_int_arr = [i for i in range(self.len())]

        train_ids, test_ids, _, _ = train_test_split(
            node_int_arr, 
            node_int_target, 
            test_size = test_size, 
            stratify = node_int_target
            )

        train_set = self[train_ids]
        test_set = self[test_ids]

        return train_set, test_set        
