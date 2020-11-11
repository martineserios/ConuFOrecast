# -*- coding: utf-8 -*-
import os
import os.path as osp
import pickle
from collections import defaultdict
from ConuForecast.src.graph_utils import GraphManager
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Dataset, Data




class ConuGraphDataset(Dataset):
    
    def __init__(self, root:str, time_step:int, graph_manager:GraphManager, attrs_dict:dict, clean:bool=True, transform=None, pre_transform=None):
        # self.elapsed_time = elapsed_time
        self.time_step = time_step
        self.target_dict = defaultdict(int)
        self.graph_manager = graph_manager
        self.clean = clean
        self.attrs_dict = attrs_dict
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
        for time in time_steps:
            a_graph = graphs.build_digraph(time, self.attrs_dict, in_place=False)
        # [graphs.subgraph_to_torch(self.step, node, self.raw_dir, to_pickle=True) for node in a_graph.nodes]
        
        # [self.subgraph_to_torch(time, node, self.raw_dir, to_pickle=True)
        # for time in (sorted(self.graph_data_loader.get_time_steps[1])[::self.time_step])
        # ]

            for node in a_graph.nodes:
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