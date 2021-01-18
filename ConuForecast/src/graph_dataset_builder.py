# -*- coding: utf-8 -*-
import os
import os.path as osp
import pickle
import torch
from tqdm.std import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data

# local packages
from ConuForecast.src.graph_utils import GraphEngine

class ConuGraphDataset(Dataset):
    def __init__(
        self, root:str, 
        time_step:int, 
        graph_manager:GraphEngine, 
        attrs_dict:dict, 
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
        self.graph_name_dict = defaultdict(str)
        if self.clean:  
            [os.remove(f'{self.root_dir}/raw/{file}') for file in os.listdir(f'{self.root_dir}/raw/')]
            [os.remove(f'{self.root_dir}/processed/{file}') for file in os.listdir(f'{self.root_dir}/processed/')]
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
        graphs = self.graph_manager
        time_steps = (sorted(self.graph_manager.get_time_steps()[1])[::self.time_step])


        for time in tqdm(time_steps):
            graphs.graph_to_torch_tensor(time, self.attrs_dict, self.raw_dir, to_pickle=True)

    
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

                # if self.transform:
                #     torch_data = self.normalizer(torch_data)

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


    # def train_test_split(self, test_size:float):
    
    #     N = self.graph_manager.get_time_steps()[0].shape[0] // self.time_step
    #     train_ids, test_ids, _, _ = train_test_split(
    #         node_int_arr, 
    #         node_int_target, 
    #         test_size = test_size, 
    #         stratify = node_int_target
    #         )

    #     train_set = self[train_ids]
    #     test_set = self[test_ids]

    #     return train_set, test_set        
