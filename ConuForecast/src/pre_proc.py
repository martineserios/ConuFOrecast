# -*- coding: utf-8 -*-
from networkx.classes import graph
from ConuForecast.src.graph_utils import DBconnector, GraphManager
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Dataset, Data
import networkx as nx
import os
import os.path as osp
import pickle
from collections import defaultdict


class GraphDataLoader(GraphManager):

    def __init__(self, attrs_dict:dict, model:str, event:str, precip:str, conn):
        super(GraphDataLoader, self).__init__(model, event, precip, conn)
        self.attrs_dict = attrs_dict
        self.node_attrs = attrs_dict['nodes']
        self.edge_attrs = attrs_dict['edges']
        self.graphtensors = {}
        self.train_DataLoaders = {}
        self.test_DataLoaders = {}



    def subgraph_to_torch(self, elapsed_time:str, node:str, raw_data_folder:str, detailed:bool=False, to_pickle:bool=True,):
        """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Took it  from torch_geometric.data.Data

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
        """
        DG = self.build_subgraph(node=node, elapsed_time=elapsed_time, attrs=self.attrs_dict, acc_data=False, in_place=False)

        graph_ = DG.copy()
        graph_ = nx.convert_node_labels_to_integers(graph_)
        graph_ = graph_.to_directed() if not nx.is_directed(graph_) else graph_
        edge_index = torch.tensor(list(graph_.edges)).t().contiguous()

        torch_data = defaultdict(int)
        torch_data['y'] = DG.nodes()[node]['target']
        graph_target = torch_data['y']
  
        if detailed:
            for i, (_, feat_dict) in enumerate(graph_.nodes(data=True)):
                for key, value in feat_dict.items():
                    torch_data['node_' + str(key)] = [value] if i == 0 else torch_data['node_' + str(key)] + [value]

            for i, (_, _, feat_dict) in enumerate(graph_.edges(data=True)):
                for key, value in feat_dict.items():
                    torch_data['edge_' + str(key)] = [value] if i == 0 else torch_data['edge_' + str(key)] + [value]
        
        torch_data['x'] = [list(v[1].values())[4:-1] for i,v in enumerate(graph_.nodes(data=True))]

        torch_data['edge_attrs'] = [list(v[2].values())[5:] for i,v in enumerate(graph_.edges(data=True))]

        torch_data['edge_index'] = edge_index.view(2, -1)

        for key, data in torch_data.items():
            try:
                if (key == 'x'):# | (key == 'edge_attrs'):
                    # torch_data[key] = torch.tensor(item)
                    torch_data[key] = torch.tensor(data)
                elif (key == 'edge_index') | (key == 'edge_attrs'):
                    torch_data[key] = torch.tensor(data, dtype=torch.long)
                elif (key == 'y'):
                    torch_data[key] = torch.tensor(data, dtype=torch.long)

            except ValueError:
                print(data)
                pass

        # torch_data = Data.from_dict(torch_data)
        # torch_data.num_nodes = graph.number_of_nodes()

        if to_pickle:
            # open a file, where you ant to store the data
            file = open(f'{raw_data_folder}/{self.event}_{node}_{elapsed_time}_{graph_target}.gpickle', 'wb')

            # dump information to that file
            pickle.dump(torch_data, file, pickle.HIGHEST_PROTOCOL)

            # close the file
            file.close()

        else:
            return torch_data
      
    # def to_DataLoader(self, elapsed_time:str, test_size=0.2,train_size=0.8, train_batch_size=32, test_batch_size=6, shuffle=True, drop_last=True, num_workers=0):
        
    #     try:
    #         geometric_torch_data = self.graphtensors[f'{self.event}_{elapsed_time}']
        
    #     except:
    #         self.graphto_torch(elapsed_time)
    #         geometric_torch_data = self.graphtensors[f'{self.event}_{elapsed_time}']

    #     train_D, test_D, train_L, test_L = train_test_split(
    #         geometric_torch_data.nodes_attrs.numpy(), geometric_torch_data.target.numpy(),
    #         test_size=test_size,train_size=train_size, shuffle=shuffle, stratify=geometric_torch_data.target.numpy()
    #         )

    #     DatasetTrain = TensorDataset(torch.from_numpy(train_D),torch.from_numpy(train_L))

    #     DatasetTest = TensorDataset(torch.from_numpy(test_D),torch.from_numpy(test_L))

    #     trainloader= DataLoader(
    #         DatasetTrain, batch_size=train_batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    #         )

    #     testloader= DataLoader(
    #         DatasetTest, batch_size=test_batch_size, drop_last=drop_last, num_workers=num_workers
    #         )

    #     self.train_DataLotrain_DataLoadersaders[f'{self.event}_{elapsed_time}'] = trainloader
    #     self.test_DataLoaders[f'{self.event}_{elapsed_time}'] = testloader




class ConuGraphDataset(Dataset):
    
    def __init__(self, root:str, elapsed_time:str, graph_data_loader:GraphDataLoader, clean:bool=True, transform=None, pre_transform=None):
        self.elapsed_time = elapsed_time
        self.target_dict = defaultdict(int)
        self.graph_data_loader = graph_data_loader
        self.clean = clean
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

        graphs = self.graph_data_loader
        a_graph = graphs.build_digraph(self.elapsed_time, graphs.attrs_dict, in_place=False)
        [graphs.subgraph_to_torch(self.elapsed_time, node, self.raw_dir, to_pickle=True) for node in a_graph.nodes]

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