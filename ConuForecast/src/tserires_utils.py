# -*- coding: utf-8 -*-
import numpy as np
import random
from tqdm import tqdm

from ConuForecast.src.tseries_dataset import TseriesManager


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def list_diff(li1, li2):
    """
    Computes the difference between twi lists
    """
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))


def build_tseries_dataset(tseries_man: TseriesManager, n_train_nodes: int, features: list, timesteps: int) -> dict:
    """
    Builds the timeseries for training. Also stores some information
    to use for some hyperparamethers.
    """
    sets = {
        'train': {},
        'test': {}
    }

    sets['train']['n_nodes'] = n_train_nodes
    sets['train']['nodes'] = random.sample(tseries_man.get_nodes(), n_train_nodes)
    
    #OJO ESTA LIMITADO A PROPOSITO
    sets['test']['nodes'] = list_diff(tseries_man.get_nodes(), sets['train']['nodes'])[:10]
    sets['test']['n_nodes'] = len(sets['test']['nodes'])

    n_features = len(features) - 1
    timesteps = timesteps
    series_size = len(tseries_man.get_time_steps()[0])

    for dataset in sets.keys():
        agg_df_X = []
        agg_df_y = []

        for node in tqdm(sets[dataset]['nodes']):
            node_seq = tseries_man.timeseries(
                node,
                features,
                datetime_index=False,
                peak_data=False
                )

            node_seq = node_seq.to_numpy().reshape(series_size, n_features + 1)

            X_node_seq, y_node_seq = split_sequences(node_seq, timesteps)

            agg_df_X.append(X_node_seq[:,:,:-1])
            agg_df_y.append(y_node_seq)

        X = np.array(agg_df_X).reshape(sets[dataset]['n_nodes'] * (series_size - timesteps),timesteps, n_features)
        y = np.array(agg_df_y).reshape(sets[dataset]['n_nodes'] * (series_size - timesteps), 1)

        sets[dataset]['X'] = X
        sets[dataset]['y'] = y

    return sets
