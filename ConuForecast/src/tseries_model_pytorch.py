# Timeseries-Supervised Learning Approach

# external library imports
import sys
import os
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
import multiprocessing

# internal library imoprts
sys.path.append(os.path.join(os.getcwd(), '../..'))
from ConuForecast.src.tseries_dataset import TseriesManager
from ConuForecast.src.graph_utils import DBconnector, GraphEngine

# fix seed
random.seed(42)

## Database connection

# db connection set-up
host = '172.17.0.1'
port = 5555
db = 'base-ina'
user = 'postgres'
password = 'postgres'

database_conn = DBconnector(host, port, db, user, password)

## Dataset building

### Setup of Tiseries Engine for each event

events_dict = defaultdict(dict)

# list events with their sizes
query_events = "SELECT event_id FROM events"
events = database_conn.query(query_events)
# just relevant events
irrelevant_events = ['007', 'sjj_157', 'sjj-38']
events = [event[0] for event in events if event[0] not in irrelevant_events]

# # list events size
# for event in events:
#     query_events_size = f"SELECT COUNT(DISTINCT elapsed_time) FROM events_nodes WHERE event_id = '{event}'"
#     event_size = database_conn.query(query_events_size)
#     events_dict[event]['size'] = event_size[0][0]
    
# #     events_sizes_dict = {event[0]:event[1] for event in events_sizes}

events_dict = defaultdict(dict)
# list events size

def main():

    global load_event_size
    def load_event_size(event):
        global events_dcit
        
        query_events_size = f"SELECT COUNT(DISTINCT elapsed_time) FROM events_nodes WHERE event_id = '{event}'"
        event_size = database_conn.query(query_events_size)
        events_dict[event]['size'] = event_size[0][0]
        
        return event

    pool = multiprocessing.Pool(10)
    pool.map(load_event_size, events)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
