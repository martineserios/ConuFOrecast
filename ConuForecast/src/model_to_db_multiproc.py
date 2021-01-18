# -*- coding: utf-8 -*-
# Loading libraries
import os
import sys
import time
from networkx.algorithms.centrality import group
import pandas as pd
import re
import csv
from swmmtoolbox import swmmtoolbox as swmm
from datetime import datetime
from os import listdir
from concurrent import futures
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
import multiprocessing

import pyproj

model_id =  'model_' + input('4-digit model_id like: 0123:    ' )
precipitation_id ='precipitation_' + input('4-digit raingage_id like 0123:    ')
event_id = input('4-digit event_id like: 0123:    ')
epsg_modelo = input('EPSG (ejemplo: 5348):    ')

project_folder = os.path.abspath(os.path.join(os.getcwd(),"../.."))
data_raw_folder = os.path.join(project_folder,'data', 'raw_swmm')
event_folder = os.path.join(data_raw_folder, 'Run_[' + event_id + ']')
model_inp = os.path.join(event_folder, 'model.inp')
model_out = os.path.join(event_folder, 'model.out')


# Connection to database
engine_base_ina = create_engine('postgresql://postgres:postgres@172.18.0.1:5555/base-ina')


RELEVANT_GROUP_TYPES_OUT = [
            'link',
            'node',
            'subcatchment',
            # 'system'
            ]

RELEVANT_GROUP_TYPES_INP = [
            'coordinates',
            'subcatchments',
            'raingages',
            'conduits',
            'orifices',
            'weirs',
            'outfalls',
            # 'vertices',
            # 'polygons',
            'subareas',
            # 'losses',
            'infiltration',
            'junctions',
            'storage',
            # 'properties',
            # "curves",
            ]

RELEVANT_LINKS = [
            # 'channel10944',
            # 'channel24416',
            # 'channel60443',
            # 'channel17459',
            # 'channel87859',
            # 'channel14380',
            # 'channel55414',
            # 'channel77496',
            # 'channel83013',
            # 'channel52767',
            # 'channel12818',
            # 'conduit11698',
            # 'channel6317',
            # 'conduit18801',
            # 'conduit50317',
            # 'conduit528',
            # 'conduit36611',
            # 'conduit50827',
            # 'conduit78108',
            # 'conduit57848',
            # 'conduit42638',
            # 'conduit34157',
            # 'conduit29340',
            # 'conduit19715',
            # 'conduit23023',
            # 'conduit37130',
            # 'conduit21772',
            # 'channel52598',
            # 'conduit75783',
            # 'conduit62715',
            # 'conduit48979',
            # 'conduit82544',
            # 'conduit83110',
            # 'conduit33678',
            # 'conduit18303',
            # 'conduit40724',
            # 'conduit13927'
        ]

RELEVANT_SUBCATCHMENTS = []

RELEVANT_NODES = []

RELEVANT_SUBAREAS = []

RELEVANT_OUTFALLS = []

RELEVANT_VERTICES = []

RELEVANT_POLYGNOS = []

RELEVANT_LINKS_CONDUITS = []

RELEVANT_LINKS_ORIFICES = []

RELEVANT_LINKS_WEIRS = []

RELEVANT_LOSSES = []

RELEVANT_INFILTRATION = []

RELEVANT_JUNCTIONS = []

RELEVANT_STORAGE = []



MODEL_OUT_COLS = {
    'SUBCATCHMENTS_COLS' : [
        'event_id',
        'elapsed_time',
        'subcatchment_id',
        'rainfall',
        'elapsed_time',
        'snow_depth',
        'evaporation_loss',
        'infiltration_loss',
        'runoff_rate',
        'groundwater_outflow',
        'groundwater_elevation',
        'soil_moisture'
    ],
    'LINKS_COLS' : [
        'event_id',
        'elapsed_time',
        'link_id',
        'flow_rate',
        'flow_depth',
        'flow_velocity',
        'froude_number',
        'capacity'
    ],
    'NODES_COLS' : [
        'event_id',
        'elapsed_time',
        'node_id',
        'depth_above_invert',
        'hydraulic_head',
        'volume_stored_ponded',
        'lateral_inflow',
        'total_inflow',
        'flow_lost_flooding'
    ]
}


MODEL_INP_COLS = {
    'NODES_COORDINATES' : [
        'node_id',
        'x_coord',
        'y_coord',
    ],
    "SUBCATCHMENTS" : [
      "subcatchment_id",
      "raingage_id",
      "outlet",
      "area",
      "imperv",
      "width",
      "slope",
      "curb_len"
    ],
    "LINKS_CONDUITS" : [
         "conduit_id",
         "from_node",
         "to_node",
         "length",
         "roughness",
         "in_offset",
         "out_offset",
         "init_flow",
         "max_flow"
    ],
    "LINKS_ORIFICES" : [
          "orifice_id",
          "from_node",
          "to_node",
          "type",
          "offset",
          "q_coeff",
          "gated",
          "close_time"
    ],
    "LINKS_WEIRS" : [
          "weir_id",
          "from_node",
          "to_node",
          "type",
          "crest_ht",
          "q_coeff",
          "gated",
          "end_con",
          "end_coeff",
          "surcharge"
    ],
    "SUBAREAS" : [
        "subcatchment_id",
        "n_imperv",
        "n_perv",
        "s_imperv",
        "s_perv",
        "pct_zero",
        "route_to"
    ],
    "NODES_STORAGE" : [
        "storage_id",
        "elevation",
        "max_depth",
        "init_depth",
        "shape",
        "curve_name_params",
        "n_a",
        "f_evap"
    ],
    "NODES_OUTFALLS" : [
        "outfall_id",
        "elevation",
        "type",
        # "stage_data",
        "gated",
        # "route_to"
    ],
    "NODES_JUNCTIONS" : [
        "junction_id",
        "elevation",
        "max_depth",
        "init_depth",
        "sur_depth",
        "aponded"
    ],
    "INFILTRATION": [
        "subcatchment_id",
        "max_rate",
        "min_rate",
        "decay",
        "dry_time",
        "max_infil",
    ],
    # "POLYGONS": [
    #     "subcatchment_id",
    #     "x_coord",
    #     "y_coord"
    # ],
    # "VERICES": [
    #     "link_id",
    #     "x_coord",
    #     "y_coord"
    # ],
    "PROPERTIES": [
        "model_name",
        "model_version",
        "flow_units",
        "infiltration",
        "flow_routing",
        "link_offsets",
        "min_slope",
        "allow_ponding",
        "skip_steady_state",
        "start_date",
        "start_time",
        "report_start_date",
        "report_start_time",
        "end_date",
        "end_time",
        "sweep_start",
        "sweep_end",
        "report_step",
        "wet_step",
        "dry_step",
        "routing_step",
        "inertial_damping",
        "normal_flow_limited",
        "force_main_equation",
        "variable_step",
        "lengthening_step",
        "min_surfarea",
        "max_trials",
        "head_tolerance",
        "sys_flow",
        "lat_flow_tol",
        "minimum_step",
        "threads"
    ]
}



# dictionary to store data
groups = {}


# Definition of starting postiion of each element
def group_start_line(model):
    with open(model, 'r') as inp:
        groups = {}
        count = 0

        lines = inp.readlines()
        for line in lines:
            if ('[' in line) & (']' in line):
                groups.update({line[1:-2].lower() : {'start': count}})
            count += 1
    # subselection of elements from MODEL_ELEMENTS
    groups = {key:value for key,value in groups.items() if key in RELEVANT_GROUP_TYPES_INP}

    LINK_TYPES = ['orifices', 'conduits', 'weirs']
    NODE_TYPES = ['outfalls', 'junctions', 'storage']

    for key in [key for key in groups.keys() if key in LINK_TYPES]:
        groups['links_' + key] = groups.pop(key)

    for key in [key for key in groups.keys() if key in NODE_TYPES]:
        groups['nodes_' + key] = groups.pop(key)

    groups['nodes_coordinates'] = groups.pop('coordinates')


    return groups



# adding header and skip-lines to elements dict
def build_groups_dicts(model):
    groups = group_start_line(model)
    count = 0

    for element, start_dict in groups.items():
        start = start_dict['start']

        with open(model, 'r') as inp:
            lines = inp.readlines()
            for index, line in enumerate(lines):
                if (index - start == 1) & (';;' in line) & (';;--' not in line):
                    groups[element].update({'header':[col for col in re.split("\s\s+", line[2:-1]) if len(col) > 1]})

                elif (index - start == 2) & (';;--------------' in line):
                    groups[element].update({'line_to_skip': index})

                elif (index - start == 3):
                    break


    # some corrrections on header because of mismatches on inp file

    # groups['properties'].update({'header': MODEL_INP_COLS['PROPERTIES']})
    groups['subcatchments'].update({'header': MODEL_INP_COLS['SUBCATCHMENTS']})
    groups['subareas'].update({'header': MODEL_INP_COLS['SUBAREAS']})
    groups['infiltration'].update({'header': MODEL_INP_COLS['INFILTRATION']})
    groups['links_conduits'].update({'header': MODEL_INP_COLS['LINKS_CONDUITS']})
    groups['links_weirs'].update({'header': MODEL_INP_COLS['LINKS_WEIRS']})
    groups['links_orifices'].update({'header': MODEL_INP_COLS['LINKS_ORIFICES']})
    groups['nodes_coordinates'].update({'header': MODEL_INP_COLS['NODES_COORDINATES']})
    groups['nodes_outfalls'].update({'header': MODEL_INP_COLS['NODES_OUTFALLS']})
    groups['nodes_storage'].update({'header': MODEL_INP_COLS['NODES_STORAGE']})
    groups['nodes_junctions'].update({'header': MODEL_INP_COLS['NODES_JUNCTIONS']})

    return groups

# %%

def list_files(directory, extension, prefix):
    return (f for f in listdir(directory) if (f.endswith('.' + extension)) & (f.startswith(prefix)))


def raingages_meta_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['raingages']['start']
    skip_rows = build_groups_dicts(model)['raingages']['line_to_skip']
    header = ['raingage_id', 'format', 'interval',  'unit']

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        formatted_line = [line[0].split()[0], line[0].split()[1], line[0].split()[2],line[0].split()[7]]
                        contents.append(formatted_line)

    df = pd.DataFrame(data = contents, columns= header,)
    df['interval'] = df['interval'].map( lambda x: datetime.strptime(x, '%H:%M'))
    df.insert(0, 'precipitation_id', precipitation_id)
    print('raingages','df created!')
    return df



def date_parser(line):
    year = line[0].split()[1]
    month = line[0].split()[2].zfill(2)
    day = line[0].split()[3].zfill(2)
    hour = line[0].split()[4].zfill(2)
    minute = line[0].split()[5].zfill(2)

    str_date = '-'.join([year, month, day, hour, minute] )

    date_format = '%Y-%m-%d-%H-%M'
    return datetime.strptime(str_date, date_format)

# %%
def raingages_to_df(event_folder, event_id, model, model_id):
    contents = []

    for file in list_files(event_folder, 'txt', 'P'):
        raingage_id = file.split('.')[0]

        with open(os.path.join(event_folder, file), newline='') as f:
            r = csv.reader(f)
            for i, line in enumerate(r):
                try:
                    formatted_line = [
                        raingage_id,
                        date_parser(line),
                        line[0].split()[6]
                        ]
                    contents.append(formatted_line)
                except:
                    print('error')
    df_timeseries = pd.DataFrame(data = contents, columns= ['raingage_id', 'elapsed_time', 'value'])
    df_timeseries.insert(0, 'precipitation_id', precipitation_id)

    df_metadata = raingages_meta_to_dfs(model, model_id)
    return df_metadata, df_timeseries
# %%

def load_raingages_to_db(event_folder, event_id, model, model_id):
    raingage_metadata, raingage_timeseries = raingages_to_df(event_folder, event_id, model, model_id)

    table_metadata = 'raingages_metadata'
    table_timeseries = 'raingages_timeseries'

    try:
        raingage_metadata.to_sql(table_metadata, engine_base_ina, index=False, if_exists='append')
    except Exception as e:
        print(e)
    try:
        raingage_timeseries.to_sql(table_timeseries, engine_base_ina, index=False, if_exists='append')
    except Exception as e:
        print(e)

# def group_type_to_dfs(model, model_id, group, id_col, col_to_check, own_relevant__list, relevant_dependent_list):
#     """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
#     """
#     start = build_groups_dicts(model)[group]['start']
#     skip_rows = build_groups_dicts(model)[group]['line_to_skip']
#     header = build_groups_dicts(model)[group]['header']

#     global own_relevant__list
#     own_relevant_list = []

#     df = pd.DataFrame()
#     with open(model, newline='') as f:
#         contents = []
#         r = csv.reader(f)
#         for i, line in enumerate(r):
#             if i >=  start + 1:
#                 if i != skip_rows:
#                     if not line:
#                         break
#                     # elif i == start + 1:
#                     #     headers = line
#                     else:
#                         if len(relevant_dependecy_list) == 0:
#                             own_relevant__list.append(line[0].split()[id_col])
#                             contents.append(line[0].split())
#                         else:
#                             if line[0].split()[col_to_check].lower() in relevant_dependent_list:
#                                 own_relevant__list.append(line[0].split()[id_col])
#                                 contents.append(line[0].split())

#     df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
#     df.insert(0, 'model_id', model_id)
#     print(group,'df created!')
#     return df








def conduits_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['links_conduits']['start']
    skip_rows = build_groups_dicts(model)['links_conduits']['line_to_skip']
    header = build_groups_dicts(model)['links_conduits']['header']

    global RELEVANT_LINKS_CONDUITS
    RELEVANT_LINKS_CONDUITS = []

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        if len(RELEVANT_LINKS) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0].lower() in RELEVANT_LINKS:
                                RELEVANT_LINKS_CONDUITS.append(line[0].split()[0])
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('conduits','df created!')
    return df

def weirs_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['links_weirs']['start']
    skip_rows = build_groups_dicts(model)['links_weirs']['line_to_skip']
    header = build_groups_dicts(model)['links_weirs']['header']

    global RELEVANT_LINKS_WEIRS
    RELEVANT_LINKS_WEIRS = []

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        if len(RELEVANT_LINKS) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0].lower() in RELEVANT_LINKS:
                                RELEVANT_LINKS_WEIRS.append(line[0].split()[0])
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('weirs','df created!')
    return df


def orifices_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['links_orifices']['start']
    skip_rows = build_groups_dicts(model)['links_orifices']['line_to_skip']
    header = build_groups_dicts(model)['links_orifices']['header']

    global RELEVANT_LINKS_ORIFICES
    RELEVANT_LINKS_ORIFICES = []

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        if len(RELEVANT_LINKS) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0].lower() in RELEVANT_LINKS:
                                RELEVANT_LINKS_ORIFICES.append(line[0].split()[0])
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('orifices','df created!')
    return df



def get_nodes_from_links(model, model_id):
    conduits_df = conduits_to_dfs(model, model_id)
    orifices_df = orifices_to_dfs(model, model_id)
    weirs_df = weirs_to_dfs(model, model_id)

    links_dfs = [
        conduits_df,
        orifices_df,
        weirs_df
    ]

    nodes = []
    for df in links_dfs:
        for col in [col for col in df.columns if 'node' in col]:
            nodes += df[col].unique().tolist()

    return nodes

#cambio de coordenadas
def convert_coords(coord_tuple):
    transformer = pyproj.Transformer.from_crs(crs_from='epsg:' + epsg_modelo, crs_to='epsg:4326')
    lon, lat = transformer.transform(coord_tuple[0], coord_tuple[1])
    return (lon,lat)

def nodes_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    global RELEVANT_NODES
    RELEVANT_NODES = get_nodes_from_links(model, model_id)

    start = build_groups_dicts(model)['nodes_coordinates']['start']
    skip_rows = build_groups_dicts(model)['nodes_coordinates']['line_to_skip']
    header = build_groups_dicts(model)['nodes_coordinates']['header']

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif (i == start + 1):
                    #     headers = line

                    else:
                        if len(RELEVANT_NODES) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0] in RELEVANT_NODES:
                                contents.append(line[0].split())


    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)

    cols =['lat', 'lon']
    coords = []

    coordinates = [(j[0], j[1]) for i,j in df[['x_coord', 'y_coord']].iterrows()]

    pool = multiprocessing.Pool(8)
    coords.append(pool.map(convert_coords, coordinates))
    pool.close()
    pool.join()

    # for i in df[['x_coord', 'y_coord']].iterrows():
    #     coords.append(convert_coords(i[1]))


    # from pyproj import Transformer

    # def convert_coords(coord_tuple):
    #     global coords
    #     transformer = Transformer.from_crs(crs_from='epsg:5348' , crs_to='epsg:4326')
    #     lon, lat = transformer.transform(coord_tuple[0], coord_tuple[1])
    #     coords.append((lon, lat, coord_tuple[2]))

    #     return coords


    # import concurrent.futures

    # coords = []
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     for result in executor.map(convert_coords, [(i[1], i[2], i[3]) for i in coordinates]):
    #         pass

    # coords = result



    df = pd.concat([df, pd.DataFrame(coords[0], columns=cols)], axis=1)

    print('nodes','df created!')
    return df


def outfalls_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    global RELEVANT_NODES
    RELEVANT_NODES = get_nodes_from_links(model, model_id)

    start = build_groups_dicts(model)['nodes_outfalls']['start']
    skip_rows = build_groups_dicts(model)['nodes_outfalls']['line_to_skip']
    header = build_groups_dicts(model)['nodes_outfalls']['header']

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif (i == start + 1):
                    #     headers = line

                    else:
                        if len(RELEVANT_NODES) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0] in RELEVANT_NODES:
                                contents.append(line[0].split())


    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)

    print('outfalls','df created!')
    return df


def junctions_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    global RELEVANT_NODES
    RELEVANT_NODES = get_nodes_from_links(model, model_id)

    start = build_groups_dicts(model)['nodes_junctions']['start']
    skip_rows = build_groups_dicts(model)['nodes_junctions']['line_to_skip']
    header = build_groups_dicts(model)['nodes_junctions']['header']

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif (i == start + 1):
                    #     headers = line

                    else:
                        if len(RELEVANT_NODES) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0] in RELEVANT_NODES:
                                contents.append(line[0].split())


    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)

    print('junctions','df created!')
    return df


def storage_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    global RELEVANT_NODES
    RELEVANT_NODES = get_nodes_from_links(model, model_id)

    start = build_groups_dicts(model)['nodes_storage']['start']
    skip_rows = build_groups_dicts(model)['nodes_storage']['line_to_skip']
    header = build_groups_dicts(model)['nodes_storage']['header']

    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif (i == start + 1):
                    #     headers = line

                    else:
                        if len(RELEVANT_NODES) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[0] in RELEVANT_NODES:
                                contents.append(line[0].split())


    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)

    print('storage','df created!')
    return df


def subcatch_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['subcatchments']['start']
    skip_rows = build_groups_dicts(model)['subcatchments']['line_to_skip']
    header = build_groups_dicts(model)['subcatchments']['header']

    global RELEVANT_SUBCATCHMENTS
    RELEVANT_SUBCATCHMENTS = []


    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        relevant_nodes = [node for node in RELEVANT_NODES]
                        if len(relevant_nodes) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[2] in relevant_nodes:
                                RELEVANT_SUBCATCHMENTS.append(line[0].split()[0])
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('subcatch','df created!')
    return df


def infiltration_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['infiltration']['start']
    skip_rows = build_groups_dicts(model)['infiltration']['line_to_skip']
    header = build_groups_dicts(model)['infiltration']['header']


    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        relevant_nodes = [node for node in RELEVANT_NODES]
                        if len(relevant_nodes) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[2] in relevant_nodes:
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('infiltration','df created!')
    return df


def subareas_to_dfs(model, model_id):
    """ Read a .CSV into a Pandas DataFrame until a blank line is found, then stop.
    """
    start = build_groups_dicts(model)['subareas']['start']
    skip_rows = build_groups_dicts(model)['subareas']['line_to_skip']
    header = build_groups_dicts(model)['subareas']['header']


    df = pd.DataFrame()
    with open(model, newline='') as f:
        contents = []
        r = csv.reader(f)
        for i, line in enumerate(r):
            if i >  start + 1:
                if i != skip_rows:
                    if not line:
                        break
                    # elif i == start + 1:
                    #     headers = line
                    else:
                        relevant_nodes = [node for node in RELEVANT_NODES]
                        if len(relevant_nodes) == 0:
                            contents.append(line[0].split())
                        else:
                            if line[0].split()[2] in relevant_nodes:
                                contents.append(line[0].split())

    df = pd.DataFrame(data = contents, columns= [col.lower().replace("-", "_").replace("%", "").replace(" ", "_") for col in header],)
    df.insert(0, 'model_id', model_id)
    print('subareas','df created!')
    return df




# functions to interact with the db
def df_to_db(df, engine, table, if_exists):
    if update_model:

        session_factory = sessionmaker(bind=engine_base_ina)
        Session = scoped_session(session_factory)

        df.to_sql(table, engine, index=False, if_exists=if_exists)

        Session.remove()

        print(elements,'to database!')



def query_to_db(engine, query):
    session_factory = sessionmaker(bind=engine_base_ina)
    Session = scoped_session(session_factory)

    engine.execute(query)

    Session.remove()





# def group_type_to_dfs(model, model_id, group, id_col, col_to_check, own_relevant__list, relevant_dependecy_list):


# def inp_to_db(model, model_id, engine):
#     for gro
#     df_to_db(
#         group_type_to_dfs(
#             model,
#             model_id),
#         engine,
#         'coordinates',
#         'append'
#         )
#     print('Listo coordinates!')








def inp_to_db(model, model_id, engine):
    df_to_db(
        nodes_to_dfs(
            model,
            model_id),
        engine,
        'nodes_coordinates',
        'append'
        )
    print('Listo coordinates!')
    for element in [col for col in elements if col != 'coordinates']:
        if element == 'links_conduits':
            df_to_db(
                conduits_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo conduits!')
        if element == 'links_orifices':
            df_to_db(
                orifices_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo orifices!')
        if element == 'links_weirs':
            df_to_db(
                weirs_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo weirs!')
        if element == 'nodes_junctions':
            df_to_db(
                junctions_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo junctions!')
        if element == 'nodes_storage':
            df_to_db(
                storage_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo storages!')
        if element == 'nodes_outfalls':
            df_to_db(
                outfalls_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo outfalls!')
        if element == 'subareas':
            df_to_db(
                subareas_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo subareas!')
        if element == 'infiltration':
            df_to_db(
                infiltration_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo infiltration!')
        if element == 'subcatchments':
            df_to_db(
                subcatch_to_dfs(
                    model,
                    model_id,),
                engine,
                element,
                'append'
                )
            print('Listo subcatch!')


def time_series_vars_to_db_multiproc(model_out, tipo, evento, conn, sub_set, cols_tipo):
    """ Esta función genera un diccionario con todas las series temporales del modelo para una tipo.
        El diccionario tiene la siguiente estructura:
            {tipo :
                {parte:
                    {variable: serie}
                }
            }
        Argumentos:
            model.out: el archivo .out que devuelve swmm
            tipo: str --> The type are "subcatchment", "node", "link", "pollutant", "system".
            evento: ID del evento
        Return:
            dicc
            # hace un insert en la base de datos
    """
    series = {tipo: {}}
    partes = set([item[1] for item in swmm.catalog(model_out) if item[0] == tipo])
    if len(sub_set) == 0:
        partes = [parte for parte in partes]
    else:
        partes = [parte for parte in partes if parte in sub_set]
    print('Cantidad de partes:', len(partes), partes)
    count = 0

    global time_series_vars_to_db
    # for parte in partes:
    def time_series_vars_to_db(parte):
        nonlocal count
        series_df = swmm.extract(model_out, tipo + ',' + parte + ',')
        series_df.columns = [col[len(tipo + '_' + parte + '_'):].lower() for col in series_df.columns]
        series_df.reset_index(inplace=True)
        series_df = series_df.rename({'index':'elapsed_time'}, axis=1)
        series_df[tipo + '_id'] = parte
        series_df['event_id'] = evento
        series_df = series_df[cols_tipo]

        series.get(tipo).update({parte: series_df})
        print(tipo, parte)

        tabla = 'events_' + tipo + 's'

        # session_factory = sessionmaker(bind=engine_base_ina)
        # Session = scoped_session(session_factory)
        engine_base_ina.dispose()
        series.get(tipo).get(parte).to_sql(tabla, conn, index=False, if_exists='append')

        # Session.remove()

        count += 1
        print(tipo + ': ' + str(count) + ' de ' + str(len(partes)))

        return series
    

    pool = multiprocessing.Pool(8)
    pool.map(time_series_vars_to_db, partes)
    pool.close()
    pool.join()


def out_to_db(model_out, event, engine):
    for tipo in RELEVANT_GROUP_TYPES_OUT:
        if tipo == 'link':
            time_series_vars_to_db_multiproc(model_out, tipo, event, engine, RELEVANT_LINKS, MODEL_OUT_COLS['LINKS_COLS'])
        elif tipo == 'node':
            time_series_vars_to_db_multiproc(model_out, tipo, event, engine, RELEVANT_NODES, MODEL_OUT_COLS['NODES_COLS'])
        elif tipo == 'subcatchment':
            time_series_vars_to_db_multiproc(model_out, tipo, event, engine, RELEVANT_SUBCATCHMENTS, MODEL_OUT_COLS['SUBCATCHMENTS_COLS'])


# def out_to_db(tipo, model_out, event, engine):
#     if tipo == 'link':
#         time_series_vars_to_db(model_out, tipo, event, engine, RELEVANT_LINKS, MODEL_OUT_COLS['LINKS_COLS'])
#     elif tipo == 'node':
#         time_series_vars_to_db(model_out, tipo, event, engine, RELEVANT_NODES, MODEL_OUT_COLS['NODES_COLS'])
#     elif tipo == 'subcatchment':
#         time_series_vars_to_db(model_out, tipo, event, engine, RELEVANT_SUBCATCHMENTS, MODEL_OUT_COLS['SUBCATCHMENTS_COLS'])


# def multi_proc(model_out, event, engine):
#     import multiprocessing
#     processes = []
#     for tipo in RELEVANT_GROUP_TYPES_OUT:
#         t = multiprocessing.Process(target=out_to_db, args=(tipo, model_out, event, engine_base_ina))
#         processes.append(t)
#         t.start()


    # for process in processes:
    #     process.join()



if __name__ == "__main__":

    # The Session object created here will be used by the function.
    starttime = time.time()

    update_raingage = True
    update_model = True
    update_event = True


    try:
        query_to_db(engine_base_ina, "INSERT INTO precipitation_event(precipitation_id) VALUES('{}')".format(precipitation_id))
    except Exception as e:
        # update_model = False
        print('precipitation already loaded!')

    load_raingages_to_db(event_folder, event_id, model_inp, model_id)

    try:
        query_to_db(engine_base_ina, "INSERT INTO models(model_id) VALUES('{}')".format(model_id))
    except Exception as e:
        # update_model = False
        print('Model already loaded!')
    try:
        query_to_db(engine_base_ina, "INSERT INTO events(event_id, model_id, precipitation_id) VALUES('{}', '{}', '{}')".format(event_id, model_id, precipitation_id))
    except Exception as e:
        # update_event = False
        print('Event, model and raingage already loaded!')

    # groups = {}

    elements =  group_start_line(model_inp).keys()

    inp_to_db(model_inp, model_id, engine_base_ina)

    out_to_db(model_out, event_id, engine_base_ina) 

    print('Listo todo!')
    print('That took {} seconds'.format(time.time() - starttime))
