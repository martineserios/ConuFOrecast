#%%
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import psycopg2
import datatable as dt
import pickle
import plotly.express as px
# from plotly.subplots import make_subplots
from collections import namedtuple, defaultdict
from datetime import datetime

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Dataset, Data


class DBconnector():
    """
    Connection to the database
    """
    #host="172.18.0.1", port = 5555, database="base-ina", user="postgres", password="postgres"
    def __init__(self, url: str, port: int, database: str, user: str, password: str) -> None:
        self.pg_conn = psycopg2.connect(host=url, port=port, database=database, user=user, password=password)
    
    def query(self, query:str):
        
        cur0 = self.pg_conn.cursor()
        cur0.execute(query)
        query_result = cur0.fetchall()
        
        return query_result


class GraphEngine():
    """
    Initializes the graph of the whole model by doing the correponding queries to the database.
    """
    def get_time_steps(self):
        time_range_query = f"""
        SELECT DISTINCT
            elapsed_time 
        FROM
            events_nodes en 
        WHERE
            event_id = '{self.event}'
        """

        cur0 = self.conn.cursor()
        cur0.execute(time_range_query)
        time_range_query_result = cur0.fetchall()

        time_range_query_result = sorted([time for time in time_range_query_result])

        time_range_query_result_formatted = sorted([time[0].strftime("%Y-%m-%d %H:%M:%S") for time in time_range_query_result])

        return time_range_query_result, time_range_query_result_formatted 


    def which_time_step(self):
        
        time_range = self.get_time_steps()[0]
        first_two = time_range[:2]
        time_step = (first_two[1][0] - first_two[0][0]).seconds // 60

        return time_step, f'{time_step} minutes'
    
    
    def get_nodes(self):
        nodes_query = f"""
        SELECT
            node_id
        FROM
            nodes_coordinates nc 
        WHERE
            nc.model_id = '{self.model}'
        """

        cur1 = self.conn.cursor()
        cur1.execute(nodes_query)
        nodes_query_result = cur1.fetchall()

        nodes = sorted([node for nodes in nodes_query_result for node in nodes])

        return nodes 


    def nodal_linkage_query(self, elapsed_time:str, attrs:dict, persist:bool=True):
        
        link_attrs = ', '.join(['edge', 'link_id', 'from_node', 'to_node', 'elapsed_time']) + ', ' + ', '.join(attrs['edges'])

        nodal_linkage_query = f"""
        WITH links_event AS 
        (
            SELECT
                *
            FROM
                events_links el
            WHERE
                el.event_id = '{self.event}'
                AND el.elapsed_time = '{elapsed_time}'
                AND el.flow_rate != 0
        )
        ,
        links_conduits_model AS
        (
            SELECT
                *
            FROM
                links_conduits AS lc
            WHERE
                lc.model_id = '{self.model}'
        )
        ,
        links_orifices_model AS
        (
            SELECT
                *
            FROM
                links_orifices AS lo
            WHERE
                lo.model_id = '{self.model}'
        )
        ,
        links_weirs_model AS
        (
            SELECT
                *
            FROM
                links_weirs AS lw
            WHERE
                lw.model_id = '{self.model}'
        )
        ,
        links_types_event AS
        (
            SELECT
        (
                CASE
                    WHEN
                        links.flow_rate > 0
                    THEN
                        concat(conduits.from_node, '->', conduits.to_node)
                    ELSE
                        concat(conduits.to_node, '->', conduits.from_node)
                END
        ) AS edge, links.link_id,
                (
                    CASE
                        WHEN
                            links.flow_rate > 0
                        THEN
                            conduits.from_node
                        ELSE
                            conduits.to_node
                    END
                )
                AS from_node,
                (
                    CASE
                        WHEN
                            links.flow_rate < 0
                        THEN
                            conduits.from_node
                        ELSE
                            conduits.to_node
                    END
                )
                AS to_node, elapsed_time, ABS(flow_rate) AS flow_rate , flow_depth, ABS(flow_velocity ) AS flow_velocity, froude_number, capacity, conduits.length, conduits.roughness
            FROM
                links_event AS links
                LEFT JOIN
                    links_conduits_model AS conduits
                    ON conduits.conduit_id = links.link_id
            WHERE
                from_node NOTNULL
            UNION
            SELECT
        (
                CASE
                    WHEN
                        links.flow_rate > 0
                    THEN
                        concat(orifices.from_node, '->', orifices.to_node)
                    ELSE
                        concat(orifices.to_node, '->', orifices.from_node)
                END
        ) AS edge, links.link_id,
                (
                    CASE
                        WHEN
                            links.flow_rate > 0
                        THEN
                            orifices.from_node
                        ELSE
                            orifices.to_node
                    END
                )
                AS from_node,
                (
                    CASE
                        WHEN
                            links.flow_rate < 0
                        THEN
                            orifices.from_node
                        ELSE
                            orifices.to_node
                    END
                )
                AS to_node, elapsed_time, ABS(flow_rate) AS flow_rate , flow_depth, ABS(flow_velocity ) AS flow_velocity, froude_number, capacity, 0 AS length, 0 AS roughness
            FROM
                links_event AS links
                LEFT JOIN
                    links_orifices_model AS orifices
                    ON orifices.orifice_id = links.link_id
            WHERE
                from_node NOTNULL
            UNION
            SELECT
        (
                CASE
                    WHEN
                        links.flow_rate > 0
                    THEN
                        concat(weirs.from_node, '->', weirs.to_node)
                    ELSE
                        concat(weirs.to_node, '->', weirs.from_node)
                END
        ) AS edge, links.link_id,
                (
                    CASE
                        WHEN
                            links.flow_rate > 0
                        THEN
                            weirs.from_node
                        ELSE
                            weirs.to_node
                    END
                )
                AS from_node,
                (
                    CASE
                        WHEN
                            links.flow_rate < 0
                        THEN
                            weirs.from_node
                        ELSE
                            weirs.to_node
                    END
                )
                AS to_node, elapsed_time, ABS(flow_rate) AS flow_rate , flow_depth, ABS(flow_velocity ) AS flow_velocity, froude_number, capacity, 0 AS length, 0 AS roughness
            FROM
                links_event AS links
                LEFT JOIN
                    links_weirs_model AS weirs
                    ON weirs.weir_id = links.link_id
            WHERE
                from_node NOTNULL
        )
--        , rain_mdata AS
--        (
--            SELECT
--                *
--            FROM
--                raingages_metadata rm
--           WHERE
--                rm.precipitation_id = '{self.precip}'
--        )
--        ,
--        rain_tseries AS
--        (
--            SELECT
--                *
--            FROM
--                raingages_timeseries rt
--            WHERE
--                rt.precipitation_id = '{self.precip}'
--                AND rt.elapsed_time = '{elapsed_time}'
--        )
--        ,
--        rain AS
--        (
--            SELECT
--                rt.raingage_id,
--                rt.elapsed_time,
--                rt.VALUE,
--                rm.format,
--               rm.unit
--            FROM
--                raingages_timeseries rt
--                JOIN
--                   raingages_metadata AS rm
--                    ON rt.precipitation_id = rm.precipitation_id
--        )
        ,
        subc AS
        (
            SELECT
                *
            FROM
                subcatchments s2
            WHERE
                s2.model_id = '{self.model}'
        )
        ,
        event_subc AS
        (
            SELECT
                *
            FROM
                events_subcatchments es
            WHERE
                es.event_id = '{self.event}'
                AND es.elapsed_time = '{elapsed_time}'
        )
        ,
        event_subc_outlet AS
        (
            SELECT DISTINCT
                subc.subcatchment_id,
                subc.outlet,
                subc.raingage_id,
                elapsed_time,
                event_subc.rainfall
            FROM
                subc
                INNER JOIN
                    event_subc
                    ON subc.subcatchment_id = event_subc.subcatchment_id
        )
--        ,
--        event_subc_rainfall AS
--        (
--            SELECT DISTINCT
--                eso.*,
--                rain.VALUE,
--                rain.format,
--                rain.unit
--            FROM
--                event_subc_outlet eso
--                INNER JOIN
--                    rain
--                    ON rain.raingage_id = eso.raingage_id
--                    AND rain.elapsed_time = eso.elapsed_time
--        )
        ,
        final AS
        (
        SELECT DISTINCT
            lte.*,
            COALESCE (esr.rainfall, 0) AS rainfall--,
--            COALESCE (esr.VALUE, 0) AS rainfall_acc
        FROM
            links_types_event lte
            LEFT JOIN
                event_subc_outlet esr
                ON lte.from_node = esr.outlet
                AND lte.elapsed_time = esr.elapsed_time
        )
        SELECT {link_attrs}
        FROM final
        """

        cur1 = self.conn.cursor()
        cur1.execute(nodal_linkage_query)
        nodal_linkage_query_result = cur1.fetchall()
        
        if persist:
            self.nodal_linkage_query_results[f'{self.event}_{elapsed_time}'] = nodal_linkage_query_result
        else:
            return nodal_linkage_query_result



    def get_nodal_linkage(self, elapsed_time:str, attrs:dict, persist:bool=True):

        link_attrs = ','.join(['edge', 'link_id', 'from_node', 'to_node', 'elapsed_time']) + ',' + ','.join(attrs['edges'])


        if persist:
            try:
                self.nodal_linkage_dict[f'{self.event}_{elapsed_time}']
            
            except:
                self.nodal_linkage_query(elapsed_time, attrs)
                    
                nodal_linkage = {i[0]:
                dict(zip(link_attrs.split(','), i))
                         for i in self.nodal_linkage_query_results[f'{self.event}_{elapsed_time}']
                 }

                self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'] = nodal_linkage
        else:
            query = self.nodal_linkage_query(elapsed_time, attrs, persist=False)
                    
            nodal_linkage = {i[0]:
            dict(zip(link_attrs.split(','), i))
                    for i in query
            }

            return nodal_linkage


    def nodal_data_query(self, elapsed_time:str, attrs:dict, persist:bool=True):

        node_attrs = ','.join(['node_id', 'subcatchment_id', 'elapsed_time', 'depth_above_invert']) + ',' + ','.join(attrs['nodes'])

        nodal_data_query = f"""
        WITH model_node_coordinates AS 
        (
            SELECT
                * 
            FROM
                nodes_coordinates AS nc 
            WHERE
                nc.model_id = '{self.model}' 
        )
        ,
        junctions AS 
        (
            SELECT
                * 
            FROM
                nodes_junctions nj 
            WHERE
                nj.model_id = '{self.model}' 
        )
        ,
        storages AS 
        (
            SELECT
                * 
            FROM
                nodes_storage ns 
            WHERE
                ns.model_id = '{self.model}' 
        )
        ,
        outfalls AS 
        (
            SELECT
                * 
            FROM
                nodes_outfalls AS no2 
            WHERE
                no2.model_id = '{self.model}' 
        )
        ,
        nodes AS 
        (
            SELECT
                mnc.node_id,
                mnc.lat,
                mnc.lon,
                j.elevation,
                j.init_depth,
                j.max_depth 
            FROM
                model_node_coordinates mnc 
                JOIN
                    junctions j 
                    ON mnc.node_id = j.junction_id 
            WHERE
                elevation NOTNULL 
            UNION ALL
            SELECT
                mnc.node_id,
                mnc.lat,
                mnc.lon,
                s.elevation,
                s.init_depth,
                s.max_depth 
            FROM
                model_node_coordinates mnc 
                JOIN
                    storages s 
                    ON mnc.node_id = s.storage_id 
            WHERE
                elevation NOTNULL 
            UNION ALL
            SELECT
                mnc.node_id,
                mnc.lat,
                mnc.lon,
                o.elevation,
                0 AS init_depth,
                0 AS max_depth 
            FROM
                model_node_coordinates mnc 
                JOIN
                    outfalls o 
                    ON mnc.node_id = o.outfall_id 
            WHERE
                elevation NOTNULL 
        )
        ,
        subcatch AS 
        (
            SELECT
                * 
            FROM
                subcatchments s 
            WHERE
                s.model_id = '{self.model}' 
        )
        ,
        event_nodes AS 
        (
            SELECT
                * 
            FROM
                events_nodes en 
            WHERE
                event_id = '{self.event}' 
                AND en.elapsed_time = '{elapsed_time}' 
        )
        ,
        event_subc AS 
        (
            SELECT
                * 
            FROM
                events_subcatchments es 
            WHERE
                es.event_id = '{self.event}' 
                AND es.elapsed_time = '{elapsed_time}' 
        )
        ,
        event_subc_outlet AS 
        (
            SELECT
                event_subc.*,
                subcatch.outlet,
                subcatch.raingage_id 
            FROM
                subcatch 
                LEFT JOIN
                    event_subc 
                    ON subcatch.subcatchment_id = event_subc.subcatchment_id 
        )
        ,
        nodal_out_data AS
        (
            SELECT
                en.node_id,
                COALESCE (subcatchment_id, 'SIN CUENCA DE APORTE') AS subcatchment_id,
                en.elapsed_time,
                en.depth_above_invert,
                en.flow_lost_flooding,
                en.hydraulic_head,
                en.lateral_inflow,
                en.total_inflow,
                en.volume_stored_ponded,
                COALESCE (eso.rainfall, 0) AS rainfall,
                COALESCE (eso.evaporation_loss, 0) AS evaporation_loss,
                COALESCE (eso.runoff_rate, 0) AS runoff_rate,
                COALESCE (eso.infiltration_loss, 0) AS infiltration_loss 
            FROM
                event_nodes AS en 
                LEFT JOIN
                    event_subc_outlet AS eso 
                    ON eso.elapsed_time = en.elapsed_time 
                    AND eso.outlet = en.node_id 
        )
        ,
        nodal_inp_data AS 
        (
            SELECT
                nodes.*,
                COALESCE (s.area, 0) AS area,
                COALESCE (s.imperv, 0) AS imperv,
                COALESCE (s.slope, 0) AS slope,
                COALESCE (s.width, 0) AS width,
                COALESCE (s.curb_len, 0) AS curb_len,
                COALESCE (s.raingage_id, '') AS raingage_id 
            FROM
                nodes 
                LEFT JOIN
                    subcatch s 
                    ON s.outlet = nodes.node_id 
        )
        ,
        nodal_data AS 
        (
            SELECT
                nod.*,
                nid.lon,
                nid.lat,
                nid.elevation,
                nid.init_depth,
                nid.max_depth,
                nid.area,
                nid.imperv,
                nid.slope,
                nid.width,
                nid.curb_len
--                nid.raingage_id 
            FROM
                nodal_out_data AS nod 
                LEFT JOIN
                    nodal_inp_data AS nid 
                    ON nod.node_id = nid.node_id 
        )
--        ,
--        rain_mdata AS 
--        (
--            SELECT
--                * 
--            FROM
--                raingages_metadata rm 
--            WHERE
--                rm.precipitation_id = '{self.precip}' 
--        )
--        ,
--        rain_tseries AS 
--        (
--            SELECT
--                * 
--            FROM
--                raingages_timeseries rt 
--            WHERE
--                rt.precipitation_id = '{self.precip}' 
--        )
--        ,
--        rain AS
--        (
--            SELECT
--                rt.raingage_id,
--                rt.elapsed_time,
--                COALESCE (rt.VALUE, 0) AS rainfall_acc,
--                rm.format,
--                rm.unit 
--            FROM
--                raingages_timeseries rt 
--                JOIN
--                    raingages_metadata AS rm 
--                    ON rt.precipitation_id = rm.precipitation_id 
--        )
--        ,
--        final AS
--        (
--        SELECT DISTINCT
--            nd.*,
--            COALESCE (r.rainfall_acc, 0) AS rainfall_acc,
--            COALESCE (r.format, '') AS format,
--            COALESCE (r.unit, '') AS unit 
--       FROM
--            nodal_data nd 
--            LEFT JOIN
--               rain r 
--                ON nd.raingage_id = r.raingage_id 
--               AND nd.elapsed_time = r.elapsed_time
--        )
        SELECT {node_attrs}
        FROM nodal_data
        """

        cur2 = self.conn.cursor()
        cur2.execute(nodal_data_query)
        nodal_data_query_result = cur2.fetchall()

        if persist:
            self.nodal_data_query_results[f'{self.event}_{elapsed_time}'] = nodal_data_query_result
        else:
            return nodal_data_query_result



    def get_nodal_data(self, elapsed_time:str, attrs:dict, persist:bool=True):

        node_attrs = ','.join(['node_id', 'subcatchment_id', 'elapsed_time', 'depth_above_invert']) + ',' + ','.join(attrs['nodes'])

        if persist:
            try:
                self.nodal_data_dict[f'{self.event}_{elapsed_time}']

            except:
                self.nodal_data_query(elapsed_time, attrs)
                
                nodal_data = {
                    i[0]: dict(zip(node_attrs.split(','), i))
                    for i in self.nodal_data_query_results[f'{self.event}_{elapsed_time}']
                }

                self.nodal_data_dict[f'{self.event}_{elapsed_time}'] = nodal_data
        else:
            query = self.nodal_data_query(elapsed_time, attrs, persist=False)
            
            nodal_data = {i[0]: dict(zip(node_attrs.split(','), i))
            for i in query
            }
            
            return nodal_data



    # graph creation

    def build_digraph(self, elapsed_time:str, attrs:dict, persist:bool=True):
        if persist:
            self.get_nodal_data(elapsed_time, attrs, persist=True)
            self.get_nodal_linkage(elapsed_time, attrs, persist=True)

            #target definition
            def risk_classes(level):
                high_risk_level = 0.25
                mid_risk_level = 0.15

                if level < mid_risk_level:
                    return 0
                elif (level >= mid_risk_level) & (level < high_risk_level):
                    return 1
                else:
                    return 2 

            try:
                self.digraphs[f'{self.event}_{elapsed_time}']

            except:
                DG = nx.DiGraph(elapsed_time = elapsed_time, model=self.model, event=self.event)
                [DG.add_edge(i[1]['from_node'], i[1]['to_node'], **i[1]) for i in self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'].items()]
                [DG.add_node(i[0], **i[1]) for i in self.nodal_data_dict[f'{self.event}_{elapsed_time}'].items()]

                #target definition
                [DG.add_node(i[0], **{'target': risk_classes(i[1]['depth_above_invert'])}) for i in self.nodal_data_dict[f'{self.event}_{elapsed_time}'].items()]

                if persist:
                    self.digraphs[f'{self.event}_{elapsed_time}'] = DG
                    self.num_nodes[f'{self.event}_{elapsed_time}'] = len(DG.nodes())
                    self.num_edges[f'{self.event}_{elapsed_time}'] = len(DG.edges())


        
        else:
            nodal_data = self.get_nodal_data(elapsed_time, attrs, persist=False)
            nodal_linkage = self.get_nodal_linkage(elapsed_time, attrs, persist=False)

            #target definition
            def risk_classes(level):
                high_risk_level = 0.25
                mid_risk_level = 0.15

                if level < mid_risk_level:
                    return 0
                elif (level >= mid_risk_level) & (level < high_risk_level):
                    return 1
                else:
                    return 2 


            DG = nx.DiGraph(elapsed_time = elapsed_time, model=self.model, event=self.event)
            [DG.add_edge(i[1]['from_node'], i[1]['to_node'], **i[1]) for i in nodal_linkage.items()]
            [DG.add_node(i[0], **i[1]) for i in nodal_data.items()]

            #target definition
            [DG.add_node(i[0], **{'target': risk_classes(i[1]['depth_above_invert'])}) for i in nodal_data.items()]
            self.num_nodes[f'{self.event}_{elapsed_time}'] = len(DG.nodes())
            self.num_edges[f'{self.event}_{elapsed_time}'] = len(DG.edges())
            
            return DG



    def build_coordinates_dict(self, elevation:bool=False):
        nodes_coordinates_query = f"""
        WITH node_coordinates_model AS
        (
            SELECT
                * 
            FROM
                nodes_coordinates AS nc 
            WHERE
                nc.model_id = '{self.model}' 
        )
        SELECT
            nc.node_id,
            nc.lat,
            nc.lon,
            nj.elevation,
            nj.init_depth,
            nj.max_depth 
        FROM
            node_coordinates_model nc 
            JOIN
                nodes_junctions nj 
                ON nc.node_id = nj.junction_id 
        WHERE
            nj.model_id = '{self.model}' 
        UNION ALL
        SELECT
            nc.node_id,
            nc.lat,
            nc.lon,
            ns.elevation,
            ns.init_depth,
            ns.max_depth 
        FROM
            node_coordinates_model nc 
            JOIN
                nodes_storage ns 
                ON nc.node_id = ns.storage_id 
        WHERE
            ns.model_id = '{self.model}' 
        UNION ALL
        SELECT
            nc.node_id,
            nc.lat,
            nc.lon,
            no2.elevation,
            0 AS init_depth,
            0 AS max_depth 
        FROM
            node_coordinates_model nc 
            JOIN
                nodes_outfalls no2 
                ON nc.node_id = no2.outfall_id 
        WHERE
            no2.model_id = '{self.model}'
        """

        cur3 = self.conn.cursor()
        cur3.execute(nodes_coordinates_query)
        coordinates_query_result = cur3.fetchall()

        if elevation:
            coordinates = {i[0]: {'lat':i[1], 'lon':i[2], 'elevation':i[3]} for i in coordinates_query_result}
            return coordinates

        else:
            coordinates = {i[0]: {'lat':i[1], 'lon':i[2]} for i in coordinates_query_result}
            return coordinates




    def __init__(self, model:str, event:str, precip:str, conn) -> None:
        self.conn = conn.pg_conn
        # self.elapsed_time = elapsed_time
        self.model = model
        self.event = event
        self.precip = precip
        self.time_range = self.get_time_steps()
        self.nodal_linkage_query_results = {}
        self.nodal_linkage_dict = {}
        self.nodal_data_query_results = {}
        self.nodal_data_dict = {}
        self.digraphs = {}
        self.sub_digraphs = {}
        self.pos_dict = self.build_coordinates_dict()
        self.num_nodes = defaultdict(int)
        self.num_edges = defaultdict(int)
        self.torch_data = {}



    def build_subgraph(self, node:str, elapsed_time:str, attrs:dict, acc_data:bool, persist:bool=True):
        
        try:
            self.digraphs[f'{self.event}_{elapsed_time}']
        except:
            self.build_digraph(elapsed_time, attrs)

        if persist:
            try:
                self.sub_digraphs[f'{self.event}_{node}_{elapsed_time}']
            
            except:
                preds_list = [(i[0],i[1]) for i in nx.edge_dfs(self.digraphs[f'{self.event}_{elapsed_time}'], node, 'reverse')]
                if len(preds_list) == 0:
                    preds_list = [node]

                graph_preds = nx.DiGraph(elapsed_time = elapsed_time, model= self.model, outlet_node = node)

                # own node data, for the cases without preds
                graph_preds.add_node(node, **self.nodal_data_dict[f'{self.event}_{elapsed_time}'][node])


                #target definition
                def risk_classes(level):
                    high_risk_level = 0.25
                    mid_risk_level = 0.15

                    if level < mid_risk_level:
                        return 0
                    elif level == None:
                        return 0
                    elif (level >= mid_risk_level) & (level < high_risk_level):
                        return 1
                    else:
                        return 2 


                if isinstance(preds_list[0], tuple):
                    [graph_preds.add_edge(edge[0], edge[1], **self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'][edge[0] + '->' + edge[1]]) for edge in preds_list]
                    [graph_preds.add_node(i, **self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]
                    [graph_preds.add_node(
                        i, **{'target': risk_classes(self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]['depth_above_invert'])}
                        ) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]
                else:
                    [graph_preds.add_node(i, **self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]) for i in preds_list]
                    [graph_preds.add_node(
                        i, **{'target': risk_classes(self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]['depth_above_invert'])}
                        ) for i in preds_list]


                def division_exception(a, b, default_value):
                    try:
                        return a / b
                    except:
                        return default_value

                if acc_data:
                    vars_acc = {
                    'area_aporte_ha': round(sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                    'perm_media_%':
                    round(
                    division_exception(sum(
                    [graph_preds.nodes()[i]['area'] * graph_preds.nodes()[i]['imperv']
                    for i in graph_preds.nodes()]
                    )
                    , sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),0),
                    4),

                    'manning_medio_flow_s/m^1/3':
                    round(
                    division_exception(sum(
                    [
                    graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                    * graph_preds.edges()[edge[0], edge[1]]['length']
                    * graph_preds.edges()[edge[0], edge[1]]['roughness']
                    for edge in graph_preds.edges()
                    ])
                    , sum([graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                    * graph_preds.edges()[edge[0], edge[1]]['length']
                    for edge in graph_preds.edges()
                    ]),0),
                    3),

                    'manning_medio_s/m^1/3':
                    round(
                    division_exception(sum(
                    [
                    graph_preds.edges()[edge[0], edge[1]]['length']
                    * graph_preds.edges()[edge[0], edge[1]]['roughness']
                    for edge in graph_preds.edges()
                    ])
                    , sum([graph_preds.edges()[edge[0], edge[1]]['length']
                    for edge in graph_preds.edges()
                    ]),0),
                    3),

                    # 'precip_media_mm/ha': 
                    # division_exception(
                    # round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                    # , sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),
                    # 0),

                    'infilt_media_mm/hs': round(np.average([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes()]),2),

                    # 'vol_almacenado_mm': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                    #     - sum([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes])
                    #     - sum([graph_preds.nodes()[i]['evaporation_loss'] for i in graph_preds.nodes])
                    #     - sum([graph_preds.nodes()[i]['runoff_rate'] * graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                    # 'vol_precipitado_mm_acc': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges()]),2),

                    #     'vol_precipitado_mm': round(sum([graph_preds.edges()[edge[0], edge[1]]['rainfall'] for edge in graph_preds.edges()]),2),


                    'delta_h_medio_m/m':
                    round(
                    division_exception(
                    (
                    max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                    - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                    ) , np.sqrt(10000 * sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()])),0),
                    2),

                    'pendiente_media_m/m':
                    division_exception(
                    (
                    max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                    - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                    ) , sum([graph_preds.edges()[edge[0], edge[1]]['length'] for edge in graph_preds.edges()]),
                    0)
                    }


                    graph_preds.add_node(node, **vars_acc)

                    self.sub_digraphs[self.event + '_' + node +  '_' + elapsed_time + '_acc'] = graph_preds
        
        else:
            try:
                graph = self.digraphs[f'{self.event}_{elapsed_time}']
            except:
                graph = self.build_digraph(elapsed_time, attrs, persist=True)
            
            nodal_data_dict = self.get_nodal_data(elapsed_time, attrs, persist=False)
            nodal_linkage_dict = self.get_nodal_linkage(elapsed_time, attrs, persist=False)

            preds_list = [(i[0],i[1]) for i in nx.edge_dfs(graph, node, 'reverse')]
            if len(preds_list) == 0:
                preds_list = [node]


            graph_preds = nx.DiGraph(elapsed_time = elapsed_time, model= self.model, outlet_node = node)

            # own node data, for th cases without preds
            graph_preds.add_node(node, **nodal_data_dict[node])

            #target definition
            def risk_classes(level):
                high_risk_level = 0.25
                mid_risk_level = 0.15

                if level < mid_risk_level:
                    return 0
                elif (level >= mid_risk_level) & (level < high_risk_level):
                    return 1
                else:
                    return 2 


            if isinstance(preds_list[0], tuple):
                [graph_preds.add_edge(edge[0], edge[1], **nodal_linkage_dict[edge[0] + '->' + edge[1]]) for edge in preds_list]
                [graph_preds.add_node(i, **nodal_data_dict[i]) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]
                [graph_preds.add_node(
                    i, **{'target': risk_classes(nodal_data_dict[i]['depth_above_invert'])}
                    ) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]
            else:
                [graph_preds.add_node(i, **nodal_data_dict[i]) for i in preds_list]
                [graph_preds.add_node(
                    i, **{'target': risk_classes(nodal_data_dict[i]['depth_above_invert'])}
                    ) for i in preds_list]



            def division_exception(a, b, default_value):
                try:
                    return a / b
                except:
                    return default_value

            if acc_data:
                vars_acc = {
                'area_aporte_ha': round(sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                'perm_media_%':
                round(
                division_exception(sum(
                [graph_preds.nodes()[i]['area'] * graph_preds.nodes()[i]['imperv']
                for i in graph_preds.nodes()]
                )
                , sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),0),
                4),

                'manning_medio_flow_s/m^1/3':
                round(
                division_exception(sum(
                [
                graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                * graph_preds.edges()[edge[0], edge[1]]['length']
                * graph_preds.edges()[edge[0], edge[1]]['roughness']
                for edge in graph_preds.edges()
                ])
                , sum([graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                * graph_preds.edges()[edge[0], edge[1]]['length']
                for edge in graph_preds.edges()
                ]),0),
                3),

                'manning_medio_s/m^1/3':
                round(
                division_exception(sum(
                [
                graph_preds.edges()[edge[0], edge[1]]['length']
                * graph_preds.edges()[edge[0], edge[1]]['roughness']
                for edge in graph_preds.edges()
                ])
                , sum([graph_preds.edges()[edge[0], edge[1]]['length']
                for edge in graph_preds.edges()
                ]),0),
                3),

                # 'precip_media_mm/ha': 
                # division_exception(
                # round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                # , sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),
                # 0),

                # '     infilt_media_mm/hs': round(np.average([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes()]),2),

                # 'vol_almacenado_mm': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                #     - sum([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes])
                #     - sum([graph_preds.nodes()[i]['evaporation_loss'] for i in graph_preds.nodes])
                #     - sum([graph_preds.nodes()[i]['runoff_rate'] * graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                # 'vol_precipitado_mm_acc': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges()]),2),

                #     'vol_precipitado_mm': round(sum([graph_preds.edges()[edge[0], edge[1]]['rainfall'] for edge in graph_preds.edges()]),2),


                'delta_h_medio_m/m':
                round(
                division_exception(
                (
                max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                ) , np.sqrt(10000 * sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()])),0),
                2),

                'pendiente_media_m/m':
                division_exception(
                (
                max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                ) , sum([graph_preds.edges()[edge[0], edge[1]]['length'] for edge in graph_preds.edges()]),
                0)
                }


                graph_preds.add_node(node, **vars_acc)
                return graph_preds
        
            else:
                return graph_preds



    def graph_to_torch_tensor(self, elapsed_time:str, attrs_dict:dict, raw_data_folder:str, detailed:bool=False, to_pickle:bool=True,):
        """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Took it from torch_geometric.data.Data

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
        """

        # node_attrs = attrs_dict['nodes']
        # edge_attrs = attrs_dict['edges']
        # graphtensors = {}
        # train_DataLoaders = {}
        # test_DataLoaders = {}


        DG = self.build_digraph(elapsed_time=elapsed_time, attrs=attrs_dict, persist=False)

        graph_ = DG.copy()
        graph_ = nx.convert_node_labels_to_integers(graph_)
        edge_index = torch.tensor(list(graph_.edges)).t().contiguous()

        torch_data = defaultdict(int)
        # torch_data['y'] = DG.nodes()[node]['target']
        # graph_target = torch_data['y']
  
        if detailed:
            for i, (_, feat_dict) in enumerate(graph_.nodes(data=True)):
                for key, value in feat_dict.items():
                    torch_data['node_' + str(key)] = [value] if i == 0 else torch_data['node_' + str(key)] + [value]

            for i, (_, _, feat_dict) in enumerate(graph_.edges(data=True)):
                for key, value in feat_dict.items():
                    torch_data['edge_' + str(key)] = [value] if i == 0 else torch_data['edge_' + str(key)] + [value]
        
        torch_data['x'] = [list(v[1].values())[4:-1] for v in graph_.nodes(data=True)]
        torch_data['x'] = [[v[1][i] for i in attrs_dict['nodes'][:-1]] for v in graph_.nodes(data=True)]

        torch_data['y'] = [list(v[1].values())[-1] for v in graph_.nodes(data=True)]
        torch_data['y'] = [v[1][attrs_dict['nodes'][-1]] for v in graph_.nodes(data=True)]


        # torch_data['edge_attrs'] = [list(v[2].values())[5:] for v in graph_.edges(data=True)]
        # torch_data['edge_attrs'] = [[v[1][i] for i in attrs_dict['edges'][-1]] for v in graph_.edges(data=True)]

        torch_data['edge_index'] = edge_index.view(2, -1)

        for key, data in torch_data.items():
            try:
                if (key == 'x'):# | (key == 'edge_attrs'):
                    # torch_data[key] = torch.tensor(item)
                    torch_data[key] = torch.tensor(data)
                elif (key == 'y'):# | (key == 'edge_attrs'):
                    # torch_data[key] = torch.tensor(item)
                    torch_data[key] = torch.tensor(data, dtype=torch.long)
                elif (key == 'edge_index') | (key == 'edge_attrs'):
                    torch_data[key] = torch.tensor(data, dtype=torch.long)
                # elif (key == 'y'):
                #     torch_data[key] = torch.tensor(data, dtype=torch.long)

            except ValueError:
                print(data)
                pass

        # torch_data = Data.from_dict(torch_data)
        # torch_data.num_nodes = graph.number_of_nodes()

        if to_pickle:
            # open a file, where you ant to store the data
            file = open(f'{raw_data_folder}/{self.event}_{elapsed_time}.gpickle', 'wb')

            # dump information to that file
            pickle.dump(torch_data, file, pickle.HIGHEST_PROTOCOL)

            # close the file
            file.close()

        else:
            return torch_data



    def subgraphs_to_torch_tensors(self, elapsed_time:str, node:str, attrs_dict:dict, \
        raw_data_folder:str, detailed:bool=False, to_pickle:bool=True,):
        """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Took it from torch_geometric.data.Data

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
        """

        node_attrs = attrs_dict['nodes']
        edge_attrs = attrs_dict['edges']
        graphtensors = {}
        train_DataLoaders = {}
        test_DataLoaders = {}


        DG = self.build_subgraph(node=node, elapsed_time=elapsed_time, attrs=attrs_dict, acc_data=False, persist=False)

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
            file = open(f'{raw_data_folder}/{self.event}_{elapsed_time}_{node}_{graph_target}.gpickle', 'wb')

            # dump information to that file
            pickle.dump(torch_data, file, pickle.HIGHEST_PROTOCOL)

            # close the file
            file.close()

        else:
            return torch_data

    

    def sub_digraphs_timeseries(self, node:str, var:str, time_step:int = 4):
        
        # hardcoded
        attrs_dict = {
            'nodes': ['area', 'imperv', 'infiltration_loss', 'elevation', 'rainfall' ],
            'edges':['flow_rate', 'length', 'roughness', ],
            }
        
        [self.build_subgraph(
            node, elapsed_time=time, attrs=attrs_dict, acc_data=True, 
            ) for time in (sorted(self.time_range[1])[::time_step])
        ]
        
        df = pd.DataFrame(
            [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S'), self.sub_digraphs[f'{self.event}_{node}_{time}_acc'].nodes()[node][var]
            ) for time in (sorted(self.time_range[1])[::time_step])])\
                .rename({0:'elapsed_time', 1: var}, axis=1).set_index('elapsed_time')
        return df        


    def multi_subgraph_tseries_viz(self, nodes:list, var:str, time_step:int):

        rainfall_step = max(self.which_time_step()[0], int((self.which_time_step()[0]) * time_step))
        
        subfig = make_subplots(specs=[[{"secondary_y": True}]])

        for i, node in enumerate(nodes):

            df_plot_0 = self.sub_digraphs_timeseries(node, 'rainfall', time_step=time_step,)
            df_plot_0 = df_plot_0.resample(f'{rainfall_step}min').mean()
            plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

            df_plot_1 = self.sub_digraphs_timeseries(node, var,time_step=time_step)
            plot_var_max = 1.5 * df_plot_1[var].max()
            plot_var_min = 1.1 * df_plot_1[var].min()

            splitted_var = var.split('_')
            plot_var_legend = ' '.join([word.capitalize() for word in splitted_var][:-1]) + f' [{splitted_var[-1]}]'




            # create two independent figures with px.line each containing data from multiple columns
            fig = px.bar(df_plot_0, y='rainfall')#, render_mode="webgl",)
            fig2 = px.line(df_plot_1, y=var)
            fig2.update_traces(line={'width':5, 'color':'#125AEF'})
            fig2.update_traces(yaxis="y2")


            # subfig.add_trace(fig, row=1, col=i+1)
            subfig.add_trace(fig2.data, row=1, col=i+1)# + fig3.data)
            subfig['layout']['yaxis1'].update(title='Precipitation intensity (mm/h)',range=[0, plot_rainfall_max], autorange='reversed')
            subfig['layout']['yaxis2'].update(title= plot_var_legend, range=[plot_var_min, plot_var_max], autorange=False)
            # subfig.for_each_trace(lambda t: t.update(marker=dict(color=['black'])))
            subfig['layout']['xaxis'].update(title='', tickformat='%d-%b %Hh')
            subfig['layout'].update(plot_bgcolor='white', font={'size':25})#, template='plotly_white')
            subfig.update_xaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
            subfig.update_yaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
            subfig.update_xaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)
            subfig.update_yaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)

            subfig['layout'].update(height=600, width=1200)
            subfig.update_layout(showlegend=False)

        return subfig



    def subgraph_tseries_viz(self, node:str, var:list, time_step:int):
        rainfall_step = max(self.which_time_step()[0], int((self.which_time_step()[0]) * time_step))

        df_plot_0 = self.sub_digraphs_timeseries(node, 'rainfall', time_step=time_step,)
        df_plot_0 = df_plot_0.resample(f'{rainfall_step}min').mean()
        plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

        df_plot_1 = self.sub_digraphs_timeseries(node, var,time_step=time_step)
        plot_var_max = 1.5 * df_plot_1[var].max()
        plot_var_min = 1.1 * df_plot_1[var].min()

        splitted_var = var.split('_')
        plot_var_legend = ' '.join([word.capitalize() for word in splitted_var][:-1]) + f' [{splitted_var[-1]}]'



        subfig = make_subplots(specs=[[{"secondary_y": True}]])

        # create two independent figures with px.line each containing data from multiple columns
        fig = px.bar(df_plot_0, y='rainfall')#, render_mode="webgl",)
        fig2 = px.line(df_plot_1, y=var)
        fig2.update_traces(line={'width':5, 'color':'#125AEF'})


        fig2.update_traces(yaxis="y2")


        subfig.add_traces(fig.data + fig2.data)# + fig3.data)
        subfig['layout']['yaxis1'].update(title='Precipitation intensity (mm/h)',range=[0, plot_rainfall_max], autorange='reversed')
        subfig['layout']['yaxis2'].update(title= plot_var_legend, range=[plot_var_min, plot_var_max], autorange=False)
        subfig.for_each_trace(lambda t: t.update(marker=dict(color=['black'])))
        subfig['layout']['xaxis'].update(title='', tickformat='%d-%b %Hh')
        subfig['layout'].update(plot_bgcolor='white', font={'size':25})#, template='plotly_white')
        subfig.update_xaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        subfig.update_yaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        subfig.update_xaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)
        subfig.update_yaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)

        subfig['layout'].update(height=600, width=1200)
        subfig.update_layout(showlegend=False)

        return subfig



    def timeseries(self, item: str, var:list):
        """
        Generates the timeseries of any variable of any element.
        """
        if item.startswith('NODO'):
            nodal_data_vars_query = f"""
            WITH model_node_coordinates AS 
            (
                SELECT
                    * 
                FROM
                    nodes_coordinates AS nc 
                WHERE
                    nc.model_id = '{self.model}' 
                    AND nc.node_id = '%(node)' 
            )
            ,
            junctions AS 
            (
                SELECT
                    * 
                FROM
                    nodes_junctions nj 
                WHERE
                    nj.model_id = '{self.model}' 
                    AND nj.junction_id = '{item}' 
            )
            ,
            storages AS 
            (
                SELECT
                    * 
                FROM
                    nodes_storage ns 
                WHERE
                    ns.model_id = '{self.model}' 
                    AND ns.storage_id = '{item}' 
            )
            ,
            outfalls AS 
            (
                SELECT
                    * 
                FROM
                    nodes_outfalls AS no2 
                WHERE
                    no2.model_id = '{self.model}' 
                    AND no2.outfall_id = '{item}' 
            )
            ,
            nodes AS 
            (
                SELECT
                    mnc.node_id,
                    mnc.lat,
                    mnc.lon,
                    j.elevation,
                    j.init_depth,
                    j.max_depth 
                FROM
                    model_node_coordinates mnc 
                    JOIN
                        junctions j 
                        ON mnc.node_id = j.junction_id 
                WHERE
                    elevation NOTNULL 
                UNION ALL
                SELECT
                    mnc.node_id,
                    mnc.lat,
                    mnc.lon,
                    s.elevation,
                    s.init_depth,
                    s.max_depth 
                FROM
                    model_node_coordinates mnc 
                    JOIN
                        storages s 
                        ON mnc.node_id = s.storage_id 
                WHERE
                    elevation NOTNULL 
                UNION ALL
                SELECT
                    mnc.node_id,
                    mnc.lat,
                    mnc.lon,
                    o.elevation,
                    0 AS init_depth,
                    0 AS max_depth 
                FROM
                    model_node_coordinates mnc 
                    JOIN
                        outfalls o 
                        ON mnc.node_id = o.outfall_id 
                WHERE
                    elevation NOTNULL 
            )
            ,
            subcatch AS 
            (
                SELECT
                    * 
                FROM
                    subcatchments s 
                WHERE
                    s.model_id = '{self.model}' 
                    AND s.outlet = '{item}' 
            )
            ,
            event_nodes AS 
            (
                SELECT
                    * 
                FROM
                    events_nodes en 
                WHERE
                    event_id = '{self.event}' 
                    AND en.node_id = '{item}' 
            )
            ,
            event_subc AS 
            (
                SELECT
                    * 
                FROM
                    events_subcatchments es 
                WHERE
                    es.event_id = '{self.event}' 
            )
            ,
            event_subc_outlet AS 
            (
                SELECT
                    event_subc.*,
                    subcatch.outlet,
                    subcatch.raingage_id 
                FROM
                    subcatch 
                    LEFT JOIN
                        event_subc 
                        ON subcatch.subcatchment_id = event_subc.subcatchment_id 
            )
            ,
            nodal_out_data AS
            (
                SELECT
                    en.node_id,
                    COALESCE (subcatchment_id, 'SIN CUENCA DE APORTE') AS subcatchment_id,
                    en.elapsed_time,
                    en.depth_above_invert,
                    en.flow_lost_flooding,
                    en.hydraulic_head,
                    en.lateral_inflow,
                    en.total_inflow,
                    en.volume_stored_ponded,
                    COALESCE (eso.evaporation_loss, 0) AS evaporation_loss,
                    COALESCE (eso.runoff_rate, 0) AS runoff_rate,
                    COALESCE (eso.infiltration_loss, 0) AS infiltration_loss,
                    COALESCE (eso.rainfall, 0) AS rainfall 
                FROM
                    event_nodes AS en 
                    LEFT JOIN
                        event_subc_outlet AS eso 
                        ON eso.elapsed_time = en.elapsed_time 
                        AND eso.outlet = en.node_id 
            )
            ,
            nodal_inp_data AS 
            (
                SELECT
                    nodes.*,
                    COALESCE (s.area, 0) AS area,
                    COALESCE (s.imperv, 0) AS imperv,
                    COALESCE (s.slope, 0) AS slope,
                    COALESCE (s.width, 0) AS width,
                    COALESCE (s.curb_len, 0) AS curb_len,
                    COALESCE (s.raingage_id, '') AS raingage_id 
                FROM
                    nodes 
                    LEFT JOIN
                        subcatch s 
                        ON s.outlet = nodes.node_id 
            )
            ,
            nodal_data AS
            (
                SELECT
                    nod.*,
                    nid.elevation,
                    nid.init_depth,
                    nid.max_depth,
                    nid.area,
                    nid.imperv,
                    nid.slope,
                    nid.width,
                    nid.curb_len,
                    nid.raingage_id 
                FROM
                    nodal_out_data AS nod 
                    LEFT JOIN
                        nodal_inp_data AS nid 
                        ON nod.node_id = nid.node_id 
            )
            ,
            rain_mdata AS 
            (
                SELECT
                    * 
                FROM
                    raingages_metadata rm 
                WHERE
                    rm.precipitation_id = '{self.precip}' 
            )
            ,
            rain_tseries AS 
            (
                SELECT
                    * 
                FROM
                    raingages_timeseries rt 
                WHERE
                    rt.precipitation_id = '{self.precip}' 
            )
            ,
            rain AS
            (
                SELECT
                    rt.raingage_id,
                    rt.elapsed_time,
                    COALESCE (rt.VALUE, 0) AS rainfall_acc,
                    rm.format,
                    rm.unit 
                FROM
                    raingages_timeseries rt 
                    LEFT JOIN
                        raingages_metadata AS rm 
                        ON rt.precipitation_id = rm.precipitation_id 
            )
            SELECT DISTINCT
                nd.*,
                r.rainfall_acc,
                r.format,
                r.unit 
            FROM
                nodal_data nd 
                LEFT JOIN
                    rain r 
                    ON nd.raingage_id = r.raingage_id 
                    AND nd.elapsed_time = r.elapsed_time
                ORDER by nd.elapsed_time
            """

            cur5 = self.conn.cursor()
            cur5.execute(nodal_data_vars_query)
            nodal_data_result = cur5.fetchall()

            nodal_data_cols = [
            'node_id',
            'subcatchment_id',
            'elapsed_time',
            'depth_above_invert',
            'flow_lost_flooding',
            'hydraulic_head',
            'lateral_inflow',
            'total_inflow',
            'volume_stored_ponded',
            'evaporation_loss',
            'runoff_rate',
            'infiltration_loss',
            'rainfall',
            'elevation',
            'init_depth',
            'max_depth',
            'area',
            'imperv',
            'slope',
            'width',
            'curb_len',
            'raingage_id',
            'rainfall_acc',
            'format',
            'unit'
            ]

            NodeVars = namedtuple('NodeVars', nodal_data_cols)

            dt_nodes = dt.Frame([i for i in map(NodeVars._make, [i for i in nodal_data_result])], names=nodal_data_cols)

 
            if len(var) == 0:
                df = dt_nodes.to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df

            else:
                # if 'depth_above_invert' in var:
                df = dt_nodes[:, ['node_id','elapsed_time'] + var].to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df['time_to_peak'] = df.iloc[(df['depth_above_invert'].idxmax())]['elapsed_time']
                df['peak'] = df['depth_above_invert'].max()
                df = df.set_index('elapsed_time')
                return df
                # else:
                #     df = dt_nodes[:, ['node_id','elapsed_time'] + var].to_pandas()
                #     df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                #     df = df.set_index('elapsed_time')
                #     return df


        else:
            nodal_linkage_query_link = f"""
            WITH links_event AS 
            (
            SELECT
                * 
            FROM
                events_links el 
            WHERE
                el.event_id = '{self.event}' 
                AND el.link_id = '{item}' 
            )
            ,
            links_conduits_model AS 
            (
            SELECT
                * 
            FROM
                links_conduits AS lc 
            WHERE
                lc.model_id = '{self.model}' 
                AND lc.conduit_id = '{item}' 
            )
            ,
            links_orifices_model AS 
            (
            SELECT
                * 
            FROM
                links_orifices AS lo 
            WHERE
                lo.model_id = '{self.model}' 
                AND lo.orifice_id = '{item}' 
            )
            ,
            links_weirs_model AS 
            (
            SELECT
                * 
            FROM
                links_weirs AS lw 
            WHERE
                lw.model_id = '{self.model}' 
                AND lw.weir_id = '{item}' 
            )
            ,
            links_types_event AS 
            (
            SELECT
                links.link_id,
                from_node,
                to_node,
                elapsed_time,
                flow_rate AS flow_rate,
                flow_depth,
                flow_velocity AS flow_velocity,
                froude_number,
                capacity,
                conduits.length,
                conduits.roughness 
            FROM
                links_event AS links 
                LEFT JOIN
                    links_conduits_model AS conduits 
                    ON conduits.conduit_id = links.link_id 
            WHERE
                from_node NOTNULL 
            UNION
            SELECT
                links.link_id,
                from_node,
                to_node,
                elapsed_time,
                flow_rate AS flow_rate,
                flow_depth,
                flow_velocity AS flow_velocity,
                froude_number,
                capacity,
                0 AS length,
                0 AS roughness 
            FROM
                links_event AS links 
                LEFT JOIN
                    links_orifices_model AS orifices 
                    ON orifices.orifice_id = links.link_id 
            WHERE
                from_node NOTNULL 
            UNION
            SELECT
                links.link_id,
                from_node,
                to_node,
                elapsed_time,
                flow_rate AS flow_rate,
                flow_depth,
                flow_velocity AS flow_velocity,
                froude_number,
                capacity,
                0 AS length,
                0 AS roughness 
            FROM
                links_event AS links 
                LEFT JOIN
                    links_weirs_model AS weirs 
                    ON weirs.weir_id = links.link_id 
            WHERE
                from_node NOTNULL 
            )
            ,
            rain_mdata AS 
            (
            SELECT
                * 
            FROM
                raingages_metadata rm 
            WHERE
                rm.precipitation_id = 'precipitation_{self.event}' 
            )
            ,
            rain_tseries AS 
            (
            SELECT
                * 
            FROM
                raingages_timeseries rt 
            WHERE
                rt.precipitation_id = 'precipitation_{self.event}' 
            )
            ,
            rain AS
            (
            SELECT
                rt.raingage_id,
                rt.elapsed_time,
                rt.VALUE,
                rm.format,
                rm.unit 
            FROM
                raingages_timeseries rt 
                LEFT JOIN
                    raingages_metadata AS rm 
                    ON rt.raingage_id = rm.raingage_id 
                    AND rt.precipitation_id = rm.precipitation_id 
            )
            ,
            subc AS 
            (
            SELECT
                * 
            FROM
                subcatchments s2 
            WHERE
                s2.model_id = '{self.model}' 
                AND s2.outlet = 
                (
                    SELECT DISTINCT
                        from_node 
                    FROM
                        links_types_event
                )
            )
            ,
            event_subc AS 
            (
            SELECT
                * 
            FROM
                events_subcatchments es 
            WHERE
                es.event_id = '{self.event}' 
            )
            ,
            event_subc_outlet AS 
            (
            SELECT DISTINCT
                subc.subcatchment_id,
                subc.outlet,
                subc.raingage_id,
                elapsed_time,
                event_subc.rainfall,
                subc.area 
            FROM
                subc 
                INNER JOIN
                    event_subc 
                    ON subc.subcatchment_id = event_subc.subcatchment_id 
            )
            ,
            event_subc_rainfall AS 
            (
            SELECT DISTINCT
                eso.*,
                rain.VALUE,
                rain.format,
                rain.unit 
            FROM
                event_subc_outlet eso 
                INNER JOIN
                    rain 
                    ON rain.raingage_id = eso.raingage_id 
                    AND rain.elapsed_time = eso.elapsed_time 
            )
            SELECT
            lte.*,
            esr.rainfall	--, coalesce (esr.value, 0) as rainfall_acc, esr.format, esr.unit
            FROM
            links_types_event lte 
            LEFT JOIN
                event_subc_outlet esr 
                ON lte.from_node = esr.outlet 
                AND lte.elapsed_time = esr.elapsed_time 
            ORDER BY
            outlet,
            elapsed_time
            """

            cur4 = self.conn.cursor()
            cur4.execute(nodal_linkage_query_link)
            nodal_linkage_result_link = cur4.fetchall()

            nodal_linkage_cols = [
            'link_id',
            'from_node',
            'to_node',
            'elapsed_time',
            'flow_rate',
            'flow_depth',
            'flow_velocity',
            'froude_number',
            'capacity',
            'length',
            'roughness',
            'rainfall',
            # 'rainfall_acc',
            # 'format',
            # 'unit'
            ]

            LinkVars = namedtuple('LinksVars', nodal_linkage_cols)

            dt_links = dt.Frame([i for i in map(LinkVars._make, [i for i in nodal_linkage_result_link])], names=nodal_linkage_cols)

            if len(var) == 0:
                df = dt_links.to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df

            else:
                df = dt_links[:, ['link_id','elapsed_time'] + var].to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df
    

    
    def timeseries_viz(self, item:str, var:list):

        df_plot_0 = self.timeseries(item, ['rainfall'])
        df_plot_0 = df_plot_0.resample(f'30min').mean()
        plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

        df_plot_1 = self.timeseries(item, var)
        plot_var_max = 1.5 * df_plot_1[var[0]].max()
        plot_var_min = 1.1 * df_plot_1[var[0]].min()
        splitted_var = var[0].split('_')
        plot_var_legend = ' '.join([word.capitalize() for word in splitted_var][:-1]) + f' [{splitted_var[-1]}]'
            


        subfig = make_subplots(specs=[[{"secondary_y": True}]])

        # create two independent figures with px.line each containing data from multiple columns
        fig = px.bar(df_plot_0, y='rainfall')#, render_mode="webgl",)
        fig2 = px.line(df_plot_1, y=var[0])
        fig2.update_traces(line={'width':4, 'color':'#125AEF'})


        fig2.update_traces(yaxis="y2")


        subfig.add_traces(fig.data + fig2.data)# + fig3.data)
        subfig['layout']['yaxis1'].update(title='Precipitation intensity (mm/h)',range=[0, plot_rainfall_max], autorange='reversed')
        subfig['layout']['yaxis2'].update(title= plot_var_legend, range=[plot_var_min, plot_var_max], autorange=False)
        subfig.for_each_trace(lambda t: t.update(marker=dict(color=['black'])))
        subfig['layout']['xaxis'].update(title='', tickformat='%d-%b %Hh')
        subfig['layout'].update(plot_bgcolor='white', font={'size':25})#, template='plotly_white')
        subfig.update_xaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        subfig.update_yaxes(showline=True, linewidth=3, linecolor='black', mirror=True)
        subfig.update_xaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)
        subfig.update_yaxes(ticks="inside", tickwidth=2, tickcolor='black', ticklen=10)

        subfig['layout'].update(height=600, width=1200)
        subfig.update_layout(showlegend=False)

        return subfig