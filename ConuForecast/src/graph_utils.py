#%%
# -*- coding: utf-8 -*-
from os import remove
import pandas as pd
import numpy as np
import networkx as nx
import psycopg2
import datatable as dt
import plotly.express as px

from plotly.subplots import make_subplots
from collections import namedtuple
from datetime import datetime

class DBconnector():
    """
    Connection to the database
    """
    #host="172.18.0.1", port = 5555, database="base-ina", user="postgres", password="postgres"


    def __init__(self, url: str, port: int, database: str, user: str, password: str) -> None:
        self.pg_conn = psycopg2.connect(host=url, port=port, database=database, user=user, password=password)

conn = DBconnector('172.18.0.1', 5555, 'base-ina', 'postgres', 'postgres')


class GraphInit():
    """
    Initializes the graph of the whole model by doing the correponding queries to the database.
    The DB must be turned on.
    """
    def get_time_range(self):
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
        
        time_range = self.get_time_range()[0]
        first_two = time_range[:2]
        time_step = (first_two[1][0] - first_two[0][0]).seconds // 60

        return time_step, f'{time_step} minutes'
    

    def nodal_linkage_query(self, elapsed_time):
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
        , rain_mdata AS
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
                AND rt.elapsed_time = '{elapsed_time}'
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
                JOIN
                    raingages_metadata AS rm
                    ON rt.precipitation_id = rm.precipitation_id
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
        SELECT DISTINCT
            lte.*,
            COALESCE (esr.rainfall, 0) AS rainfall,
            COALESCE (esr.VALUE, 0) AS rainfall_acc
        FROM
            links_types_event lte
            LEFT JOIN
                event_subc_rainfall esr
                ON lte.from_node = esr.outlet
                AND lte.elapsed_time = esr.elapsed_time
        """

        cur1 = self.conn.cursor()
        cur1.execute(nodal_linkage_query)
        nodal_linkage_query_result = cur1.fetchall()

        self.nodal_linkage_query_results[f'{self.event}_{elapsed_time}'] = nodal_linkage_query_result



    def get_nodal_linkage(self, elapsed_time):

        try:
            self.nodal_linkage_dict[f'{self.event}_{elapsed_time}']
            pass
        
        except:
            self.nodal_linkage_query(elapsed_time)
                
            nodal_linkage = {i[0]:
                        {
                            'link_id':i[1],
                            'from_node':i[2],
                            'to_node':i[3],
                            'elapsed_time': i[4],
                            'flow_rate':i[5],
                            'flow_depth':i[6],
                            'flow_velocity':i[7],
                            'froud_number':i[8],
                            'capacity':i[9],
                            'length':i[10],
                            'roughness': i[11],
                            'rainfall':i[12],
                            'rainfall_acc':i[13]
                        } for i in self.nodal_linkage_query_results[f'{self.event}_{elapsed_time}']}

            self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'] = nodal_linkage


    def nodal_data_query(self, elapsed_time):
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
                JOIN
                    raingages_metadata AS rm 
                    ON rt.precipitation_id = rm.precipitation_id 
        )
        SELECT DISTINCT
            nd.*,
            COALESCE (r.rainfall_acc, 0) AS rainfall_acc,
            COALESCE (r.format, '') AS format,
            COALESCE (r.unit, '') AS unit 
        FROM
            nodal_data nd 
            LEFT JOIN
                rain r 
                ON nd.raingage_id = r.raingage_id 
                AND nd.elapsed_time = r.elapsed_time
        """

        cur2 = self.conn.cursor()
        cur2.execute(nodal_data_query)
        nodal_data_query_result = cur2.fetchall()

        self.nodal_data_query_results[f'{self.event}_{elapsed_time}'] = nodal_data_query_result



    def get_nodal_data(self, elapsed_time):

        try:
            self.nodal_data_dict[f'{self.event}_{elapsed_time}']
            pass

        except:
            self.nodal_data_query(elapsed_time)
            
            nodal_data = {i[0]: {
            'subcatchment_id':i[1],
            'elapsed_time':i[2],
            'depth_above_invert':i[3],
            'flow_lost_flooding':i[4],
            'hydraulic_head':i[5],
            'lateral_inflow':i[6],
            'total_inflow':i[7],
            'volume_stored_ponded':i[8],
            'rainfall': i[9],
            'evaporation_loss':i[10],
            'runoff_rate':i[11],
            'infiltration_loss':i[12],
            'lon':i[13],
            'lat':i[14],
            'elevation':i[15],
            'pos': (i[14], i[13]),
            'init_depth':i[16],
            'max_depth':i[17],
            'area':i[18],
            'imperv':i[19],
            'slope':i[20],
            'width':i[21],
            'curb_len':i[22],
            'raingage_id':i[23],
            'rainfall_acc':i[24],
            'format':i[25],
            'unit':i[26],
            } for i in self.nodal_data_query_results[f'{self.event}_{elapsed_time}']}

            self.nodal_data_dict[f'{self.event}_{elapsed_time}'] = nodal_data

    # graph creation

    def build_graph(self, elapsed_time):

        self.get_nodal_data(elapsed_time)
        self.get_nodal_linkage(elapsed_time)

        try:
            self.digraphs[f'{self.event}_{elapsed_time}']
            pass

        except:
            DG = nx.DiGraph(elapsed_time = elapsed_time[1:-1], model= self.model)
            [DG.add_edge(i[1]['from_node'], i[1]['to_node'], **i[1]) for i in self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'].items()]
            [DG.add_node(i[0], **i[1]) for i in self.nodal_data_dict[f'{self.event}_{elapsed_time}'].items()]

            #target definition
            [DG.add_node(i[0], **{'target': i[1]['depth_above_invert'] > 0.2}) for i in self.nodal_data_dict[f'{self.event}_{elapsed_time}'].items()]

            self.digraphs[f'{self.event}_{elapsed_time}'] = DG



    def build_coordinates_dict(self, elevation:bool = False):
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
            self.pos_dict = coordinates
        else:
            coordinates = {i[0]: {'lat':i[1], 'lon':i[2]} for i in coordinates_query_result}
            self.pos_dict = coordinates


    def remove_attrs(self, attrs_dict:dict, elapsed_time):
        """
        attrs_dcit = {'nodes': [...], ''edges': [...]}
        """
        pruned_digraph = self.digraphs[f'{self.event}_{elapsed_time}'].copy()

        for node in pruned_digraph.nodes():
            for n_attr in attrs_dict['nodes']:
                del pruned_digraph.nodes()[node][n_attr]

        for edge in pruned_digraph.edges():
            for e_attr in attrs_dict['edges']:
                del pruned_digraph.edges()[edge][e_attr]
            
        self.pruned_digraph = pruned_digraph

    
    def select_attrs(self, attrs_dict:dict, elapsed_time):
        """
        attrs_dcit = {'nodes': [...], ''edges': [...]}
        """
        pruned_digraph = self.digraphs[f'{self.event}_{elapsed_time}'].copy()
        pruned_digraph = nx.convert_node_labels_to_integers(pruned_digraph)
        
        node_attrs = list(pruned_digraph.nodes(data=True)[0].keys())
        attrs_dict['nodes'] = [x for x in node_attrs if x not in attrs_dict['nodes']]
        
        edge_attrs = list(list(pruned_digraph.edges(data=True))[0][2].keys())
        attrs_dict['edges'] = [x for x in edge_attrs if x not in attrs_dict['edges']]


        return self.remove_attrs(attrs_dict) 

 

    def __init__(self, model:str, event:str, precip:str, conn) -> None:
        self.conn = conn.pg_conn
        # self.elapsed_time = elapsed_time
        self.model = model
        self.event = event
        self.precip = precip
        self.build_coordinates_dict()
        self.time_range = self.get_time_range()
        self.nodal_linkage_query_results = {}
        self.nodal_linkage_dict = {}
        self.nodal_data_query_results = {}
        self.nodal_data_dict = {}
        self.digraphs = {}
        self.pos_dict = {}
        self.subgraphs = {}

        


    def build_subgraph(self, node:str, acc_data:bool, elapsed_time):
        try:
            self.digraphs[f'{self.event}_{elapsed_time}']
            pass
        except:
            self.build_graph(elapsed_time)

        try:
            self.subgraphs[self.event + '_' + node +  '_' + elapsed_time]
            pass
        
        except:
            preds_list = [(i[0],i[1]) for i in nx.edge_dfs(self.digraphs[f'{self.event}_{elapsed_time}'], node, 'reverse')]

            graph_preds = nx.DiGraph(elapsed_time = elapsed_time, model= self.model, outlet_node = node)

            # own node data, for th cases without preds
            graph_preds.add_node(node, **self.nodal_data_dict[f'{self.event}_{elapsed_time}'][node])

            
            [graph_preds.add_edge(edge[0], edge[1], **self.nodal_linkage_dict[f'{self.event}_{elapsed_time}'][edge[0] + '->' + edge[1]]) for edge in preds_list]
            [graph_preds.add_node(i, **self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]

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

            [graph_preds.add_node(
                i, **{'target': risk_classes(self.nodal_data_dict[f'{self.event}_{elapsed_time}'][i]['depth_above_invert'])}
                ) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]

            if acc_data:
                vars_acc = {
                'area_aporte_ha': round(sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                # 'perm_media_%':
                # round(sum(
                # [graph_preds.nodes()[i]['area'] * graph_preds.nodes()[i]['imperv']
                # for i in graph_preds.nodes()]
                # )
                # / sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),4),

                # 'manning_medio_flow_s/m^1/3':
                # round(sum(
                # [
                # graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                # * graph_preds.edges()[edge[0], edge[1]]['length']
                # * graph_preds.edges()[edge[0], edge[1]]['roughness']
                # for edge in graph_preds.edges()
                # ])
                # / sum([graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                # * graph_preds.edges()[edge[0], edge[1]]['length']
                # for edge in graph_preds.edges()
                # ]),3),

                # 'manning_medio_s/m^1/3':
                # round(sum(
                # [
                # graph_preds.edges()[edge[0], edge[1]]['length']
                # * graph_preds.edges()[edge[0], edge[1]]['roughness']
                # for edge in graph_preds.edges()
                # ])
                # / sum([graph_preds.edges()[edge[0], edge[1]]['length']
                # for edge in graph_preds.edges()
                # ]),3),

                # 'precip_media_mm/ha': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                # / sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                'infilt_media_mm/hs': round(np.average([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes()]),2),

                # 'vol_almacenado_mm': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
                #     - sum([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes])
                #     - sum([graph_preds.nodes()[i]['evaporation_loss'] for i in graph_preds.nodes])
                #     - sum([graph_preds.nodes()[i]['runoff_rate'] * graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

                # 'vol_precipitado_mm_acc': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges()]),2),

                #     'vol_precipitado_mm': round(sum([graph_preds.edges()[edge[0], edge[1]]['rainfall'] for edge in graph_preds.edges()]),2),


                # 'delta_h_medio_m/m':
                # round(
                # (
                # max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                # - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                # ) / np.sqrt(10000 * sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]))
                # ,2),

                # 'pendiente_media_m/m':
                # (
                # max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                # - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
                # ) / sum([graph_preds.edges()[edge[0], edge[1]]['length'] for edge in graph_preds.edges()])
                }


                graph_preds.add_node(node, **vars_acc)

                self.subgraphs[self.event + '_' + node +  '_' + elapsed_time + '_acc'] = graph_preds
            
            self.subgraphs[self.event + '_' + node +  '_' + elapsed_time] = graph_preds



    def subgraphs_timeseries(self, node:str, var:str, time_step:int = 4, acc_data=True):
        [self.build_subgraph(
            node, acc_data=acc_data, elapsed_time=time
            ) for time in (sorted(self.time_range[1])[::time_step])
        ]
        
        if acc_data:
            df = pd.DataFrame(
                [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S'), self.subgraphs[f'{self.event}_{node}_{time}_acc'].nodes()[node][var]
                ) for time in (sorted(self.time_range[1])[::time_step])])\
                    .rename({0:'elapsed_time', 1: var}, axis=1).set_index('elapsed_time')
            return df        
        df = pd.DataFrame(
            [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S'), self.subgraphs[f'{self.event}_{node}_{time}'].nodes()[node][var]
            ) for time in (sorted(self.time_range[1])[::time_step])])\
                    .rename({0:'elapsed_time', 1: var}, axis=1).set_index('elapsed_time')

        return df



    def subgraph_tseries_viz(self, node:str, var:str, time_step:int, acc_data:bool):
        rainfall_step = max(self.which_time_step()[0], int((self.which_time_step()[0]) * time_step))

        df_plot_0 = self.subgraphs_timeseries(node, 'rainfall', time_step=time_step)
        df_plot_0 = df_plot_0.resample(f'{rainfall_step}min').mean()
        plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

        df_plot_1 = self.subgraphs_timeseries(node, var,time_step=time_step, acc_data=acc_data )
        plot_var_max = 1.5 * df_plot_1[var].max()
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
        subfig['layout']['yaxis2'].update(title= plot_var_legend, range=[0, plot_var_max], autorange=False)
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
                df = dt_nodes[:, ['node_id','elapsed_time'] + var].to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df

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
                    elapsed_time, flow_rate , flow_depth, flow_velocity, froude_number, capacity, conduits.length, conduits.roughness 
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
                    elapsed_time, flow_rate , flow_depth, flow_velocity, froude_number, capacity, 0 AS length, 0 AS roughness 
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
                    elapsed_time, flow_rate , flow_depth, flow_velocity, froude_number, capacity, 0 AS length, 0 AS roughness 
                FROM
                    links_event AS links 
                    LEFT JOIN
                        links_weirs_model AS weirs 
                        ON weirs.weir_id = links.link_id 
                WHERE
                    from_node NOTNULL 
            )
            , rain_mdata AS 
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
                    rt.VALUE,
                    rm.format,
                    rm.unit 
                FROM
                    raingages_timeseries rt 
                    JOIN
                        raingages_metadata AS rm 
                        ON rt.precipitation_id = rm.precipitation_id 
            )
            ,
            subc AS 
            (
                SELECT
                    * 
                FROM
                    subcatchments s2 
                WHERE
                    s2.model_id = '%(model)s' 
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
                    es.event_id = '%(event)s' 
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
                COALESCE (esr.rainfall, 0) AS rainfall,
                COALESCE (esr.VALUE, 0) AS rainfall_acc,
                esr.format,
                esr.unit 
            FROM
                links_types_event lte 
                LEFT JOIN
                    event_subc_rainfall esr 
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
            'rainfall_acc',
            'format',
            'unit'
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
        plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

        df_plot_1 = self.timeseries(item, var)
        plot_var_max = 1.5 * df_plot_1[var[0]].max()
        splitted_var = var[0].split('_')
        plot_var_legend = ' '.join([word.capitalize() for word in splitted_var][:-1]) + f' [{splitted_var[-1]}]'
            


        subfig = make_subplots(specs=[[{"secondary_y": True}]])

        # create two independent figures with px.line each containing data from multiple columns
        fig = px.bar(df_plot_0, y='rainfall')#, render_mode="webgl",)
        fig2 = px.line(df_plot_1, y=var[0])
        fig2.update_traces(line={'width':5, 'color':'#125AEF'})


        fig2.update_traces(yaxis="y2")


        subfig.add_traces(fig.data + fig2.data)# + fig3.data)
        subfig['layout']['yaxis1'].update(title='Precipitation intensity (mm/h)',range=[0, plot_rainfall_max], autorange='reversed')
        subfig['layout']['yaxis2'].update(title= plot_var_legend, range=[0, plot_var_max], autorange=False)
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

        
#%%
# df_nodes_indiv = (pd.DataFrame(pos)).T.reset_index().rename({'index':'nodo', 'lat':'lat', 'lon':'lon'}, axis=1)
