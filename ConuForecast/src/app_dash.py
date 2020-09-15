# %%
# -*- coding: utf-8 -*-
from _plotly_utils.colors import color_parser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.graph_objs as go
import numpy as np
import networkx as nx
import pandas as pd
import networkx as nx
from collections import namedtuple
import datatable as dt
import psycopg2


pg_conn = psycopg2.connect(host="172.18.0.1", port=5555, database="base-ina", user="postgres", password="postgres")

# Create a cursor object
cur = pg_conn.cursor()

# random elapsed_elpased
# double quotes are necessary for the insert on the SQL querys

elapsed_time = '2019-10-11 12:10:00'
model = 'model_sjj-070'
event = 'sjj-070'
precip = 'precipitation_sjj-070'

# elapsed_time = '2017-01-01 12:00:00'
# model = 'model_007'
# event = '007'
# precip = 'precipitation_007'


# query with links and nodes information
nodal_linkage_query = """
with links_event as (
select *
from events_links el
where el.event_id = '%(event)s'
	and el.elapsed_time = '%(elapsed_time)s'
)
, links_conduits_model as (
select *
from links_conduits as lc
where lc.model_id = '%(model)s'
)
, links_orifices_model as (
select *
from links_orifices as lo
where lo.model_id = '%(model)s'
)
, links_weirs_model as (
select *
from links_weirs as lw
where lw.model_id = '%(model)s'
)
, links_types_event as (
	select
		(case
			when links.flow_rate >= 0 then concat(conduits.from_node, '->', conduits.to_node)
			else concat(conduits.to_node, '->', conduits.from_node)
		end) as edge,
		links.link_id,
		(case
			when links.flow_rate >= 0 then conduits.from_node
			else conduits.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then conduits.from_node
			else conduits.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		conduits.length,
		conduits.roughness
	from links_event as links
	left join links_conduits_model AS conduits
	    ON conduits.conduit_id = links.link_id
	where from_node notnull
	union
	select
		(case
			when links.flow_rate >= 0 then concat(orifices.from_node, '->', orifices.to_node)
			else concat(orifices.to_node, '->', orifices.from_node)
		end) as edge,
		links.link_id,
		(case
			when links.flow_rate >= 0 then orifices.from_node
			else orifices.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then orifices.from_node
			else orifices.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		0 as length,
		0 as roughness
	from links_event as links
	left join links_orifices_model AS orifices
	    ON orifices.orifice_id = links.link_id
	where from_node notnull
	union
	select
		(case
			when links.flow_rate >= 0 then concat(weirs.from_node, '->', weirs.to_node)
			else concat(weirs.to_node, '->', weirs.from_node)
		end) as edge,
		links.link_id,
		(case
			when links.flow_rate >= 0 then weirs.from_node
			else weirs.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then weirs.from_node
			else weirs.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		0 as length,
		0 as roughness
	from links_event as links
	left join links_weirs_model AS weirs
	    ON weirs.weir_id = links.link_id
	where from_node notnull
)
,rain_mdata as (
select *
from raingages_metadata rm
where rm.precipitation_id = '%(precip)s'
)
, rain_tseries as (
select *
from raingages_timeseries rt
where rt.precipitation_id = '%(precip)s'
	and rt.elapsed_time = '%(elapsed_time)s'
)
, rain as(
select  rt.raingage_id, rt.elapsed_time, rt.value, rm.format, rm.unit
from raingages_timeseries rt
join raingages_metadata as rm
	on rt.precipitation_id = rm.precipitation_id
), subc as (
select  *
from subcatchments s2
where s2.model_id = '%(model)s'
), event_subc as (
select  *
from events_subcatchments es
where es.event_id = '%(event)s'
	and es.elapsed_time = '%(elapsed_time)s'
), event_subc_outlet as (
select distinct subc.subcatchment_id, subc.outlet, subc.raingage_id, elapsed_time , event_subc.rainfall
from subc
inner join event_subc
	on subc.subcatchment_id = event_subc.subcatchment_id
)
, event_subc_rainfall as (
select distinct eso.*, rain.value, rain.format, rain.unit
from event_subc_outlet eso
inner join rain
	on rain.raingage_id = eso.raingage_id
	and rain.elapsed_time = eso.elapsed_time
)
select distinct lte.*, coalesce (esr.rainfall, 0) as rainfall,  coalesce (esr.value, 0) as rainfall_acc
from links_types_event lte
left join event_subc_rainfall esr
	on lte.from_node = esr.outlet
	and lte.elapsed_time = esr.elapsed_time
    """% {'elapsed_time': elapsed_time, 'event': event, 'model': model, 'precip':precip}

cur.execute(nodal_linkage_query)
nodal_linkage_result = cur.fetchall()

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
                 } for i in nodal_linkage_result}
#%%

nodal_data_query = """
with
model_node_coordinates as (
select *
from nodes_coordinates as nc
where nc.model_id  = '%(model)s'
)
, junctions as (
select *
from nodes_junctions nj
where nj.model_id = '%(model)s'
)
, storages as (
select *
from nodes_storage ns
where ns.model_id= '%(model)s'
)
, outfalls as (
select *
from nodes_outfalls as no2
where no2.model_id = '%(model)s'
)
, nodes as (
    select mnc.node_id, mnc.lat, mnc.lon, j.elevation, j.init_depth, j.max_depth
    from model_node_coordinates mnc
    join junctions j
    	on mnc.node_id = j.junction_id
    where elevation notnull
    union all
    select mnc.node_id, mnc.lat, mnc.lon, s.elevation, s.init_depth, s.max_depth
    from model_node_coordinates mnc
    join storages s
    	on mnc.node_id = s.storage_id
    where elevation notnull
    union all
    select mnc.node_id, mnc.lat, mnc.lon, o.elevation, 0 as init_depth, 0 as max_depth
    from model_node_coordinates mnc
    join outfalls o
    	on mnc.node_id = o.outfall_id
    where elevation notnull
)
, subcatch as (
	select *
	from subcatchments s
	where s.model_id  = '%(model)s'
)
, event_nodes as (
select *
from events_nodes en
where event_id = '%(event)s'
	and en.elapsed_time = '%(elapsed_time)s'
)
, event_subc as (
select  *
from events_subcatchments es
where es.event_id = '%(event)s'
	and es.elapsed_time = '%(elapsed_time)s'
)
, event_subc_outlet as (
select  event_subc.*, subcatch.outlet, subcatch.raingage_id
from subcatch
left join event_subc
	on subcatch.subcatchment_id = event_subc.subcatchment_id
)
, nodal_out_data as(
select
	en.node_id,
	coalesce (subcatchment_id, 'SIN CUENCA DE APORTE') as subcatchment_id,
	en.elapsed_time,
	en.depth_above_invert,
	en.flow_lost_flooding,
	en.hydraulic_head,
	en.lateral_inflow,
	en.total_inflow,
	en.volume_stored_ponded,
    coalesce (eso.rainfall, 0) as rainfall,
	coalesce (eso.evaporation_loss,0) as evaporation_loss,
	coalesce (eso.runoff_rate, 0) as runoff_rate,
	coalesce (eso.infiltration_loss, 0) as infiltration_loss
from event_nodes as en
left join event_subc_outlet as eso
	on eso.elapsed_time = en.elapsed_time
	and eso.outlet = en.node_id
)
, nodal_inp_data as (
select
	nodes.*,
	coalesce (s.area,0) as area,
	coalesce (s.imperv,0) as imperv,
	coalesce (s.slope, 0) as slope,
	coalesce (s.width,0) as width,
	coalesce (s.curb_len, 0) as curb_len,
	coalesce (s.raingage_id, '') as raingage_id
from nodes
left jOIN subcatch s
	on s.outlet = nodes.node_id
)
, nodal_data as (
select
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
from nodal_out_data as nod
left join nodal_inp_data as nid
	on nod.node_id = nid.node_id
),
rain_mdata as (
select *
from raingages_metadata rm
where rm.precipitation_id = '%(precip)s'
)
, rain_tseries as (
select *
from raingages_timeseries rt
where rt.precipitation_id = '%(precip)s'
)
, rain as(
select  rt.raingage_id, rt.elapsed_time, coalesce (rt.value, 0) as rainfall_acc, rm.format, rm.unit
from raingages_timeseries rt
join raingages_metadata as rm
	on rt.precipitation_id = rm.precipitation_id
)
select distinct nd.*, coalesce (r.rainfall_acc, 0) as rainfall_acc, coalesce (r.format, '') as format, coalesce (r.unit, '') as unit
from nodal_data nd
left join rain r
	on nd.raingage_id = r.raingage_id
	and nd.elapsed_time = r.elapsed_time
    """% {'elapsed_time': elapsed_time, 'event': event, 'model': model, 'precip':precip}

cur.execute(nodal_data_query)
nodal_data_results = cur.fetchall()


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
} for i in nodal_data_results}



def build_coordinates_dict():
    nodes_coordinates_query = """
with node_coordinates_model as(
select *
from nodes_coordinates as nc
where nc.model_id  = '%(model)s'
)
select nc.node_id, nc.lat, nc.lon, nj.elevation, nj.init_depth, nj.max_depth
from node_coordinates_model nc
join nodes_junctions nj
    on nc.node_id = nj.junction_id
where nj.model_id = '%(model)s'
union all
select nc.node_id, nc.lat, nc.lon, ns.elevation, ns.init_depth, ns.max_depth
from node_coordinates_model nc
join nodes_storage ns
    on nc.node_id = ns.storage_id
where ns.model_id= '%(model)s'
union all
select nc.node_id, nc.lat, nc.lon, no2.elevation, 0 as init_depth, 0 as max_depth
from node_coordinates_model nc
join nodes_outfalls no2
    on nc.node_id = no2.outfall_id
where no2.model_id = '%(model)s'
    """% {'model': model}

    cur.execute(nodes_coordinates_query)
    coordinates = cur.fetchall()
    return {i[0]: {'lat':i[1], 'lon':i[2], 'elevation':i[3]} for i in coordinates}

# graph creation
def graph_init(time, df, nodal_data):
    global DG

    DG = nx.DiGraph(elapsed_time = time[1:-1], model= model)

    [DG.add_edge(i[1]['from_node'], i[1]['to_node'], **i[1]) for i in nodal_linkage.items()]


    [DG.add_node(i[0], **i[1]) for i in nodal_data.items()]



    return DG

# init
DG = graph_init(elapsed_time, nodal_linkage, nodal_data)
pos = build_coordinates_dict()

def find_predecessors_tree(node):
    return [(i[0],i[1]) for i in nx.edge_dfs(DG, node, 'reverse')]

def build_tree_subgraph(node):
    preds_list = find_predecessors_tree(node)

    graph_preds = nx.DiGraph(elapsed_time = elapsed_time[1:-1], model= model, node=node)

    [graph_preds.add_edge(edge[0], edge[1], **nodal_linkage[edge[0] + '->' + edge[1]]) for edge in preds_list]
    [graph_preds.add_node(i, **nodal_data[i]) for i in set([i[0] for i in preds_list] + [i[1] for i in preds_list])]


    vars_acc = {
    'area_aporte_ha': round(sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

    'perm_media_%':
        round(sum(
            [graph_preds.nodes()[i]['area'] * graph_preds.nodes()[i]['imperv']
             for i in graph_preds.nodes()]
        )
        / sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),4),

    'manning_medio_flow_s/m^1/3':
        round(sum(
            [
                graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                * graph_preds.edges()[edge[0], edge[1]]['length']
                * graph_preds.edges()[edge[0], edge[1]]['roughness']
                for edge in graph_preds.edges()
            ])
        / sum([graph_preds.edges()[edge[0], edge[1]]['flow_rate']
                * graph_preds.edges()[edge[0], edge[1]]['length']
                for edge in graph_preds.edges()
            ]),3)
        ,

    'manning_medio_s/m^1/3':
        round(sum(
            [
                graph_preds.edges()[edge[0], edge[1]]['length']
                * graph_preds.edges()[edge[0], edge[1]]['roughness']
                for edge in graph_preds.edges()
            ])
        / sum([graph_preds.edges()[edge[0], edge[1]]['length']
                for edge in graph_preds.edges()
            ]),3)
      ,

    'precip_media_mm/ha': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
        / sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

    'infilt_media_mm/hs': round(np.average([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes()]),2),

#     'vol_almacenado': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges])
#         - sum([graph_preds.nodes()[i]['infiltration_loss'] for i in graph_preds.nodes])
#         - sum([graph_preds.nodes()[i]['evaporation_loss'] for i in graph_preds.nodes])
#         - sum([graph_preds.nodes()[i]['runoff_rate'] * graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]),2),

    'vol_precipitado_mm_acc': round(max([graph_preds.edges()[edge[0], edge[1]]['rainfall_acc'] for edge in graph_preds.edges()]),2),

#     'vol_precipitado_mm': round(sum([graph_preds.edges()[edge[0], edge[1]]['rainfall'] for edge in graph_preds.edges()]),2),


    'delta_h_medio_m/m':
        round(
            (
            max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
            - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
            ) / np.sqrt(10000 * sum([graph_preds.nodes()[i]['area'] for i in graph_preds.nodes()]))
        ,2),

    'pendiente_media_m/m':
            (
            max([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
            - min([graph_preds.nodes()[i]['elevation'] for i in graph_preds.nodes()])
            ) / sum([graph_preds.edges()[edge[0], edge[1]]['length'] for edge in graph_preds.edges()])
    }


    graph_preds.add_node(node, **vars_acc)

    return graph_preds

def link_vars_to_df(link_id, var):
    nodal_linkage_query_link = """
with links_event as (
select *
from events_links el
where el.event_id = '%(event)s'
	and el.link_id = '%(link_id)s'
)
, links_conduits_model as (
select *
from links_conduits as lc
where lc.model_id = '%(model)s'
	and lc.conduit_id = '%(link_id)s'
)
, links_orifices_model as (
select *
from links_orifices as lo
where lo.model_id = '%(model)s'
	and lo.orifice_id = '%(link_id)s'
)
, links_weirs_model as (
select *
from links_weirs as lw
where lw.model_id = '%(model)s'
	and lw.weir_id = '%(link_id)s'
)
, links_types_event as (
	select
		links.link_id,
		(case
			when links.flow_rate >= 0 then conduits.from_node
			else conduits.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then conduits.from_node
			else conduits.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		conduits.length,
		conduits.roughness
	from links_event as links
	left join links_conduits_model AS conduits
	    ON conduits.conduit_id = links.link_id
	where from_node notnull
	union
	select
		links.link_id,
		(case
			when links.flow_rate >= 0 then orifices.from_node
			else orifices.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then orifices.from_node
			else orifices.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		0 as length,
		0 as roughness
	from links_event as links
	left join links_orifices_model AS orifices
	    ON orifices.orifice_id = links.link_id
	where from_node notnull
	union
	select
		links.link_id,
		(case
			when links.flow_rate >= 0 then weirs.from_node
			else weirs.to_node
		end) as from_node,
		(case
			when links.flow_rate < 0 then weirs.from_node
			else weirs.to_node
		end) as to_node,
		elapsed_time,
		abs(flow_rate) as flow_rate ,
		flow_depth,
		abs(flow_velocity ) as flow_velocity,
		froude_number,
		capacity,
		0 as length,
		0 as roughness
	from links_event as links
	left join links_weirs_model AS weirs
	    ON weirs.weir_id = links.link_id
	where from_node notnull
),
rain_mdata as (
select *
from raingages_metadata rm
where rm.precipitation_id = '%(precip)s'
)
, rain_tseries as (
select *
from raingages_timeseries rt
where rt.precipitation_id = '%(precip)s'
)
, rain as(
select  rt.raingage_id, rt.elapsed_time, rt.value, rm.format, rm.unit
from raingages_timeseries rt
join raingages_metadata as rm
	on rt.precipitation_id = rm.precipitation_id
), subc as (
select  *
from subcatchments s2
where s2.model_id = '%(model)s'
	and s2.outlet = (select distinct from_node from links_types_event)
), event_subc as (
select  *
from events_subcatchments es
where es.event_id = '%(event)s'
), event_subc_outlet as (
select distinct subc.subcatchment_id, subc.outlet, subc.raingage_id, elapsed_time , event_subc.rainfall, subc.area
from subc
inner join event_subc
	on subc.subcatchment_id = event_subc.subcatchment_id
)
, event_subc_rainfall as (
select distinct eso.*, rain.value, rain.format, rain.unit
from event_subc_outlet eso
inner join rain
	on rain.raingage_id = eso.raingage_id
	and rain.elapsed_time = eso.elapsed_time
)
select lte.*, coalesce (esr.rainfall, 0) as rainfall, coalesce (esr.value, 0) as rainfall_acc, esr.format, esr.unit
from links_types_event lte
left join event_subc_rainfall esr
	on lte.from_node = esr.outlet
	and lte.elapsed_time = esr.elapsed_time
order by outlet, elapsed_time
"""% {'event': event, 'model': model, 'link_id':link_id, 'precip':precip}

    cur3 = pg_conn.cursor()
    cur3.execute(nodal_linkage_query_link)
    nodal_linkage_result_link = cur3.fetchall()
    cur3.close()



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

def node_vars_to_df(node_id, var):
    nodal_data_vars_query = """
with
model_node_coordinates as (
select *
from nodes_coordinates as nc
where nc.model_id  = '%(model)s'
	and nc.node_id = '%(node)s'
)
, junctions as (
select *
from nodes_junctions nj
where nj.model_id = '%(model)s'
	and nj.junction_id = '%(node)s'
)
, storages as (
select *
from nodes_storage ns
where ns.model_id= '%(model)s'
	and ns.storage_id = '%(node)s'
)
, outfalls as (
select *
from nodes_outfalls as no2
where no2.model_id = '%(model)s'
	and no2.outfall_id = '%(node)s'
)
, nodes as (
    select mnc.node_id, mnc.lat, mnc.lon, j.elevation, j.init_depth, j.max_depth
    from model_node_coordinates mnc
    join junctions j
    	on mnc.node_id = j.junction_id
    where elevation notnull
    union all
    select mnc.node_id, mnc.lat, mnc.lon, s.elevation, s.init_depth, s.max_depth
    from model_node_coordinates mnc
    join storages s
    	on mnc.node_id = s.storage_id
    where elevation notnull
    union all
    select mnc.node_id, mnc.lat, mnc.lon, o.elevation, 0 as init_depth, 0 as max_depth
    from model_node_coordinates mnc
    join outfalls o
    	on mnc.node_id = o.outfall_id
    where elevation notnull
)
, subcatch as (
select *
from subcatchments s
where s.model_id  = '%(model)s'
	and s.outlet = '%(node)s'
)
, event_nodes as (
select *
from events_nodes en
where event_id = '%(event)s'
	and en.node_id = '%(node)s'
)
, event_subc as (
select  *
from events_subcatchments es
where es.event_id = '%(event)s'
)
, event_subc_outlet as (
select  event_subc.*, subcatch.outlet, subcatch.raingage_id
from subcatch
left join event_subc
	on subcatch.subcatchment_id = event_subc.subcatchment_id
)
, nodal_out_data as(
select
	en.node_id,
	coalesce (subcatchment_id, 'SIN CUENCA DE APORTE') as subcatchment_id,
	en.elapsed_time,
	en.depth_above_invert,
	en.flow_lost_flooding,
	en.hydraulic_head,
	en.lateral_inflow,
	en.total_inflow,
	en.volume_stored_ponded,
	coalesce (eso.evaporation_loss,0) as evaporation_loss,
	coalesce (eso.runoff_rate, 0) as runoff_rate,
	coalesce (eso.infiltration_loss, 0) as infiltration_loss,
    coalesce (eso.rainfall, 0) as rainfall
from event_nodes as en
left join event_subc_outlet as eso
	on eso.elapsed_time = en.elapsed_time
	and eso.outlet = en.node_id
)
, nodal_inp_data as (
select
	nodes.*,
	coalesce (s.area,0) as area,
	coalesce (s.imperv,0) as imperv,
	coalesce (s.slope, 0) as slope,
	coalesce (s.width,0) as width,
	coalesce (s.curb_len, 0) as curb_len,
	coalesce (s.raingage_id, '') as raingage_id
from nodes
left jOIN subcatch s
	on s.outlet = nodes.node_id
)
, nodal_data as(
select
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
from nodal_out_data as nod
left join nodal_inp_data as nid
	on nod.node_id = nid.node_id
),
rain_mdata as (
select *
from raingages_metadata rm
where rm.precipitation_id = '%(precip)s'
)
, rain_tseries as (
select *
from raingages_timeseries rt
where rt.precipitation_id = '%(precip)s'
)
, rain as(
select  rt.raingage_id, rt.elapsed_time, coalesce (rt.value, 0) as rainfall_acc, rm.format, rm.unit
from raingages_timeseries rt
join raingages_metadata as rm
	on rt.precipitation_id = rm.precipitation_id
)
select distinct nd.*, r.rainfall_acc, r.format, r.unit
from nodal_data nd
join rain r
	on nd.raingage_id = r.raingage_id
	and nd.elapsed_time = r.elapsed_time
    """% {'event': event, 'model': model, 'node':node_id, 'precip':precip}

    cur3 = pg_conn.cursor()
    cur3.execute(nodal_data_vars_query)
    nodal_data_result = cur3.fetchall()
    cur3.close()



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

df_nodes_indiv = (pd.DataFrame(pos)).T.reset_index().rename({'index':'nodo', 'lat':'lat', 'lon':'lon'}, axis=1)
#%%


# Step 1. Launch the application
app = dash.Dash()



# dropdown options
nodes = set(pos.keys())
opts = [{'label' : i, 'value' : i} for i in nodes]

# Step 3. Create a plotly figure
trace0 = go.Scatter(x=df_nodes_indiv['lat'], y=df_nodes_indiv['lon'],
                mode='markers',
                marker={'size': 5, 'color': 'lightgrey'})

fig = go.Figure(data=trace0, layout= go.Layout(margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=1100,
                            width=840,
                            plot_bgcolor='white',
        ))

#%%

# Step 4. Create a Dash layout
app.layout = html.Div([
                # adding a plot
                dcc.Graph(id = 'plot', figure = fig),
                # dropdown
                html.P([
                    html.Label("Choose a feature"),
                    dcc.Dropdown(id = 'opt',
                                 options = opts,
                                 #value = opts[0]
                                 )
                        ], style = {'width': '400px',
                                    'fontSize' : '20px',
                                    'position': 'fixed',
                                    'right': '30px',
                                    'bottom':'30px',
                                    'padding-right' : '100px',
                                    'padding-bottom': '150px',
                                    'display': 'inline-block',
                                    })
                      ])




# Step 5. Add callback functions
@app.callback(Output('plot', 'figure'),
             [Input('opt', 'value')])

def update_figure(node):

    traceRecode = []
    traceRecode.append(trace0)

    DG_i = build_tree_subgraph(node)

    # for node_id in DG_i.nodes:
    #     DG_i.nodes[node_id]['pos'] = pos[node_id]

    # Create Edges
    for edge in DG_i .edges():
        x0, y0 = DG_i.nodes[edge[0]]['pos']
        x1, y1 = DG_i.nodes[edge[1]]['pos']

        trace_edge_i = go.Scatter(
        x = tuple([x0, x1, None]),
        y = tuple([y0, y1, None]),
        hoverinfo='none',
        mode='lines+markers',
        line={'width': min(10, 20 * DG_i.edges[edge]['flow_rate'])},
        marker=dict(size=5, color='#125AEF'),
        line_shape='spline',
        opacity=1
        )

        traceRecode.append(trace_edge_i)


    #     # Create Edges
    # for node in DG_i.nodes():
    #     x, y = DG_i.nodes[node[0]]['pos']

    #     trace_node_i = go.Scatter(
    #     x = tuple([x]),
    #     y = tuple([y]),
    #     mode='markers',
    #     marker=dict(size=DG_i.nodes[node]['total_inflow'], color='SkyBlue'),
    #     )

    #     traceRecode.append(trace_node_i)

    # node_trace = go.Scatter(
    # x=[],
    # y=[],
    # text=[],
    # mode='markers',
    # hoverinfo='text',
    # marker=dict(
    #     color=[],
    #     size=10,
    #     colorbar=dict(
    #         thickness=15,
    #         title='Node Connections',
    #         xanchor='left',
    #         titleside='right'
    #     ),
    #     line=dict(width=2)))
    # for node in DG_i.nodes():
    #     x, y = DG_i.nodes[node]['pos']
    #     node_trace['x'] += tuple([x])
    #     node_trace['y'] += tuple([y])

    # traceRecode.append(node_trace)

    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",marker={'size': 20, 'color': '#125AEF'},opacity=0)
    index = 0
    for edge in DG_i.edges:
        x0, y0 = DG_i.nodes[edge[0]]['pos']
        x1, y1 = DG_i.nodes[edge[1]]['pos']

    for node_id in DG_i.nodes():
        x, y = DG_i.nodes[node_id]['pos']
        # hovertext = DG_i.nodes[node_id]['node_id']#"From: " + str(DG.edges[edge]['Source']) + "<br>" + "To: " + str(
    #         G.edges[edge]['Target']) + "<br>" + "TransactionAmt: " + str(
    #         G.edges[edge]['TransactionAmt']) + "<br>" + "TransactionDate: " + str(G.edges[edge]['Date'])
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])

    traceRecode.append(middle_hover_trace)

    trace2 = go.Scatter(x=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['lat'], y=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['lon'],
                    mode='markers',
                    name=node,
                    marker={'size': 15, 'color': 'black'},
    )

    traceRecode.append(trace2)


    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Nodos de Aporte y sentido del Escurrimiento', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=1100,
                            width=840,
                            plot_bgcolor='white',
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(DG_i.nodes[edge[0]]['pos'][0] + DG_i.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=(DG_i.nodes[edge[0]]['pos'][1] + DG_i.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    x=(DG_i.nodes[edge[1]]['pos'][0] * 3 + DG_i.nodes[edge[0]]['pos'][0]) / 4,
                                    y=(DG_i.nodes[edge[1]]['pos'][1] * 3 + DG_i.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor='#125AEF',
                                    arrowsize=2,
                                    arrowwidth=1,
                                    opacity=1
                                ) for edge in DG_i.edges]
                             )
                            }



    return figure






















#     node_trace = go.Scatter(x=[], y=[],text=[], mode='markers+text', textposition="bottom center",
#                              marker={'size': 20, 'color': 'LightSkyBlue'})

#     index = 0
#     for node_id in DG_i.nodes:
#         x, y = DG_i.nodes[node_id]['pos']
#         # hovertext = DG_i.nodes[node_id]['node_id']#"CustomerName: " + str(G.nodes[node]['CustomerName']) + "<br>" + "AccountType: " + str(
#     # #         G.nodes[node]['Type'])
#     # #     text = node1['Account'][index]
#         node_trace['x'] += tuple([x])
#         node_trace['y'] += tuple([y])
#     # #     node_trace['hovertext'] += tuple([hovertext])
#     # #     node_trace['text'] += tuple([text])
#         index = index + 1

#     traceRecode.append(node_trace)

#     # colors = list(Color('lightcoral').range_to(Color('darkred'), len(DG.edges())))
#     # colors = ['rgb' + str(x.rgb) for x in colors]

#     index = 0
#     for edge in DG_i.edges:
#         x0, y0 = DG_i.nodes[edge[0]]['pos']
#         x1, y1 = DG_i.nodes[edge[1]]['pos']
#         #weight = float(DG.edges[edge]['TransactionAmt']) / max(edge1['TransactionAmt']) * 10
#         trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
#                         mode='lines+markers',
#                         line={'width': 2},#weight},
#                         marker=dict(size=5, color='LightSkyBlue'),
#                         line_shape='spline',
#                         opacity=1,
#                         # hover_name=edge[1], hover_data=["continent", "pop"]
#                         )
#         traceRecode.append(trace)
#         index = index + 1


#     middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",marker={'size': 20, 'color': 'LightSkyBlue'},opacity=0)
#     index = 0
#     for edge in DG_i.edges:
#         x0, y0 = DG_i.nodes[edge[0]]['pos']
#         x1, y1 = DG_i.nodes[edge[1]]['pos']

#     for node_id in DG_i.nodes():
#         x, y = DG_i.nodes[node_id]['pos']
#         # hovertext = DG_i.nodes[node_id]['node_id']#"From: " + str(DG.edges[edge]['Source']) + "<br>" + "To: " + str(
#     #         G.edges[edge]['Target']) + "<br>" + "TransactionAmt: " + str(
#     #         G.edges[edge]['TransactionAmt']) + "<br>" + "TransactionDate: " + str(G.edges[edge]['Date'])
#     #     middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
#     #     middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
#     #     middle_hover_trace['hovertext'] += tuple([hovertext])
#         index = index + 1
#     traceRecode.append(middle_hover_trace)

#     trace2 = go.Scatter(x=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['x_coord'], y=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['y_coord'],
#                     mode='markers',
#                     name=node,
#                     marker={'size': 15, 'color': 'black'})

#     traceRecode.append(trace2)



#     figure = {
#         "data": traceRecode,
#         "layout": go.Layout(title='Nodos de Aporte y sentido del Escurrimiento', showlegend=False, hovermode='closest',
#                             margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
#                             xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
#                             yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
#                             height=940,
#                             width=940,
#                             clickmode='event+select',
#                             annotations=[
#                                 dict(
#                                     ax=(DG_i.nodes[edge[0]]['pos'][0] + DG_i.nodes[edge[1]]['pos'][0]) / 2,
#                                     ay=(DG_i.nodes[edge[0]]['pos'][1] + DG_i.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
#                                     x=(DG_i.nodes[edge[1]]['pos'][0] * 3 + DG_i.nodes[edge[0]]['pos'][0]) / 4,
#                                     y=(DG_i.nodes[edge[1]]['pos'][1] * 3 + DG_i.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
#                                     showarrow=True,
#                                     arrowhead=2,
#                                     arrowcolor='LightSkyBlue',
#                                     arrowsize=2,
#                                     arrowwidth=1,
#                                     opacity=1
#                                 ) for edge in DG_i.edges]
#                             )}
#     return figure
# # %%

# # def update_figure(node):

# #     fig = go.Figure(data = [trace0, trace2])
# #     return fig


# Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server(debug = True)
