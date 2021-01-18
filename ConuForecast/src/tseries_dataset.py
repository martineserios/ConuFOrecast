#%%
# -*- coding: utf-8 -*-
from numpy.core.fromnumeric import _var_dispatcher
import pandas as pd
import numpy as np
import networkx as nx
import psycopg2
import datatable as dt
import pickle
import plotly.express as px
import plotly
from collections import namedtuple, defaultdict
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


# import local packages
from ConuForecast.src.graph_utils import GraphEngine

# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from torch_geometric.data import Dataset, Data

class DBconnector():
    """
    Connection to the database
    """
    #host="172.18.0.1", port = 5555, database="base-ina", user="postgres", password="postgres"
    def __init__(self, url: str, port: int, database: str, user: str, password: str) -> None:
        self.pg_conn = psycopg2.connect(host=url, port=port, database=database, user=user, password=password)



class TseriesManager(GraphEngine):
    def __init__(self, model:str, event:str, precip:str, conn) -> None:
        super(TseriesManager, self).__init__(model, event, precip, conn)


    def precipitation_tseries(self):
        query = f"""
        WITH rain_meta AS 
        (
            SELECT
                * 
            FROM
                raingages_metadata rm 
            WHERE
                precipitation_id = '{self.precip}' 
        )
        ,
        rain_tseries AS 
        (
            SELECT
                * 
            FROM
                raingages_timeseries rt 
            WHERE
                precipitation_id = '{self.precip}' 
        )
        SELECT
            interval,
            rt.raingage_id,
            value,
            unit,
            format 
        FROM
            rain_tseries rt 
            JOIN
                rain_meta rm 
                ON rm.raingage_id = rt.raingage_id
        """

        cur2 = self.conn.cursor()
        cur2.execute(query)
        query_result = cur2.fetchall()

        rain_tseries_cols = [
        'elapsed_time',
        'raingage_id',
        'value',
        'unit',
        'format',
        ]

        NodeVars = namedtuple('Rain_Tseries', rain_tseries_cols)

        dt_nodes = dt.Frame([i for i in map(NodeVars._make, [i for i in query_result])], names=rain_tseries_cols)
        dt_nodes = [i for i in map(NodeVars._make, query_result)]

        return np.array(dt_nodes)


    def timeseries(self, item: str, vars_list:list, peak_data:bool=False):
        """
        Generates the timeseries of any variable of any element.
        """

        nodal_data_cols = ['elapsed_time'] + vars_list
        vars = ', '.join(nodal_data_cols)


        if item.startswith('NODO'):
            nodal_data_vars_query = f"""
            WITH model_node_coordinates AS 
                (
                    SELECT
                        * 
                    FROM
                        nodes_coordinates AS nc 
                    WHERE
                        nc.model_id = 'model_{self.event}' 
                    AND nc.node_id = '{item}' 
            )
            ,
            junctions AS 
            (
                SELECT
                    * 
                FROM
                    nodes_junctions nj 
                WHERE
                    nj.model_id = 'model_{self.event}' 
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
                    ns.model_id = 'model_{self.event}' 
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
                    no2.model_id = 'model_{self.event}' 
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
                    s.model_id = 'model_{self.event}' 
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
                    rm.precipitation_id = 'precipitation_{self.event}'
                    AND rm.raingage_id = (SELECT raingage_id FROM nodal_data limit 1)
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
                    AND rt.raingage_id = (SELECT raingage_id FROM nodal_data limit 1)
            )
            ,
            rain AS
            (
                select
                    rt.precipitation_id,
                    rt.raingage_id,
                    rt.elapsed_time,
                    COALESCE (rt.VALUE, 0) AS rainfall_acc,
                    rm.format,
                    rm.unit 
                FROM
                    rain_tseries rt 
                    LEFT JOIN
                        rain_mdata AS rm 
                        ON rt.precipitation_id = rm.precipitation_id 
            ),
            all_data AS
            (
            SELECT DISTINCT
                    nd.*,
                    r.rainfall_acc,
                    r.format,
                    r.unit 
            FROM
                nodal_data nd 
                LEFT JOIN
                    rain r 
                    ON nd.elapsed_time = r.elapsed_time
                    AND nd.raingage_id = r.raingage_id 
                ORDER by nd.elapsed_time
            )
            SELECT
                {vars}
            from
                all_data
            """

            cur5 = self.conn.cursor()
            cur5.execute(nodal_data_vars_query)
            nodal_data_result = cur5.fetchall()

            NodeVars = namedtuple('NodeVars', nodal_data_cols)

            dt_nodes = dt.Frame([i for i in map(NodeVars._make, [i for i in nodal_data_result])], names=nodal_data_cols)
            
            if peak_data:
                df = dt_nodes.loc[:, ['node_id','elapsed_time'] + vars].to_pandas()
                dt_nodes[:, 'elapsed_time'] = pd.to_datetime(dt_nodes.loc[:,'elapsed_time'])
                dt_nodes['time_to_peak'] = dt_nodes.iloc[(dt_nodes['depth_above_invert'].idxmax())]['elapsed_time']
                dt_nodes['peak'] = dt_nodes['depth_above_invert'].max()
                df = dt_nodes.set_index('elapsed_time')
                return dt_nodes
            else:
                return dt_nodes



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

            if len(vars_list) == 0:
                df = dt_links.to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df

            else:
                df = dt_links[:, ['link_id','elapsed_time'] + vars_list].to_pandas()
                df.loc[:,'elapsed_time'] = pd.to_datetime(df.loc[:,'elapsed_time'])
                df = df.set_index('elapsed_time')
                return df
    
    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        return agg

    def tseries_for_superv_learning(self, item:str, vars_list:list, n_in=1, n_out=1, dropnan=True):
        #TO-DO change proceess to datatable framework
        tseries_df = self.timeseries(item, vars_list, peak_data=False).to_pandas().set_index('elapsed_time')
        # tseries_df['rainfall_acc'] = tseries_df['rainfall_acc'].diff()

        
        # load dataset
        # dataset = read_csv('pollution.csv', header=0, index_col=0)
        tseries_rain_df = tseries_df.iloc[:, -1:]
        values = tseries_rain_df.values
        # integer encode direction
        # encoder = LabelEncoder()
        # values[:,4] = encoder.fit_transform(values[:,4])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(values, n_in, n_out, dropnan)
        # drop columns we don't want to predicts
        # reframed = reframed.iloc[:, [0, -1]]#drop(reframed.columns[[6,7,8,9,10]], axis=1, inplace=True)
        # reframed.drop_duplicates(inplace=True)

        cols = reframed.columns.tolist()
        # cols_y = [col for col in cols if ('var2(t)' in col)]
        # cols_X = [col for col in cols if ('var1(t-' in col)]
        # cols = cols_X + cols_y
        reframed = reframed.loc[:, cols]

        tseries_df = tseries_df.reset_index().iloc[:,1:]
        reframed = pd.concat([tseries_df.iloc[:, 0:-1], reframed], axis=1, ignore_index=True)
        reframed = reframed.fillna(0)
        # drop rows with NaN values
        # if dropnan:
        #     reframed.dropna(inplace=True)
               
        
        return reframed
        # return reframed, tseries_df.iloc[1:, :]




    
    def timeseries_viz(self, item:str, var:list):

        df_plot_0 = self.timeseries(item, ['rainfall'])
        df_plot_0 = df_plot_0.resample(f'30min').mean()
        plot_rainfall_max = 1.5 * df_plot_0['rainfall'].max()

        df_plot_1 = self.timeseries(item, var)
        plot_var_max = 1.5 * df_plot_1[var[0]].max()
        plot_var_min = 1.1 * df_plot_1[var[0]].min()
        splitted_var = var[0].split('_')
        plot_var_legend = ' '.join([word.capitalize() for word in splitted_var][:-1]) + f' [{splitted_var[-1]}]'
            


        subfig = plotly.make_subplots(specs=[[{"secondary_y": True}]])

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
# %%
