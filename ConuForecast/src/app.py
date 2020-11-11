#%%
# -*- coding: utf-8 -*-
# from _plotly_utils.colors import color_parser
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import sys
import os

# local imports
sys.path.append(os.path.join(os.getcwd(), '../..'))
from ConuForecast.src.graph_utils import GraphManager, DBconnector


# init db connection
conn = DBconnector('172.17.0.1', 5555, 'base-ina', 'postgres', 'postgres')

# main variables
MODEL = 'model_007'
EVENT = '007'
PRECIP = 'precipitation_007'
# ET = '2017-01-01 14:15:00'
ATTRS = {
    'nodes': ['elevation', 'area', 'imperv', 'slope'],
    'edges': ['flow_rate', 'flow_velocity']
    }

#init graph
conu_basin = GraphManager(model=MODEL, event=EVENT, precip=PRECIP, conn=conn)
#%%

## init app
# Step 1. Launch the application
app = dash.Dash()



# dropdown options
nodes = [i for i in conu_basin.pos_dict.keys()]
node_opts = [{'label' : i, 'value' : i} for i in nodes]
slider_opts = [{'label' : i, 'value' : i}for i in conu_basin.get_time_steps()[1][2::4]]


# Step 3. Create a plotly figure
pos = conu_basin.build_coordinates_dict()
df_nodes_indiv = (pd.DataFrame(pos)).T.reset_index().rename({'index':'nodo', 'lat':'lat', 'lon':'lon'}, axis=1)
#%%
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
                html.H1(['Nodos de Aporte y sentido del Escurrimiento']),
                # adding a plot
                dcc.Graph(id = 'plot', figure = fig),
                # dropdown
                html.P([
                    html.Label("Choose a feature"),
                    dcc.Dropdown(id = 'node',
                                 options = node_opts,
                                 value = node_opts[0]['value']
                                 )
                    ,'',
                    dcc.Dropdown(id='slider',
                                options= slider_opts,
                                value = slider_opts[0]['value']
                                )]
                        , style = {'width': '400px',
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
             [Input('node', 'value'), Input('slider', 'value')])

def update_figure(node, slider):

    traceRecode = []
    traceRecode.append(trace0)

    DG_i = conu_basin.build_subgraph(node=node, elapsed_time=slider, attrs=ATTRS, acc_data=False, in_place=False)

    for node_id in DG_i.nodes:
        DG_i.nodes[node_id]['pos'] = pos[node_id]

    # Create Edges
    for edge in DG_i.edges:
        x0 = DG_i.nodes[edge[0]]['pos']['lat']
        y0 = DG_i.nodes[edge[0]]['pos']['lon']
        x1 = DG_i.nodes[edge[1]]['pos']['lat']
        y1 = DG_i.nodes[edge[1]]['pos']['lon']

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


    # # Create Edges
    # for node in DG_i.nodes():
    #     x, y = DG_i.nodes[node]['pos']['lat'], DG_i.nodes[node]['pos']['lon']

    #     trace_node_i = go.Scatter(
    #     x = tuple([x]),
    #     y = tuple([y]),
    #     mode='markers',
    #     marker=dict(size=DG_i.nodes[node]['elevation'], color='SkyBlue'),
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
    #     x, y = DG_i.nodes[node]['pos']['lat'], DG_i.nodes[node]['pos']['lon']
    #     node_trace['x'] += tuple([x])
    #     node_trace['y'] += tuple([y])

    # traceRecode.append(node_trace)

    # middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",marker={'size': 20, 'color': '#125AEF'},opacity=0)
    # index = 0
    # for edge in DG_i.edges:
    #     x0, y0 = DG_i.nodes[edge[0]]['pos']
    #     x1, y1 = DG_i.nodes[edge[1]]['pos']



    # traceRecode.append(middle_hover_trace)

    trace2 = go.Scatter(x=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['lat'], y=df_nodes_indiv[df_nodes_indiv['nodo'] == node]['lon'],
                    mode='markers',
                    name=node,
                    marker={'size': 15, 'color': 'black'},
    )

    traceRecode.append(trace2)


    figure = {
        "data": traceRecode,
        "layout": go.Layout(showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=1100,
                            width=840,
                            plot_bgcolor='white',
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(DG_i.nodes[edge[0]]['pos']['lat'] + DG_i.nodes[edge[1]]['pos']['lat']) / 2,
                                    ay=(DG_i.nodes[edge[0]]['pos']['lon'] + DG_i.nodes[edge[1]]['pos']['lon']) / 2, axref='x', ayref='y',
                                    x=(DG_i.nodes[edge[1]]['pos']['lat'] * 3 + DG_i.nodes[edge[0]]['pos']['lat']) / 4,
                                    y=(DG_i.nodes[edge[1]]['pos']['lon'] * 3 + DG_i.nodes[edge[0]]['pos']['lon']) / 4, xref='x', yref='y',
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
