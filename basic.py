import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import base64
from datetime import date
from dash.exceptions import PreventUpdate

app = dash.Dash()

df = pd.read_csv('Media variables 2021-11-18.csv',index_col =0)
df.index = pd.to_datetime(df.index)

features = df.columns

def empty_plot(label_annotation):
    '''
    Returns an empty plot with a centered text.
    '''

    trace1 = go.Scatter(
        x=[],
        y=[]
    )

    data = [trace1]

    layout = go.Layout(
        showlegend=False,
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        annotations=[
            dict(
                x=0,
                y=0,
                xref='x',
                yref='y',
                text=label_annotation,
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=0
            )
        ]
    )

    fig = go.Figure(data=data, layout=layout)
    # END
    return fig

app.layout = html.Div([
        html.H1('Data Exploration'),
        html.Div([
            dcc.Dropdown(
                id='xaxis',
                options=[{'label': i.title(), 'value': i} for i in features],
                value='FAA Lead Form Impressions'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis',
                options=[{'label': i.title(), 'value': i} for i in features],
                value='Search-National Brand'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    dcc.Graph(id='feature-graphic'),
    html.H1('Model Settings'),
    html.P('Select the output variable and model date range'),
    html.Div([
        dcc.Dropdown(
            id='yvar',
            options=[{'label': i.title(), 'value': i} for i in features],
            value='FAA Lead Form Impressions'
        )
    ],
    style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.DatePickerRange(
        id='my-date-picker-range',
        min_date_allowed=min(df.index),
        max_date_allowed=max(df.index),
        initial_visible_month=date(2019, 8, 5),
        end_date=date(2019, 8, 25)
        )
    ], style = {'marginBottom':'5px'}),
    html.Button('Build Model', id='submit-val', n_clicks=0, style = {'padding':'5px'}),
    html.H1('Model Validation'),
    html.Div([
        html.H4('Training Fit'),
        html.P(id='rmse-train'),
        dcc.Graph(id='fit-train')
    ],
    style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        html.H4('Test Fit'),
        html.P(id='rmse-test'),
        dcc.Graph(id='fit-test')
    ],
    style={'width': '48%', 'display': 'inline-block'}),
    html.H1('Model Contributions')

], style={'padding':10,'fontFamily':'helvetica'})

@app.callback(
    Output('feature-graphic', 'figure'),
    [Input('xaxis', 'value'),
     Input('yaxis', 'value')])

def update_graph(xaxis_name, yaxis_name):
    return {
        'data': [go.Scatter(
            x=df[xaxis_name],
            y=df[yaxis_name],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={'title': xaxis_name.title()},
            yaxis={'title': yaxis_name.title()},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

@app.callback([
    Output('fit-train', 'figure'),
    Output('fit-test', 'figure'),
    Output('rmse-train','children'),
    Output('rmse-test','children')
    ],
    [Input('submit-val','n_clicks')],
    [State('yvar','value'),
    State('my-date-picker-range','start_date'),
    State('my-date-picker-range','end_date')])

def update_output(n_clicks, yvar, start_date, end_date):
    y_train = df.loc[start_date:end_date][yvar]
    y_test = df.loc[end_date:max(df.index)][yvar]

    fig_1 = go.Figure(
                    data = [
                        go.Scatter(
                                x = df.loc[start_date:end_date].index,
                                y = y_train,
                                mode = 'markers+lines'
                                )
                            ],
                        layout = go.Layout(
                                    xaxis= {'title': 'Weeks'},
                                    yaxis = {'title': yvar}
                                    )
                        )

    fig_2 = go.Figure(
                    data = [
                        go.Scatter(
                                x = df.loc[end_date:max(df.index)].index,
                                y = y_test,
                                mode = 'markers+lines'
                                )
                            ],
                        layout = go.Layout(
                                    xaxis= {'title': 'Weeks'},
                                    yaxis = {'title': yvar}
                                    )
                        )

    training_rmse = 'RMSE: '+ str(100)
    test_rmse = 'RMSE: ' + str(200)

    if n_clicks <1:
        return empty_plot('Nothing to Display'), empty_plot('Nothing to Display')
    else:
        return fig_1, fig_2, training_rmse, test_rmse





if __name__ == '__main__':
    app.run_server()
