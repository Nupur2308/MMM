from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
import base64
import datetime
from dash.exceptions import PreventUpdate
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

app = Dash()

df = pd.read_csv('Media variables 2021-11-18.csv',index_col =0)
df.index = pd.to_datetime(df.index)

features = df.columns

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
        initial_visible_month=datetime.date(2018, 10, 22),
        end_date=datetime.date(2021, 6, 28)
        )
    ], style = {'marginBottom':'5px'}),
    html.Button('Build Model', id='submit-val', n_clicks=0, style = {'padding':'5px'}),
    html.H1('Model Validation'),
    html.Div([
        html.H4('Training Fit'),
        html.P(id='rmse-train'),
        dcc.Graph(id='fit-train')
    ],
    style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        html.H4('Test Fit'),
        html.P(id='rmse-test'),
        dcc.Graph(id='fit-test')
    ],
    style={'width': '100%', 'display': 'inline-block'}),
    html.H1('Model Contributions'),
    html.Div([
        html.H4('Total Contributions over Model Period'),
        dcc.Graph(id='all_contr'),
        html.P(id='var-info'),
        html.H4('Week over Week contributions'),
        dcc.Graph(id='wow_contr')
    ],
    style={'width': '100%', 'display': 'inline-block'}),
    html.Button('Download Model Data', id='btn_csv', n_clicks=0, style = {'padding':'5px'}),
    dcc.Download(id="download-dataframe-csv")
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

def rmse(a,p):
    return np.sqrt(mse(a,p))

def build_model(yvar,start_date,end_date):
    xvar = features[~features.str.contains(yvar)] #Grab all the data except for the y variable
    media = df.loc[:,xvar]
    end_formatted = datetime.datetime.strptime(end_date, "%Y-%m-%d") #Format your end date to be able to split into train and test

    #Get training data
    X = np.array(media.loc[start_date:end_date])
    y = np.array(df.loc[start_date:end_date][yvar])

    #Get test data
    X_test = np.array(media.loc[end_formatted+datetime.timedelta(days=7):max(media.index)])
    y_test = np.array(df.loc[end_formatted+datetime.timedelta(days=7):max(df.index)][yvar])

    #Model Build and Fit
    param_range_lasso = {'lreg__alpha':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

    pipe_lr = Pipeline([
        ('sc',MinMaxScaler()),
        ('lreg',Lasso(tol=0.1))
        ])

    LR = GridSearchCV(estimator = pipe_lr,
                  param_grid = param_range_lasso,
                  scoring = 'neg_mean_squared_error',
                  n_jobs = -1,
                  cv = 10)

    LR.fit(X,y)

    #Model Validation

    ##Training validation
    trainp = LR.best_estimator_.predict(X)
    trainy = y
    train_rmse = np.sqrt(-LR.best_score_)

    ##Testing validation
    testp = LR.best_estimator_.predict(X_test)
    testy = y_test
    test_rmse = rmse(testy,testp)

    #Getting Contributions
    sc = MinMaxScaler()
    X_scaled = sc.fit(X).transform(X)
    lcoef = LR.best_estimator_['lreg'].coef_
    wow_contr = lcoef * X_scaled
    media_contr = pd.DataFrame(wow_contr,columns = xvar).set_index(df[start_date:end_date].index)
    intercept = LR.best_estimator_['lreg'].intercept_
    base_contr = pd.DataFrame([intercept]*X.shape[0], columns = ["Base"],index =df[start_date:end_date].index) #Calculating base contribution
    contr_table = media_contr.join(base_contr) #Adding base contribution to contr_table

    return train_rmse, trainp, trainy, test_rmse, testp, testy, contr_table, lcoef


@app.callback([
    Output('fit-train', 'figure'),
    Output('fit-test', 'figure'),
    Output('rmse-train','children'),
    Output('rmse-test','children'),
    Output('all_contr', 'figure')
    ],
    [Input('submit-val','n_clicks')],
    [State('yvar','value'),
    State('my-date-picker-range','start_date'),
    State('my-date-picker-range','end_date')])

def update_output(n_clicks, yvar, start_date, end_date):
    global start_d, end_d
    start_d = start_date
    end_d = end_date
    #Call the build model function and return results
    global train_rmse, trainp, trainy, test_rmse, testp, testy, contr_table, coef
    train_rmse, trainp, trainy, test_rmse, testp, testy, contr_table, coef  = build_model(yvar,start_date,end_date)

    #Trace for training actual values
    trace1 = go.Scatter(
            x = df.loc[start_date:end_date].index,
            y = trainy,
            mode = 'markers+lines',
            name = 'training set, actuals'
            )

    #Trace for the training prediction
    trace2 = go.Scatter(
            x = df.loc[start_date:end_date].index,
            y = trainp,
            mode = 'markers+lines',
            name = 'training set, predictions'
            )

    #Plotting training actual and predictions on a figure
    fig_1 = go.Figure(
                    data = [
                        trace1, trace2
                            ],
                        layout = go.Layout(
                                    xaxis= {'title': 'Weeks'},
                                    yaxis = {'title': yvar}
                                    )
                        )

    #Trace for the test actuals
    trace3 = go.Scatter(
            x = df.loc[end_date:max(df.index)].index,
            y = testy,
            mode = 'markers+lines',
            name = 'test set, actuals'
            )

    #Trace for the test predictions
    trace4 = go.Scatter(
            x = df.loc[end_date:max(df.index)].index,
            y = testp,
            mode = 'markers+lines',
            name = 'test set, predictions'
            )

    #Plotting test actuals and predictions on a figure
    fig_2 = go.Figure(
                    data = [
                        trace3, trace4
                            ],
                        layout = go.Layout(
                                    xaxis= {'title': 'Weeks'},
                                    yaxis = {'title': yvar}
                                    )
                        )

    #Getting training and test RMSE values
    tr_rmse = 'RMSE: '+ str(np.around(train_rmse,2))
    te_rmse = 'RMSE: ' + str(np.around(test_rmse,2))

    #Calculating total contributions and % of total contributions
    global contr_sum
    contr_sum = pd.DataFrame(contr_table.sum(axis=0),columns = ["total"])
    contr_sum['pct'] = (contr_sum.total / contr_sum.total.sum()).round(2)

    #Plotting contributions on a figure
    fig_3  = go.Figure(
                        data=
                            [go.Bar(
                                x=contr_sum.index.tolist(),
                                y=contr_sum['total'].tolist()
                            )],

                        layout=go.Layout(
                                    xaxis = {'title':'Variables',
                                            'tickangle':-45
                                            },
                                    yaxis = {'title':'Contribution'},
                                    margin = {'b':150,'l':150},
                                    hovermode = 'closest'
                                        )
                        )

    #Returning figures only if the build model button is clicked
    if n_clicks > 0:
        return fig_1, fig_2, tr_rmse, te_rmse, fig_3

@app.callback(
    [Output('var-info', 'children'),
    Output('wow_contr','figure')],
    [Input('all_contr', 'clickData')])
def callback_image(clickData):
    i = clickData['points'][0]['pointIndex']
    var_media = clickData['points'][0]['x']
    var_contr = np.around(clickData['points'][0]['y'],2)
    var_return = "Variable: " + str(var_media) + ", Contribution = " + str(var_contr) + ", Contribution % = " + str(contr_sum.pct[i])

    trace0 = go.Scatter(
            x = df.loc[start_d:end_d].index,
            y = contr_table[var_media],
            mode = 'markers+lines',
            name = var_media
            )
    fig_1 = go.Figure(
                    data = [
                        trace0
                            ],
                        layout = go.Layout(
                                    xaxis= {'title': 'Weeks'},
                                    yaxis = {'title': 'WoW Contribution'}
                                    )
                        )

    return var_return, fig_1

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(contr_table.to_csv, "contributions.csv")

if __name__ == '__main__':
    app.run_server()
