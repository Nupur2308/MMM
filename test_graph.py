import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

df = pd.read_csv('Media variables 2021-11-18.csv',index_col =0)
df.index = pd.to_datetime(df.index)

df2 = pd.DataFrame(df.sum(axis=0),columns = ["total"])
df2['pct'] = (df2.total / df2.total.sum()).round(2)

fig_3  = go.Figure(
                        data=
                            [go.Bar(
                                x=df2.index.tolist(),
                                y=df2['total'].tolist(),
                                text = df2.pct,
                                hoverinfo = "text"
                            )],

                        layout=go.Layout(
                                    xaxis = dict(title='Variables', tickangle=-45,categoryorder = 'total ascending'),
                                    yaxis = {'title':'Contribution'},
                                    margin = dict(b =150,l=150)
                                        )
                        )

pyo.plot(fig_3, filename='fig3.html')
