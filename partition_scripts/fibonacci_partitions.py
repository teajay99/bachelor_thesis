#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import math
from math import sin, cos, acos, sqrt


def getSpherePoints(N):
    def getCartesianCoords(t, p):
        return [sin(t) * cos(p), sin(t) * sin(p), cos(t)]

    out = [[0, 0, 0] for i in range(N)]

    for n in range(N):
        t = acos(1 - ((2 * n) / (N + 1)))
        p = 2 * math.pi * ((n * sqrt(2)) % 1)
        out[n] = getCartesianCoords(t, p)

    return out


def getCubePoints(N):
    out = [[0, 0, 0] for i in range(N)]

    for n in range(N):
        out[n] = [n / (N + 1), (n * sqrt(2)) % 1, 0]

    return out


def getWebApp():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # assume you have a "long-form" data frame
    # see https://plotly.com/python/px-arguments/ for more options
    data = getSpherePoints(400)
    df = pd.DataFrame({
        "x": [i[0] for i in data],
        "y": [i[1] for i in data],
        "z": [i[2] for i in data]
    })

    fig = px.scatter_3d(df, x="x", y="y", z="z", size_max=0.1)

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='''
        Dash: A web application framework for Python.
        '''),
        html.Div(children=[
            html.Label("Manifold"),
            dcc.Dropdown(options=[{
                'label': 'Sphere',
                'value': 'Sphere'
            }, {
                'label': 'Torus',
                'value': 'Torus'
            }, {
                'label': 'Cube',
                'value': 'Cube'
            }],
                         value="S2"),
            html.Label("Points"),
            dcc.Slider(min=2,
                       max=500,
                       value=42,
                       tooltip={
                           "always_visible": False,
                           "placement": "top"
                       })
        ],
                 style={'columnCount': 2}),
        dcc.Graph(id='example-graph', figure=fig)
    ])

    return app


if __name__ == '__main__':
    app = getWebApp()

    app.run_server(debug=True)
