#!/usr/bin/env python3

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import math
from math import sin, cos, acos, sqrt
import sympy as sp
import mpmath
import numpy as np


def getSpherePoints(N):
    def getCartesianCoords(t, p):
        return [sin(t) * cos(p), sin(t) * sin(p), cos(t)]

    out = [[0, 0, 0] for i in range(N)]

    for n in range(N):
        t = acos(1 - ((2 * (n + 1)) / (N + 1)))
        p = 2 * sp.pi * sp.Mod((n + 1) * sp.sqrt(2), 1)
        p = sp.N(p, 20)
        out[n] = getCartesianCoords(t, p)

    return out


def getCubePoints(N):
    out = [[0, 0, 0] for i in range(N**2)]

    for n in range(N):
        out[n] = [(n + 1) / (N + 1), ((n + 1) * sqrt(2)) % 1, 0]

    return out


def getTorusPoints(N):
    R = 4
    r = 2

    def getCartesianCoords(t, p):

        return [(R + (r * cos(p))) * cos(t), (R + (r * cos(p))) * sin(t),
                r * sin(p)]

    out = [[0, 0, 0] for i in range(N)]

    for n in range(N):
        p = p = 2 * sp.pi * sp.Mod((n + 1) * sp.sqrt(2), 1)
        p = sp.N(p, 20)
        t = mpmath.findroot(
            lambda theta: ((1 / (2 * np.pi * R)) *
                           (R * theta - r * sin(theta))) - ((n + 1) / (N + 1)),
            3)
        print(t)
        out[n] = getCartesianCoords(t, p)
    return out


def getWebApp():
    #external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    #external_stylesheets = ['https://raw.githubusercontent.com/plotly/dash-app-stylesheets/master/dash-technical-charting.css']

    #app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app = dash.Dash(__name__)

    # assume you have a "long-form" data frame
    # see https://plotly.com/python/px-arguments/ for more options

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
                         value="Sphere",
                         id="mfld-select"),
            html.Label("Points", id="ptcnt"),
            dcc.Slider(min=2, max=500, value=42, id="pt-select")
        ],
                 style={'columnCount': 2}),
        dcc.Graph(id='point-plot', config={"fillFrame": True})
    ],
                          style={"height": "100%"})

    @app.callback(Output("ptcnt", "children"), Output("point-plot", "figure"),
                  Input("mfld-select", "value"), Input("pt-select", "value"))
    def cb_parameters_changed(mfld, pt_count):
        points = []
        if mfld == "Sphere":
            points = getSpherePoints(pt_count)
        elif mfld == "Torus":
            points = getTorusPoints(pt_count)
        elif mfld == "Cube":
            points = getCubePoints(pt_count)

        df = pd.DataFrame({
            "x": [i[0] for i in points],
            "y": [i[1] for i in points],
            "z": [i[2] for i in points]
        })

        fig = px.scatter_3d(df, x="x", y="y", z="z", size_max=0.1)

        fig.update_layout(transition_duration=500)

        return "Points: " + str(pt_count), fig

    return app


if __name__ == '__main__':
    app = getWebApp()

    app.run_server(debug=True)
