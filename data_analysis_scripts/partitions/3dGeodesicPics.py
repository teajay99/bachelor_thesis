#!/usr/bin/env python3

import numpy as np
import fresnel
import matplotlib
import matplotlib.pyplot as plt
import PIL
import sys
import os
import math
import sympy as sp

PHI = (np.sqrt(5) + 1) / 2

SAMPLES = 64
LIGHT_SAMPLES = 40
RESOLUTION = 600
#SAMPLES = 16
#LIGHT_SAMPLES = 10
#RESOLUTION = 100
ORIENTATION1 = [0.975528, 0.154508, -0.154508, -0.024472]
ORIENTATION2 = [0.975528, 0.154508, -0.154508, -0.024472]


def removeDulicates(points):
    out = []

    for p in points:
        found = False
        for q in out:
            matches = True
            for i in range(3):
                if np.abs(q[i] - p[i]) > 1e-8:
                    matches = False
            if matches == True:
                found = True
                break
        if found is False:
            out.append(p)

    return out


def getSubdivs(v1, v2, v3, subdivs):
    a = v1
    b = v2 - v1
    c = v3 - v1

    rate = 1.0 / (subdivs + 1)

    out = []

    for i in range(subdivs + 2):
        for j in range(subdivs + 2):
            if i + j < subdivs + 2:
                out.append(a + (i * rate * b) + (j * rate * c))
    return out


def get_ico_vertices(subdivs=0):
    verts = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            verts.append(np.array([0, i, j * PHI]))
            verts.append(np.array([i, j * PHI, 0]))
            verts.append(np.array([j * PHI, 0, i]))

    verts = [v / np.linalg.norm(v) for v in verts]

    minDist = 10
    for v in verts[1:]:
        dist = np.linalg.norm(v - verts[0])
        if dist < minDist:
            minDist = dist

    newVerts = []

    for v1 in verts:
        for v2 in verts:
            for v3 in verts:
                if (np.abs(np.linalg.norm(v1 - v2) - minDist) < 1e-8) and (
                        np.abs(np.linalg.norm(v1 - v3) - minDist) < 1e-8) and (
                            np.abs(np.linalg.norm(v2 - v3) - minDist) < 1e-8):
                    newVerts.extend(getSubdivs(v1, v2, v3, subdivs))

    for i in range(len(newVerts)):
        newVerts[i] += 0.1 * newVerts[i] / np.linalg.norm(newVerts[i])

    verts.extend(newVerts)
    return removeDulicates(verts)


def get_cube_vertices(subdivs=0):

    verts = []
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            for k in range(-1, 2, 2):
                verts.append(0.5 * np.array([i, j, k]))

    newVerts = []

    rate = 1.0 / (subdivs + 1)

    for sign in range(-1, 2, 2):
        for i in range(3):
            for j in range(subdivs + 2):
                for k in range(subdivs + 2):
                    newV = np.array([0., 0., 0.])
                    newV[i] = sign * 0.5
                    newV[(i + 1) % 3] = j * rate - 0.5
                    newV[(i + 2) % 3] = k * rate - 0.5
                    newVerts.append(newV)

    # for i in range(len(newVerts)):
    #     newVerts[i] += 0.07 * newVerts[i] / np.linalg.norm(newVerts[i])
    verts.extend(newVerts)

    return removeDulicates(verts / np.linalg.norm(verts[0]))


def render_stuff(vertList, fname):
    scene = fresnel.Scene(fresnel.Device("cpu"))
    #scene.lights = fresnel.light.cloudy()nd such a map for $S_3$ (and therefore \SUTwo) we start by embedding $S_3$ into $\mathbb{H}$ by introducing sp
    scene.lights = fresnel.light.lightbox()

    cmap = matplotlib.cm.get_cmap('tab10')

    for i in range(len(vertList)):
        poly_info = fresnel.util.convex_polyhedron_from_vertices(vertList[i])
        geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                     poly_info,
                                                     position=[i * 2.3, 0, 0],
                                                     orientation=ORIENTATION2,
                                                     outline_width=0.015)
        geometry.material = fresnel.material.Material(
            color=fresnel.color.linear(cmap(i)[:3]),
            roughness=0.07,
            specular=0.2,
            metal=0.8)
        geometry.outline_material = fresnel.material.Material(color=(0., 0.,
                                                                     0.),
                                                              roughness=0.1,
                                                              metal=1.0)

    scene.camera = fresnel.camera.Orthographic.fit(scene,
                                                   view='front',
                                                   margin=0.1)
    out = fresnel.pathtrace(scene,
                            samples=SAMPLES,
                            light_samples=LIGHT_SAMPLES,
                            w=RESOLUTION * len(vertList),
                            h=RESOLUTION)
    PIL.Image.fromarray(out[:], mode='RGBA').save(fname)


def render_cube(subdivs, fname):
    scene = fresnel.Scene(fresnel.Device("cpu"))
    scene.lights = fresnel.light.lightbox()

    cmap = matplotlib.cm.get_cmap('tab10')

    rate = 1.0 / (subdivs + 1)

    squares = []
    insideSquares = []

    for sign in range(-1, 2, 2):
        for face in range(3):
            for i in range(subdivs + 1):
                for j in range(subdivs + 1):
                    a, b, c, d = np.array([0., 0., 0.]), np.array(
                        [0., 0., 0.]), np.array([0., 0.,
                                                 0.]), np.array([0., 0., 0.])
                    ai, bi, ci, di = np.array([0., 0., 0.]), np.array(
                        [0., 0., 0.]), np.array([0., 0.,
                                                 0.]), np.array([0., 0., 0.])

                    INSIDE_MARGIN = 0.015
                    
                    a[face] = sign * 0.5
                    b[face] = sign * 0.5
                    c[face] = sign * 0.5
                    d[face] = sign * 0.5

                    ai[face] = sign * 0.5
                    bi[face] = sign * 0.5
                    ci[face] = sign * 0.5
                    di[face] = sign * 0.5

                    a[(face + 1) % 3] = (rate * i) - 0.5
                    b[(face + 1) % 3] = (rate * i) - 0.5
                    c[(face + 1) % 3] = (rate * (i + 1)) - 0.5
                    d[(face + 1) % 3] = (rate * (i + 1)) - 0.5

                    ai[(face + 1) % 3] = a[(face + 1) % 3] + INSIDE_MARGIN
                    bi[(face + 1) % 3] = b[(face + 1) % 3] + INSIDE_MARGIN
                    ci[(face + 1) % 3] = c[(face + 1) % 3] - INSIDE_MARGIN
                    di[(face + 1) % 3] = d[(face + 1) % 3] - INSIDE_MARGIN

                    a[(face + 2) % 3] = (rate * j) - 0.5
                    b[(face + 2) % 3] = (rate * (j + 1)) - 0.5
                    c[(face + 2) % 3] = (rate * j) - 0.5
                    d[(face + 2) % 3] = (rate * (j + 1)) - 0.5

                    ai[(face + 2) % 3] = a[(face + 2) % 3] + INSIDE_MARGIN
                    bi[(face + 2) % 3] = b[(face + 2) % 3] - INSIDE_MARGIN
                    ci[(face + 2) % 3] = c[(face + 2) % 3] + INSIDE_MARGIN
                    di[(face + 2) % 3] = d[(face + 2) % 3] - INSIDE_MARGIN

                    a /= np.sqrt(0.25 * 3)
                    b /= np.sqrt(0.25 * 3)
                    c /= np.sqrt(0.25 * 3)
                    d /= np.sqrt(0.25 * 3)

                    ai /= np.sqrt(0.25 * 3)
                    bi /= np.sqrt(0.25 * 3)
                    ci /= np.sqrt(0.25 * 3)
                    di /= np.sqrt(0.25 * 3)
                    squares.append([a, b, c, d, np.array([0., 0., 0.])])
                    insideSquares.append(
                        [ai, bi, ci, di,
                         np.array([0., 0., 0.])])

    for s in range(len(squares)):

        for i in range(1, 3):
            poly_info = 0
            if i == 1:
                poly_info = fresnel.util.convex_polyhedron_from_vertices(
                    [1.1 * p for p in squares[s]])
            else:
                poly_info = fresnel.util.convex_polyhedron_from_vertices(
                    [p / max(np.linalg.norm(p), 1e-8) for p in squares[s]])

            geometry = fresnel.geometry.ConvexPolyhedron(
                scene,
                poly_info,
                position=[i * 2.3, 0, 0],
                orientation=ORIENTATION1,
                outline_width=0.015)
            geometry.material = fresnel.material.Material(
                color=fresnel.color.linear(cmap(i)[:3]),
                roughness=0.07,
                specular=0.2,
                metal=0.8)
            geometry.outline_material = fresnel.material.Material(
                color=(0., 0., 0.), roughness=0.1, metal=1.0)

            if i == 2:
                poly_info = fresnel.util.convex_polyhedron_from_vertices([
                    (p * 1.0005) / max(np.linalg.norm(p), 1e-8)
                    for p in insideSquares[s]
                ])

                geometry = fresnel.geometry.ConvexPolyhedron(
                    scene,
                    poly_info,
                    position=[i * 2.3, 0, 0],
                    orientation=ORIENTATION1,
                    outline_width=0.)
                geometry.material = fresnel.material.Material(
                    color=fresnel.color.linear(cmap(i)[:3]),
                    roughness=0.07,
                    specular=0.2,
                    metal=0.8)

    poly_info = fresnel.util.convex_polyhedron_from_vertices([
        1.1 * np.array([0.5, 0.5, 0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([0.5, 0.5, -0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([0.5, -0.5, 0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([0.5, -0.5, -0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([-0.5, 0.5, 0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([-0.5, 0.5, -0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([-0.5, -0.5, 0.5]) / np.sqrt(0.25 * 3),
        1.1 * np.array([-0.5, -0.5, -0.5]) / np.sqrt(0.25 * 3)
    ])

    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                                 poly_info,
                                                 position=[0, 0, 0],
                                                 orientation=ORIENTATION1,
                                                 outline_width=0.015)
    geometry.material = fresnel.material.Material(color=fresnel.color.linear(
        cmap(0)[:3]),
                                                  roughness=0.07,
                                                  specular=0.2,
                                                  metal=0.8)
    geometry.outline_material = fresnel.material.Material(color=(0., 0., 0.),
                                                          roughness=0.1,
                                                          metal=1.0)

    scene.camera = fresnel.camera.Orthographic.fit(scene,
                                                   view='front',
                                                   margin=0.1)
    out = fresnel.pathtrace(scene,
                            samples=SAMPLES,
                            light_samples=LIGHT_SAMPLES,
                            w=RESOLUTION * 3,
                            h=RESOLUTION)
    PIL.Image.fromarray(out[:], mode='RGBA').save(fname)


def getSpherePoints(N):
    def getCartesianCoords(t, p):
        return [
            math.sin(t) * math.cos(p),
            math.sin(t) * math.sin(p),
            math.cos(t)
        ]

    out = [[0, 0, 0] for i in range(N)]

    for n in range(N):
        t = math.acos(1 - ((2 * (n)) / (N)))
        p = 2 * sp.pi * sp.Mod((n) * ((1 - sp.sqrt(5)) / 2), 1)
        p = float(sp.N(p, 20))
        out[n] = getCartesianCoords(t, p)

    return out


render_cube(3, "volleyball.png")
print("volleyball.png rendered succesfully")

render_stuff([
    get_ico_vertices(0),
    get_ico_vertices(3), [v / np.linalg.norm(v) for v in get_ico_vertices(3)]
], 'icosphere.png')
print("icosphere.png rendered succesfully")

render_stuff([getSpherePoints(20),
              getSpherePoints(100),
              getSpherePoints(500)], "fibonacci3d.png")
print("fibonacci3d.png rendered succesfully")
