#!/usr/bin/env python3
import sympy as sp
import numpy as np
from sympy.algebras.quaternion import Quaternion
from pathos.pools import ProcessPool

import itertools

eta = sp.sqrt(5) / 4
tau = (sp.sqrt(5) + 1) / 2

one = sp.pi / sp.pi


def permuteEven(inp):
    a, b, c, d = inp.a, inp.b, inp.c, inp.d
    return [
        Quaternion(a, b, c, d),
        Quaternion(a, c, d, b),
        Quaternion(a, d, b, c),
        Quaternion(b, a, d, c),
        Quaternion(b, c, a, d),
        Quaternion(b, d, c, a),
        Quaternion(c, a, b, d),
        Quaternion(c, b, d, a),
        Quaternion(c, d, a, b),
        Quaternion(d, a, c, b),
        Quaternion(d, b, a, c),
        Quaternion(d, c, b, a)
    ]


def removeDuplicates(points):

    out = []

    for p in points:
        found = False
        for q in out:
            if sp.simplify(p - q) == Quaternion(0, 0, 0, 0):
                found = True
                break
        if found is False:
            out.append(p)

    return out


def removeFloatDulicates(points):
    out = []

    for p in points:
        found = False
        for q in out:
            matches = True
            for i in range(4):
                if np.abs(q[i] - p[i]) > 1e-8:
                    matches = False
            if matches == True:
                found = True
                break
        if found is False:
            out.append(p)

    return out


def get5Cell():
    out = [
        Quaternion(1, 0, 0, 0),
        Quaternion(-one / 4, eta, eta, eta),
        Quaternion(-one / 4, -eta, -eta, eta),
        Quaternion(-one / 4, -eta, eta, -eta),
        Quaternion(-one / 4, eta, -eta, -eta)
    ]

    return out


def get16Cell():
    out = []
    for i in range(4):
        for sign in range(-1, 2, 2):
            coords = [0 * one, 0 * one, 0 * one, 0 * one]
            coords[i] = sign * one
            out.append(Quaternion(*coords))
    return out


def get8Cell():
    out = []
    for i in range(16):
        coords = [one / 2, one / 2, one / 2, one / 2]
        sign = [1 - 2 * ((i & (1 << k)) >> k) for k in range(4)]
        #print(sign)
        coords = [coords[k] * sign[k] for k in range(4)]
        out.append(Quaternion(*coords))
    return out


def get24Cell():
    out = get16Cell()
    out.extend(get8Cell())
    return out


def get600Cell():
    out = get24Cell()

    for i in range(8):
        coords = [tau / 2, one / 2, 1 / (2 * tau), 0]
        for k in range(3):
            sign = 1 - 2 * ((i & (1 << k)) >> k)
            coords[k] *= sign
        out.extend(permuteEven(Quaternion(*coords)))

    return out


def get120Cell():
    out = []
    for p in get5Cell():
        for q in get600Cell():
            out.append(
                Quaternion(1 / sp.sqrt(2), 1 / sp.sqrt(2), 0, 0) * p * q)
    return out


sPPoints = get120Cell()
#points = [[sp.N(q.a), sp.N(q.b), sp.N(q.c), sp.N(q.d)] for q in sPPoints]
#print(points)

id = Quaternion(1, 0, 0, 0)

u = sPPoints[0].conjugate() * id

sPPoints = [p * u for p in sPPoints]

points = [[sp.N(q.a), sp.N(q.b), sp.N(q.c), sp.N(q.d)] for q in sPPoints]

#[print(p) for p in sPPoints]


def getNeighbours(index):
    neighbours = []

    for i in range(len(points)):
        if np.abs(
                np.dot(np.array(points[i]), np.array(points[index])) -
                0.963525491562421) < 1e-8:
            neighbours.append(sPPoints[i])

    #[print(sp.simplify(p)) for p in neighbours]

    print("\n", sPPoints[index], "\n")
    return neighbours


#print(sPPoints[:8])


def testMetropolisRotation():
    neighbours = [[] for i in range(4)]

    neighbours[0] = getNeighbours(0)
    neighbours[1] = getNeighbours(2)
    neighbours[2] = getNeighbours(7)
    neighbours[3] = getNeighbours(4)

    indexPerms = list(itertools.permutations([0, 1, 2, 3]))
    indexPerms = [list(k) for k in indexPerms]

    indexPermList = []
    for a in indexPerms:
        for b in indexPerms:
            for c in indexPerms:
                indexPermList.append([[0, 1, 2, 3], a, b, c])

    c600 = [sp.Matrix([[q.a], [q.b], [q.c], [q.d]]) for q in get600Cell()]

    c600_num = [
        np.array([np.float64(sp.N(p[i, 0])) for i in range(4)]) for p in c600
    ]

    neighbours_num = [[] for i in range(4)]
    for j in range(4):
        neighbours[j] = [
            sp.Matrix([[q.a], [q.b], [q.c], [q.d]]) for q in neighbours[j]
        ]
        neighbours_num[j] = [
            np.array([np.float64(sp.N(p[i, 0])) for i in range(4)])
            for p in neighbours[j]
        ]

    def checkPermSymb(perm):

        rotMs = [sp.Matrix([[0, 0, 0, 0] for i in range(4)]) for k in range(4)]

        for k in range(4):
            for j in range(4):
                q = neighbours[j][perm[j][k]]
                rotMs[k][:, j] = q

        # [
        #     print(sp.simplify(rot * sp.Matrix([[1], [0], [0], [0]])))
        #     for rot in rotMs
        # ]
        #
        # [print(sp.simplify(k)) for k in neighbours[0]]

        newPoints = []
        newPoints.extend(c600)
        for rot in rotMs:
            for p in c600:
                newPoints.append(rot * p)

        newFPoints = [[sp.N(p[i, 0], 10) for i in range(4)] for p in newPoints]

        dists = []
        for p in newFPoints:
            for q in newFPoints:
                dists.append(np.dot(np.array(p), np.array(q)))

        #dists = np.sort(dists)
        #neighbourDists = dists[600:4 * 600]

        neighbourCounter = 0
        for d in dists:
            if np.abs(d - 0.963525491562421) < 1e-8:
                neighbourCounter += 1

        #print(neighbourDists)
        if neighbourCounter == 600 * 4:
            print("SUCCES!!!!!")
            print([sp.simplify(m) for m in rotMs])
            print(perm)
            print(neighbourCounter)
        else:
            print(perm, " is not the solution. neighbours:", neighbourCounter)

    def checkPermNum(perm):

        rotMs = [
            np.array([[0., 0., 0., 0.] for i in range(4)]) for k in range(4)
        ]

        for k in range(4):
            for j in range(4):
                q = neighbours_num[j][perm[j][k]]
                for i in range(4):
                    rotMs[k][i, j] = q[i]

            #print(rotMs[k], q)

        # [
        #     print(sp.simplify(rot * sp.Matrix([[1], [0], [0], [0]])))
        #     for rot in rotMs
        # ]
        #
        # [print(sp.simplify(k)) for k in neighbours[0]]

        newPoints = []
        newPoints.extend(c600_num)
        for rot in rotMs:
            for p in c600_num:
                newPoints.append(np.matmul(rot, p))

        dists = []
        for p in newPoints:
            for q in newPoints:
                dists.append(np.dot(p, q))

        neighbourCounter = 0
        for d in dists:
            if np.abs(d - 0.963525491562421) < 1e-6:
                neighbourCounter += 1

        #print(neighbourDists)
        if neighbourCounter == 600 * 4:
            print("SUCCES!!!!!")
            print(rotMs)
            print(perm)
        else:
            print(perm, " is not the solution. neighbours:", neighbourCounter)

    #pool = ProcessPool()
    #rslt = pool.map(checkPermNum, indexPermList)

    checkPermSymb([[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]])


    dists = []
    for p in points:
        for q in points:
            dists.append(np.dot(np.array(p), np.array(q)))

    neighbourCounter = 0
    for d in dists:
        if np.abs(d - 0.963525491562421) < 1e-6:
            neighbourCounter += 1

    print(neighbourCounter)
