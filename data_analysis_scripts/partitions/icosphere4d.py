import sympy as sp
from pathos.pools import ProcessPool
from functools import cmp_to_key
import numpy as np

tau = (1 + sp.sqrt(5)) / 2
tauPrime = (1 - sp.sqrt(5)) / 2
one = sp.pi / sp.pi

edgeLength = 1 / tau
edgeLengthSqrd = edgeLength**2


def get600Cell():
    #   Based on https://arxiv.org/pdf/1705.04910.pdf

    out = []

    # R[0,0]
    for i in range(4):
        for a in range(2):
            el = [sp.pi * 0, sp.pi * 0, sp.pi * 0, sp.pi * 0]
            el[i] = (-one)**a
            out.append(el)

    # R[1,0]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    out.append([((-one)**a) / 2, ((-one)**b) / 2, ((-one) / 2),
                                ((-one)**d) / 2])

    # R[1,1]
    def permute(inp):
        a, b, c, d = inp[0], inp[1], inp[2], inp[3]
        return [[a, b, c, d], [a, c, d, b], [a, d, b, c], [b, a, d, c],
                [b, c, a, d], [b, d, c, a], [c, a, b, d], [c, b, d, a],
                [c, d, a, b], [d, a, c, b], [d, b, a, c], [d, c, b, a]]

    for a in range(2):
        for b in range(2):
            for c in range(2):
                out.extend(
                    permute([
                        sp.pi * 0, ((-one)**a) / 2, tauPrime * ((-one)**b) / 2,
                        tau * ((-one)**c) / 2
                    ]))

    matrixOut = [sp.Matrix(4, 1, k) for k in out]
    return matrixOut


def comparePoints(p1, p2):
    for i in range(4):
        if not (sp.simplify(p1[i, 0] - p2[i, 0]) == 0):
            return sp.N(p1[i, 0] - p2[i, 0], 3)
    return 0


POINT_CMP_KEY = cmp_to_key(comparePoints)


def getAllVertexCells(neighbours, v):
    # Source https://stackoverflow.com/questions/4555565/generate-all-subsets-of-size-k-from-a-set

    potentialCells = []

    def subset(set, left, index, list):
        if left == 0:
            potentialCells.append(list)
            return list

        l = [i for i in list]

        for i in range(index, len(set)):
            l.append(set[i])
            l = subset(set, left - 1, i + 1, l)
            l = [k for k in l[:-1]]
        return l

    def checkDistances(set):
        for i in range(len(set)):
            for j in range(len(set)):
                norm = ((set[i] - set[j]).transpose() * (set[i] - set[j]))[0,
                                                                           0]
                if j != i and (not (sp.simplify(norm - edgeLengthSqrd) == 0)):
                    return False
        return True

    subset(neighbours, 3, 0, [])

    cells = []

    for c in potentialCells:
        if checkDistances(c):
            c.append(v)
            c.sort(key=POINT_CMP_KEY)
            cells.append(c)

    return cells


def getAllCells(v, vertices):
    neighbours = []
    for w in vertices:
        norm = ((w - v).transpose() * (w - v))[0, 0]
        if sp.simplify(norm - edgeLengthSqrd) == 0:
            neighbours.append(w)
    print("*")
    return getAllVertexCells(neighbours, v)


def getCellCheckSum(cell):
    out = 0
    for i in range(4):
        for j in range(4):
            idx = (4 * i) + j
            out += (1 + sp.N(cell[i][j, 0], 10)) * (8**idx)
    return out


def getIcoSphere(subdivs):
    vertices = get600Cell()  #[70:110]

    cells = []

    pool = ProcessPool()
    rslt = pool.map(getAllCells, vertices, [vertices for i in vertices])
    for r in rslt:
        cells.extend(r)

    cellsNoDupl = []
    cellsNoDuplCheck = []

    cellsCheck = pool.map(getCellCheckSum, cells)

    for i in range(len(cells)):
        notIncluded = True
        for j in range(len(cellsNoDupl)):
            if np.abs(cellsNoDuplCheck[j] - cellsCheck[i]) < 1e-3:
                notIncluded = False
                break
        if notIncluded:
            cellsNoDupl.append(cells[i])
            cellsNoDuplCheck.append(cellsCheck[i])

    cells = [k for k in cellsNoDupl]

    #print(cells)
    print(len(cells))


getIcoSphere(2)
