#!/usr/bin/env python3

import sympy as sp
from math import sin, cos, acos
import math
import mpmath
from pynverse import inversefunc
import quatternionPlotter as qp


def generateLattice(N, outFile, includeAdjoint=False):
    out = [[0., 0., 0., 0.] for i in range(2 * N)]

    def getCartesianCoords(psi, theta, phi):
        return [
            cos(psi),
            sin(psi) * cos(theta),
            sin(psi) * sin(theta) * cos(phi),
            sin(psi) * sin(theta) * sin(phi)
        ]

    for n in range(N):

        # Calculating Psi
        ps = inversefunc(lambda x: (x - (0.5 * sin(2 * x))),
                         y_values=(math.pi * n) / (N + 1))

        # Calculating Theta
        th = sp.acos(1 - (2 * sp.Mod(n * sp.sqrt(2), 1)))
        th = sp.N(th, 20)

        # Calculating Phi
        ph = 2 * sp.pi * sp.Mod(n * sp.sqrt(3), 1)
        ph = sp.N(ph, 20)

        out[n] = getCartesianCoords(ps, th, ph)
        out[N + n] = [
            getCartesianCoords(ps, th,
                               ph)[0], -getCartesianCoords(ps, th, ph)[1],
            -getCartesianCoords(ps, th, ph)[2],
            -getCartesianCoords(ps, th, ph)[3]
        ]


    outRange = N
    if includeAdjoint:
        outRange = 2 * N

    qp.plotPoints(out[:outRange])

    file = open(outFile, "w")
    for i in range(outRange):
        line = "\t".join([str(val) for val in out[i]]) + "\n"
        file.write(line)

    file.close()


def main():
    N = 120
    generateLattice(N, "../../numerics/testPart.csv")


if __name__ == "__main__":
    main()
