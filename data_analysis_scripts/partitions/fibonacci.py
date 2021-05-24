#!/usr/bin/env python3

import sympy as sp
from math import sin, cos, acos
import math
import mpmath


def generateLattice(N, outFile):
    out = [[0., 0., 0., 0.] for i in range(2 * N)]

    def getCartesianCoords(psi, theta, phi):
        return [
            cos(psi),
            sin(psi) * cos(theta),
            sin(psi) * sin(theta) * cos(phi),
            sin(psi) * sin(theta) * sin(phi)
        ]

    for n in range(1, N + 1):

        # Calculating Psi
        ps = mpmath.findroot(
            lambda psi: (psi + (0.5 * sin(2 * psi))) -
            ((math.pi * n) / (N + 1)), (math.pi * n) / (N + 1))

        # Calculating Theta
        th = sp.acos(1 - (2 * sp.Mod(n * sp.sqrt(2), 1)))
        th = sp.N(th, 20)

        # Calculating Phi
        ph = 2 * sp.pi * sp.Mod(n * sp.sqrt(3), 1)
        ph = sp.N(ph, 20)

        out[2 * (n - 1)] = getCartesianCoords(ps, th, ph)
        out[2 * (n - 1) + 1] = [
            getCartesianCoords(ps, th,
                               ph)[0], -getCartesianCoords(ps, th, ph)[1],
            -getCartesianCoords(ps, th, ph)[2],
            -getCartesianCoords(ps, th, ph)[3]
        ]

    file = open(outFile, "w")

    for i in range(2*N):
        line = "\t".join([str(val) for val in out[i]]) + "\n"
        file.write(line)

    file.close()


def main():
    N = 500
    generateLattice(N, "../../numerics/testPart.csv")


if __name__ == "__main__":
    main()
