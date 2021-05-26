#!/usr/bin/env python3

import numpy as np
#import quatternionPlotter as qp


def generateLattice(N, outFile, includeAdjoint=False):
    out = [[0., 0., 0., 0.] for i in range(2 * N)]

    for i in range(N):
        out[i] = np.random.normal(size=4)
        out[i] /= np.linalg.norm(out[i])
        out[i + N] = [out[i][0], -out[i][1], -out[i][2], -out[i][3]]

    file = open(outFile, "w")

    outRange = N
    if includeAdjoint:
        outRange = 2 * N

    for i in range(outRange):
        line = "\t".join([str(val) for val in out[i]]) + "\n"
        file.write(line)

    file.close()


def main():
    N = 500
    generateLattice(N, "../../numerics/testPart.csv")


if __name__ == "__main__":
    main()
