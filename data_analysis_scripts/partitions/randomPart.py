#!/usr/bin/env python3

import numpy as np

def generateLattice(N, outFile, includeAdjoint=False):
    out = [[0., 0., 0., 0.] for i in range(2 * N)]



    for i in range(2*N):
        out[i] = np.random.normal(size=4)
        out[i] /= np.linalg.norm(out[i])

    for i in range(2 * N):
        line = "\t".join([str(val) for val in out[i]]) + "\n"
        file.write(line)

    file.close()


def main():
    N = 500
    generateLattice(N, "../../numerics/testPart.csv")


if __name__ == "__main__":
    main()
