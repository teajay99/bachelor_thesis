import numpy as np
import itertools
import math
import quatternionPlotter as qp

eps = 1e-3


def getSubdividedCube(subdivs):
    length = subdivs + 2
    coords = np.linspace(-1, 1, length)
    v = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if j < i:
                continue
            for k in range(len(coords)):
                if k < j:
                    continue
                v.append([coords[i], coords[j], coords[k]])
    return v


def checkSum(vert):
    out = 0
    for i in range(4):
        out += (1 + vert[i]) * (8**i)
    return out


def getSubdividedTesseract(subdivs):
    vertices = []
    for v in getSubdividedCube(subdivs):
        v.append(1)
        vertices.extend(list(itertools.permutations(v)))
        v[3] = -1
        vertices.extend(list(itertools.permutations(v)))

    vertChecks = [checkSum(k) for k in vertices]
    noDuplVert = []
    noDuplVertChecks = []

    for i in range(len(vertices)):
        notIncluded = True
        for j in range(len(noDuplVert)):
            if np.abs(vertChecks[i] - noDuplVertChecks[j]) < eps:
                notIncluded = False
                break
        if notIncluded:
            noDuplVert.append(vertices[i])
            noDuplVertChecks.append(vertChecks[i])

    vertices = [list(k) for k in noDuplVert]
    return vertices
    #print(len(vertices))


def generateLattice(subdivs, outFile):
    vertices = getSubdividedTesseract(subdivs)

    for i in range(len(vertices)):
        norm = np.sqrt(np.sum(np.array(vertices[i])**2))
        for j in range(4):
            #print(vertices[i])
            vertices[i][j] /= norm

    #qp.plotPoints(vertices)
    print(len(vertices))

    file = open(outFile, "w")
    for i in range(len(vertices)):
        line = "\t".join([str(val) for val in vertices[i]]) + "\n"
        file.write(line)

    file.close()


def main():
    N = 1
    for N in range(10):
        #generateLattice(N, "../../numerics/testPart.csv")
        vCount = 16 + (8 * (N**3)) + (24 * (N**2)) + (32 * N)
        print(N,":",vCount)


if __name__ == "__main__":
    main()
