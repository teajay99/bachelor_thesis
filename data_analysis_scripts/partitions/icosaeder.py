#!/usr/bin/env python3

import math




def get600Cell():
    #   Based on https://arxiv.org/pdf/1705.04910.pdf

    tau = (1 + math.sqrt(5)) / 2
    tauPrime = (1 - math.sqrt(5)) / 2

    out = []

    # R[0,0]
    for i in range(4):
        for a in range(2):
            el = [0., 0., 0., 0.]
            el[i] = (-1.)**a
            out.append(el)

    # R[1,0]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    out.append([
                        0.5 * ((-1.)**a), 0.5 * ((-1.)**b), 0.5 * ((-1.)**c),
                        0.5 * ((-1.)**d)
                    ])

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
                        0, 0.5 * ((-1.)**a), 0.5 * tauPrime * ((-1.)**b),
                        0.5 * tau * ((-1.)**c)
                    ]))

    sum = [0.,0.,0.,0.]
    for j in range(120):
        for i in range(4):
            sum[i] += out[j][i]/120
    print("4D:",sum)

    return out


def generateLattice(i, outFile):
    file = open(outFile, "w")
    for i in get600Cell():
        line = "\t".join([str(val) for val in i]) + "\n"
        file.write(line)

    file.close()


def main():
    generateLattice(1,"../../numerics/testPart.csv")


if __name__ == "__main__":
    main()
