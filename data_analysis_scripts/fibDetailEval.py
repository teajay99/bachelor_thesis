#!/usr/bin/env python3

import helpers
import executor
import numpy as np
import pltLib
import os, shutil, subprocess, pathlib
from partitions import fibonacci
from uncertainties import ufloat

WORK_DIR = "tmpData/fibDetailEval"


def evalBeta(c, beta, collectData, cold):
    thermTime = 20000
    iterations = 1000

    ex = executor.executor(8)
    path = WORK_DIR + "/fibScan" + "{:.1f}".format(beta) + str(c)

    if cold:
        path += "_cold"

    if collectData:  # and (not os.path.exists(path)):
        ex.recordGPUData(
            8,
            beta,
            1.0,  # doesn't matter for list partitions
            thermTime + iterations,
            path,
            partition="--partition-list",
            partitionOpt=WORK_DIR + "/fibList" + str(c) + ".csv",
            cold=cold,
            verbose=False)
        ex.runEvaluator(path, path + ".csv", thermTime)
    data = np.loadtxt(path + ".csv", dtype=np.float64)
    print(data[1], end=": ")
    return data[1]


def main():
    fibCounts = np.array(
        [int(i) for i in helpers.getRoundedLogSpace(8, 256, 41, 4)])[1:]

    print(fibCounts)

    collectData = False

    print("calculating lattices")

    for c in fibCounts:
        partPath = WORK_DIR + "/fibList" + str(c) + ".csv"
        if collectData:
            fibonacci.generateLattice(c, partPath)

    print("lattice calculations finished.\nLet's go\n")

    res = 0.1
    thresh = 0.95
    betas = np.linspace(0.1, 10, 100)

    refData = np.loadtxt("tmpData/partitionEvaluation/ref_data.csv",
                         dtype=np.float64)
    refPlaquettes = np.array(
        [refData[i, 1] for i in range(len(refData[:, 0]))])

    resultsCold = []
    resultsHot = []
    fibCountsCold = []
    fibCountsHot = []

    for cold in [True, False]:
        betaIdx = 10
        for c in fibCounts:
            while True:
                plaq = evalBeta(c, betas[betaIdx], collectData, cold)

                if plaq - refPlaquettes[betaIdx] >= 4e-2:
                    print(
                        "Found Phase Transition for beta={:.1f} and lattice size={:d}"
                        .format(betas[betaIdx], c))

                    if cold:
                        resultsCold.append(ufloat(betas[betaIdx], res))
                        fibCountsCold.append(c)
                    else:
                        resultsHot.append(ufloat(betas[betaIdx], res))
                        fibCountsHot.append(c)
                    break
                else:
                    print("no transition found for beta={:.1f}".format(
                        betas[betaIdx]))

                    if betaIdx == 99:
                        break
                    else:
                        betaIdx += 1

    pltLib.startNewPlot("", "", "")
    pltLib.plot1DErrPoints(fibCounts, resultsHot, clr="r")
    pltLib.plot1DErrPoints(fibCounts, resultsCold, clr="k")
    pltLib.endPlot()


if __name__ == "__main__":
    main()
