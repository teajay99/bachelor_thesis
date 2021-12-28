#!/usr/bin/env python3

import helpers
import executor
import numpy as np
import pltLib
import os, shutil, subprocess, pathlib
from partitions import fibonacci
from uncertainties import ufloat
from pathlib import Path

WORK_DIR = "tmpData/fibDetailEval"


CLR = [
    "lightcoral", "firebrick", "limegreen", "green", "deepskyblue",
    "steelblue", "violet", "darkviolet"
]

def evalBeta(c, beta, collectData, cold):
    thermTime = 20000
    iterations = 1000

    ex = executor.executor(8)
    path = WORK_DIR + "/fibScan" + "{:.1f}".format(beta) + str(c)

    if cold:
        path += "_cold"

    if collectData and (not os.path.exists(path)):
        ex.recordGPUData(
            8,
            beta,
            1.0,  # doesn't matter for list partitions
            thermTime + iterations,
            path,
            partition="--partition-list",
            partitionOpt=WORK_DIR + "/fibList" + str(c) + ".csv",
            cold=cold,
            verbose=False,
            hits=20)
        ex.runEvaluator(path, path + ".csv", thermTime)
    data = np.loadtxt(path + ".csv", dtype=np.float64)
    print(data[1], end=": ")
    return data[1]


def main():
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

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
    betas = np.linspace(0.1, 10, 100)

    refData = np.loadtxt("tmpData/partitionEvaluation/ref_data.csv",
                         dtype=np.float64)
    refPlaquettes = np.array(
        [refData[i, 1] for i in range(len(refData[:, 0]))])

    resultsCold = []
    resultsHot = []
    fibCountsCold = []
    fibCountsHot = []

    for cold in [True]:
        betaIdx = 2

        pltLib.startNewPlot("$\\beta$","$P$","")
        pltLib.plotLine(betas, refPlaquettes,
                        clr="k",
                        label="$\\textrm{continuous SU}(2)$",
                        zorder=42)

        clrIdx = 0

        for c in fibCounts:
            plaq = 1
            measuredBetas = []
            measuredPlaquetes = []
            while betaIdx < (len(betas)-1):
                lastPlaq = plaq
                plaq = evalBeta(c, betas[betaIdx], collectData, cold)

                measuredBetas.append(betas[betaIdx])
                measuredPlaquetes.append(plaq)

                #if ((plaq - lastPlaq) / res) >= min(0.5, 1.0 /
                #                                    (betas[betaIdx]**2)):
                if ((betas[betaIdx] < 2.) and
                    ((plaq - lastPlaq) >=
                     (0.5 * res))) or ((betas[betaIdx] >= 2.) and (
                         (plaq - lastPlaq) >
                         (2 *
                          (helpers.weakCouplingExp6(betas[betaIdx]) -
                           helpers.weakCouplingExp6(betas[betaIdx - 1]))))):
                    print(
                        "Found Phase Transition for beta={:.2f} and lattice size={:d}"
                        .format(betas[betaIdx], c))

                    if cold:
                        resultsCold.append(ufloat(betas[betaIdx - 1], res))
                        fibCountsCold.append(c)
                    else:
                        resultsHot.append(ufloat(betas[betaIdx - 1], res))
                        fibCountsHot.append(c)

                    pltLib.ax.annotate('$\\beta_c(F_{'+str(c)+'})$', xy=(betas[betaIdx-1],lastPlaq),xytext=(betas[betaIdx-1],lastPlaq-0.2), arrowprops=dict(color=CLR[clrIdx % len(CLR)],width=0.1,headwidth=2))

                    betaIdx -= 1
                    plaq = 1

                    break
                else:
                    print("no transition found for beta={:.1f}".format(
                        betas[betaIdx]))

                if betaIdx == (len(betas)-1):
                    break
                else:
                    betaIdx += 1

            pltLib.plotPoints(measuredBetas, measuredPlaquetes, clr=CLR[clrIdx % len(CLR)])
            pltLib.plotLine(measuredBetas, measuredPlaquetes, clr=CLR[clrIdx % len(CLR)], alpha=0.5)
            clrIdx+=1
        pltLib.export("export/fibPhaseScanDecisions.png", width=2.5, height=0.9)
        pltLib.endPlot()


    otherPartsData = np.genfromtxt("fibDetail.csv")
    otherParts = [
        ufloat(otherPartsData[i, 2], otherPartsData[i, 3])
        for i in range(len(otherPartsData[:, 0]))
    ]
    otherCounts = otherPartsData[:, 1]
    otherPartsData = np.genfromtxt("fibDetail.csv", dtype='str')
    otherPartNames = otherPartsData[:, 0]

    for i in range(len(fibCountsCold)):
        print("\t".join([
            "$F_{" + str(fibCountsCold[i]) + "}$",
            str(fibCountsCold[i]),
            str(resultsCold[i].n),
            str(resultsCold[i].std_dev)
        ]))

    for i in range(len(fibCountsHot)):
        print("\t".join([
            "$F_{" + str(fibCountsHot[i]) + "}$",
            str(fibCountsHot[i]),
            str(resultsHot[i].n),
            str(resultsHot[i].std_dev)
        ]))

    for i in range(len(otherParts)):
        print("\t".join([
            str(otherPartNames[i]),
            str(otherCounts[i]),
            str(otherParts[i].n),
            str(otherParts[i].std_dev)
        ]))

    def fitFunc(x, a, b):
        return a * np.sqrt(x) + b
        #return x**(a) + b

    fibCountsCold = np.array([1.0 * i for i in fibCountsCold],
                             dtype=np.float64)

    popt, perr, crs = pltLib.makeFit1DErr(fibCountsCold, resultsCold, fitFunc)

    print(popt, perr, crs)

    pltLib.startNewPlot("$n$", "$\\beta_{\\textrm{ph.}}$", "")
    pltLib.setLogScale(True, False)
    pltLib.ax.set_xlim(7, 400)
    # pltLib.plot1DErrFitFunc(fitFunc,
    #                         popt,
    #                         8,
    #                         160,
    #                         label="$\\textrm{fit function}$",
    #                         clr="b",
    #                         log=True)
    # pltLib.plot1DErrPoints(fibCountsHot,
    #                        resultsHot,
    #                        clr="r",
    #                        label="$F_n \\textrm{ hot start}$")
    pltLib.plot1DErrPoints(fibCountsCold,
                           resultsCold,
                           clr="b",
                           label="$F_n")
    pltLib.plot1DErrPoints(otherCounts, otherParts)
    for i, txt in enumerate(otherPartNames):
        pltLib.ax.annotate(txt, (otherCounts[i] * 1.05, otherParts[i].n - 0.1))
    pltLib.export("export/fibPhaseScan.pgf", width=0.9, height=0.9)
    pltLib.export("export/fibPhaseScan.png", width=0.9, height=0.9)
    # pltLib.export("export/fibPhaseScanPres.pgf",
    #               width=1.10 * 0.65 / 1.2,
    #               height=0.9 * 2 * 0.75 / (1.2))

    pltLib.endPlot()


if __name__ == "__main__":
    main()
