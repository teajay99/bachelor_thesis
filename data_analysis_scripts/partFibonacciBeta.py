#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat
from partitions import fibonacci
from partitions import randomPart

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/partFibonacciBeta"

clrList = [
    "black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "red"
]


def main(collectData=False, collectRefData=False):
    ex = executor.executor(8)

    latSize = 8
    sweeps = 1500
    thermTime = 800
    betas = helpers.getRoundedLogSpace(0.1, 10, 25)
    deltas = helpers.getDeltas(betas)

    parts = [10, 20, 50, 100, 200, 500]

    print("""
===================================
      Generating Partitions
===================================
""")

    for i in range(len(parts)):
        if collectData:
            fibonacci.generateLattice(parts[i],
                                      WORK_DIR + "/partition{}.csv".format(i))
            randomPart.generateLattice(
                parts[i], WORK_DIR + "/randPartition{}.csv".format(i))

    print("""
===================================
     Collecting Reference Data
===================================
""")
    if collectRefData:
        ex.recordGPUData(latSize, betas, deltas, 5 * sweeps,
                         WORK_DIR + "/cont_data")
    ex.runEvaluator(WORK_DIR + "/cont_data", WORK_DIR + "/cont_data.csv",
                    thermTime)

    for i in range(len(parts)):
        print("""
===================================
  Collecting Partition Data {}/{}
===================================
        """.format(i + 1, len(parts)))
        if collectData:
            ex.recordGPUData(latSize,
                             betas,
                             1,
                             sweeps,
                             WORK_DIR + "/part{}_data".format(i),
                             partition="--partition-list",
                             partitionFile=WORK_DIR +
                             "/partition{}.csv".format(i))
            ex.recordGPUData(latSize,
                             betas,
                             1,
                             sweeps,
                             WORK_DIR + "/randPart{}_data".format(i),
                             partition="--partition-list",
                             partitionFile=WORK_DIR +
                             "/randPartition{}.csv".format(i))
        ex.runEvaluator(WORK_DIR + "/randPart{}_data".format(i),
                        WORK_DIR + "/randPart{}_data.csv".format(i), thermTime)
        ex.runEvaluator(WORK_DIR + "/part{}_data".format(i),
                        WORK_DIR + "/part{}_data.csv".format(i), thermTime)

    contData = np.loadtxt(WORK_DIR + "/cont_data.csv", dtype=np.float64)

    partPlaquettes = []
    randPlaquettes = []
    for i in range(len(parts)):
        data = np.loadtxt(WORK_DIR + "/part{}_data.csv".format(i),
                          dtype=np.float64)
        partPlaquettes.append(
            [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])
        data = np.loadtxt(WORK_DIR + "/randPart{}_data.csv".format(i),
                          dtype=np.float64)
        randPlaquettes.append(
            [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])

    contPlaquettes = np.array([
        ufloat(contData[i, 1], contData[i, 2])
        for i in range(len(contData[:, 0]))
    ])

    texTable = [betas, contPlaquettes]
    for i in range(len(parts)):
        texTable.append(partPlaquettes[i])

    pltLib.printTeXTable(np.array(texTable).transpose())

    pltLib.startNewPlot("$\\beta$", "$W(1,1)$", "")
    #pltLib.ax.set_yscale('symlog', linthresh=1e-4)
    pltLib.setLogScale(True, False)
    # pltLib.plot1DErrPoints(betas,
    #                        contPlaquettes,
    #                        label="continous (" +
    #                        str((2 * sweeps) - thermTime) + " sw.)")

    for i in range(len(parts)):
        pltLib.plot1DErrPoints(betas * (1.02**(i - 3)),
                               partPlaquettes[i],
                               label="$N = {}$ (".format(parts[i]) +
                               str(sweeps - thermTime) + " sw.)",
                               clr=clrList[i])
        # pltLib.plot1DErrPoints(betas * (1.02**i),
        #                        np.abs(randPlaquettes[i] - contPlaquettes),
        #                        label="Random: $N = {}$ (".format(parts[i]) +
        #                        str(sweeps - thermTime) + " sw.)",
        #                        clr=clrList[i + len(randPlaquettes)])

    pltLib.export("export/partFibonacciBeta.pgf", width=0.8)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
