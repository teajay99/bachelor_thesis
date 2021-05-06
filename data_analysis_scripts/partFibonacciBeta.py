#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat
from partitions import fibonacci

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/partFibonacciBeta"

clrList = ["r", "b", "g", "c", "m", "y"]


def main():
    ex = executor.executor(8)

    latSize = 4
    sweeps = 1500
    thermTime = 500
    betas = helpers.getRoundedLogSpace(0.025, 25, 50)
    deltas = helpers.getDeltas(betas)

    collectData = False

    parts = [50, 500, 5000]

    print("""
===================================
      Generating Partitions
===================================
""")

    for i in range(len(parts)):
        if collectData:
            fibonacci.generateLattice(parts[i],
                                      WORK_DIR + "/partition{}.csv".format(i))

    print("""
===================================
     Collecting Reference Data
===================================
""")
    if collectData:
        ex.recordGPUData(latSize, betas, deltas, 2 * sweeps,
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
                             deltas,
                             sweeps,
                             WORK_DIR + "/part{}_data".format(i),
                             partition=WORK_DIR + "/partition{}.csv".format(i))
            ex.runEvaluator(WORK_DIR + "/part{}_data".format(i),
                            WORK_DIR + "/part{}_data.csv".format(i), thermTime)

    contData = np.loadtxt(WORK_DIR + "/cont_data.csv", dtype=np.float64)

    partPlaquettes = []
    for i in range(len(parts)):
        data = np.loadtxt(WORK_DIR + "/part{}_data.csv".format(i),
                          dtype=np.float64)
        partPlaquettes.append(
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
    pltLib.setLogScale(True, False)
    pltLib.plot1DErrPoints(betas,
                           contPlaquettes,
                           label="continous (" +
                           str((2 * sweeps) - thermTime) + " sw.)")

    for i in range(len(parts)):
        pltLib.plot1DErrPoints(betas,
                               partPlaquettes[i],
                               label="$N = {}$ (".format(parts[i]) +
                               str(sweeps - thermTime) + " sw.)",
                               clr=clrList[i])

    pltLib.export("export/partFibonacciBeta.pgf",width=0.8)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
