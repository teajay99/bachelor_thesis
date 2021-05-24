#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat
from partitions import fibonacci
from partitions import icosaeder

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/partIcoBeta"

clrList = ["r", "b", "g", "p", "o"]


def main(collectData=False):
    ex = executor.executor(8)

    latSize = 16
    sweeps = 1500
    thermTime = 1000
    #betas = helpers.getRoundedLogSpace(0.5, 7, 50)
    betas = np.linspace(5, 7, 21)
    deltas = helpers.getDeltas(betas)

    if collectData:
        print("""
===================================
     Collecting Reference Data
===================================
""")
        ex.recordGPUData(latSize, betas, deltas, 2 * sweeps,
                         WORK_DIR + "/cont_data")

        print("""
===================================
    Collecting Partition Data
===================================
""")
        ex.recordGPUData(latSize,
                         betas,
                         deltas,
                         sweeps,
                         WORK_DIR + "/iko_data",
                         partition="--partition-iko")

    ex.runEvaluator(WORK_DIR + "/cont_data", WORK_DIR + "/cont_data.csv",
                        thermTime)
    ex.runEvaluator(WORK_DIR + "/iko_data", WORK_DIR + "/iko_data.csv",
                        thermTime)

    contData = np.loadtxt(WORK_DIR + "/cont_data.csv", dtype=np.float64)
    partData = np.loadtxt(WORK_DIR + "/iko_data.csv", dtype=np.float64)

    contPlaquettes = np.array([
        ufloat(contData[i, 1], contData[i, 2])
        for i in range(len(contData[:, 0]))
    ])

    partPlaquettes = np.array([
        ufloat(partData[i, 1], partData[i, 2])
        for i in range(len(partData[:, 0]))
    ])

    texTable = [betas, contPlaquettes, partPlaquettes]

    pltLib.printTeXTable(np.array(texTable).transpose())

    pltLib.startNewPlot("$\\beta$", "$W(1,1)$", "")
    # pltLib.setLogScale(True, False)
    pltLib.plot1DErrPoints(betas,
                           contPlaquettes,
                           label="continous (" +
                           str((2 * sweeps) - thermTime) + " sweeps)")

    pltLib.plot1DErrPoints(betas,
                           partPlaquettes,
                           label="Ikosaeder (" + str(sweeps - thermTime) +
                           " sweeps)",
                           clr="b")

    pltLib.export("export/partIcoBeta.png", width=2)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
