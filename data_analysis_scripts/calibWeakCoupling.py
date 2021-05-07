#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/calibWeakCoupling"


def main():
    ex = executor.executor(8)

    latSize = 6
    #Sweeps for Reference, GPU will use 5*sweeps
    sweeps = 1500
    thermTime = 500
    betas = helpers.getRoundedLogSpace(2.5, 25, 16)
    deltas = helpers.getDeltas(betas)

    collectData = True
    collectRefData = False

    if collectRefData:
        ex.recordReferenceData(latSize, betas, deltas, sweeps,
                               WORK_DIR + "/ref_data")
        ex.runEvaluator(WORK_DIR + "/ref_data", WORK_DIR + "/ref_data.csv",
                        thermTime)

    if collectData:
        ex.recordGPUData(latSize, betas, deltas, 4 * sweeps,
                         WORK_DIR + "/gpu_data")

        ex.runEvaluator(WORK_DIR + "/gpu_data", WORK_DIR + "/data.csv",
                        thermTime)

    data = np.loadtxt(WORK_DIR + "/data.csv", dtype=np.float64)
    refData = np.loadtxt(WORK_DIR + "/ref_data.csv", dtype=np.float64)

    plaquettes = np.array(
        [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])
    refPlaquettes = np.array([
        ufloat(refData[i, 1], refData[i, 2]) for i in range(len(refData[:, 0]))
    ])


    pltLib.printTeXTable(
        np.array([
            betas, plaquettes, refPlaquettes,
            np.array([
                "{0:.6f}".format(i)
                for i in (helpers.weakCouplingExp6(betas))
            ])
        ]).transpose())

    pltLib.startNewPlot("$\\beta$", "$W(1,1)$", "")
    pltLib.setLogScale(True, False)
    pltLib.plot1DErrPoints(1/betas,
                           plaquettes,
                           label="GPU Data(" + str((4 * sweeps) - thermTime) +
                           " sweeps)")
    pltLib.plot1DErrPoints(1/betas,
                           refPlaquettes,
                           label="Ref. Data (" + str(sweeps - thermTime) +
                           " sweeps)",
                           clr="r")
    pltLib.plotFunc(helpers.inverseWeakCouplingExp6,
                    1/25,
                    1/2.5,
                    log=True,
                    label="Weak Coupling Expansion")
    pltLib.export("export/calibWeakCoupling.pgf", width=0.8)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
