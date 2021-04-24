#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/calibHighCoupling"


def main():
    ex = executor.executor(8)

    latSize = 8
    #Sweeps for Reference, GPU will use 5*sweeps
    sweeps = 1500
    thermTime = 500
    betas = helpers.getRoundedLogSpace(0.05, 1, 24)
    deltas = helpers.getDeltas(betas)

    collectData = False

    if collectData:
        ex.recordReferenceData(latSize, betas, deltas, sweeps,
                               WORK_DIR + "/ref_data")
        ex.runEvaluator(WORK_DIR + "/ref_data", WORK_DIR + "/ref_data.csv",
                        thermTime)

        ex.recordGPUData(latSize, betas, deltas, 4*sweeps, WORK_DIR + "/gpu_data")

        ex.runEvaluator(WORK_DIR + "/gpu_data", WORK_DIR + "/data.csv", thermTime)

    data = np.loadtxt(WORK_DIR + "/data.csv", dtype=np.float64)
    refData = np.loadtxt(WORK_DIR + "/ref_data.csv", dtype=np.float64)

    plaquettes = np.array(
        [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])
    refPlaquettes = np.array([
     ufloat(refData[i, 1], refData[i, 2]) for i in range(len(refData[:, 0]))
    ])

    pltLib.startNewPlot("$\\beta$",
                        "$W_{\\textrm{meas}}(1,1) - \\frac{\\beta}{4}$", "")
    pltLib.setLogScale(True, False)
    pltLib.plot1DErrPoints(betas, plaquettes - (betas / 4), label="GPU")
    pltLib.plot1DErrPoints(betas,
                           refPlaquettes - (betas / 4),
                           label="Reference Data",
                           clr="r")
    pltLib.plotFunc(helpers.highCouplingExp,
                    0.05,
                    0.83,
                    log=True,
                    label="High Coupling Expansion")
    pltLib.export("export/calibHighCoupling.pgf")
    pltLib.endPlot()


if __name__ == "__main__":
    main()
