#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/calibStrongCoupling"


def main():
    ex = executor.executor(8)

    latSize = 8
    #Sweeps for Reference, GPU will use 5*sweeps
    sweeps = 1500
    thermTime = 500
    betas = scanBetas = np.linspace(0.1, 10, 100)
    deltas = helpers.getDeltas(betas)

    collectData = False
    collectRefData = False

    # if collectRefData:
    #     ex.recordReferenceData(latSize, betas, deltas, sweeps,
    #                            WORK_DIR + "/ref_data")
    #     ex.runEvaluator(WORK_DIR + "/ref_data", WORK_DIR + "/ref_data.csv",
    #                     thermTime)

    if collectData:
        ex.recordGPUData(latSize, betas, deltas, 4 * sweeps,
                         WORK_DIR + "/gpu_data")

        ex.runEvaluator(WORK_DIR + "/gpu_data", WORK_DIR + "/data.csv",
                        thermTime)

    data = np.loadtxt(WORK_DIR + "/data.csv", dtype=np.float64)
    # refData = np.loadtxt(WORK_DIR + "/ref_data.csv", dtype=np.float64)

    plaquettes = np.array(
        [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])
    # refPlaquettes = np.array([
    #     ufloat(refData[i, 1], refData[i, 2]) for i in range(len(refData[:, 0]))
    # ])

    # pltLib.printTeXTable(
    #     np.array([
    #         betas,
    #         plaquettes,  #refPlaquettes,
    #         np.array([
    #             "{0:.5f}".format(i)
    #             for i in (helpers.strongCouplingExp(betas) + betas / 4)
    #         ])
    #     ]).transpose())

    pltLib.startNewPlot("$\\beta$", "$P - \\beta/4$", "")
    #pltLib.setLogScale(True, False)
    pltLib.plotPoints(betas[:18], [i.n for i in (plaquettes - (betas / 4))[:18]],
                           label="GPU Data(" + str((4 * sweeps) - thermTime) +
                           " sweeps)",s=10)
    # pltLib.plot1DErrPoints(betas,
    #                        refPlaquettes - (betas / 4),
    #                        label="Ref. Data (" + str(sweeps - thermTime) +
    #                        " sweeps)",
    #                        clr="r")
    pltLib.plotFunc(helpers.strongCouplingExp,
                    0.02,
                    1.4,
                    log=True,
                    label="Strong Coupling Expansion")
    pltLib.export("export/calibStrongCoupling.pgf",
                  width=0.54 / 1.2,
                  height=1.2,
                  legend=False)
    pltLib.endPlot()

    pltLib.startNewPlot("$\\beta$", "$P$", "")
    pltLib.plotPoints(betas,
                           [i.n for i in plaquettes],
                           label="GPU Data(" + str((4 * sweeps) - thermTime) +
                           " sweeps)",s=10)
    # pltLib.plotFunc(helpers.strongCouplingExp,
    #                 0.02,
    #                 1.4,
    #                 log=True,
    #                 label="Strong Coupling Expansion")
    pltLib.export("export/referenceDataSet.pgf",
                  width=0.54 / 1.08,
                  height=1.08,
                  legend=False)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
