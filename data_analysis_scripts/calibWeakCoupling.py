#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat

from pathlib import Path

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/calibWeakCoupling"


def main():
    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    ex = executor.executor(8)

    latSize = 6
    #Sweeps for Reference, GPU will use 5*sweeps
    sweeps = 10000
    thermTime = 5000
    betas = 1 / np.linspace(0.1, 0.5, 41)
    deltas = helpers.getDeltas(betas)

    collectData = False

    if collectData:
        ex.recordGPUData(latSize, betas, deltas, sweeps + thermTime,
                         WORK_DIR + "/gpu_data")

        ex.runEvaluator(WORK_DIR + "/gpu_data", WORK_DIR + "/data.csv",
                        thermTime)

    data = np.loadtxt(WORK_DIR + "/data.csv", dtype=np.float64)

    plaquettes = np.array(
        [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])

    pltLib.printTeXTable(
        np.array([
            betas, plaquettes,
            np.array([
                "{0:.6f}".format(i) for i in (helpers.weakCouplingExp6(betas))
            ])
        ]).transpose())

    pltLib.startNewPlot("$1/\\beta$", "$P$", "")
    #pltLib.setLogScale(True, False)
    pltLib.plotPoints(1 / betas, [i.n for i in plaquettes],
                      label="GPU Data(" + str((4 * sweeps) - thermTime) +
                      " sweeps)",
                      s=10)
    # pltLib.plot1DErrPoints(1/betas,
    #                        refPlaquettes,
    #                        label="Ref. Data (" + str(sweeps - thermTime) +
    #                        " sweeps)",
    #                        clr="r")
    pltLib.plotFunc(helpers.inverseWeakCouplingExp6,
                    0.08,
                    0.52,
                    log=True,
                    label="Weak Coupling Expansion")
    pltLib.export("export/calibWeakCoupling.pgf",
                  width=0.54 / 1.1,
                  height=1.1,
                  legend=False)
    pltLib.endPlot()


if __name__ == "__main__":
    main()
