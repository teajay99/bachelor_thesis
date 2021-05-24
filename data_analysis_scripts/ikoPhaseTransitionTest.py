#!/usr/bin/env python3

import numpy as np
from uncertainties import ufloat
from partitions import fibonacci

import executor
import pltLib
import helpers

WORK_DIR = "tmpData/ikoPhaseTransitionTest2"

clrs = [
    "black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]


def findIkoPhaseTransition(betas,
                           sweeps,
                           collectData=False,
                           exportFile="export.png"):
    ex = executor.executor(8)

    latSize = 8

    hits = 1
    multiSweep = 100
    deltas = 0.21

    betas = np.concatenate((betas, betas))
    betas.sort()

    cold = np.array([i % 2 == 0 for i in range(len(betas))])

    if collectData:
        ex.recordGPUData(latSize,
                         betas,
                         deltas,
                         sweeps,
                         WORK_DIR,
                         hits=hits,
                         multiSweep=multiSweep,
                         cold=cold,
                         partition="--partition-iko")

    pltLib.startNewPlot("$\\textrm{Iterationen}$", "$1-W(1,1)$", "")
    pltLib.setLogScale(True, False)
    pltLib.ax.set_ylim(0, 0.15)

    for i in range(len(betas)):
        data = np.genfromtxt(WORK_DIR + "/data-" + str(i) + ".csv")
        label = None
        if i % 2 == 0:
            label = "$\\beta = {:.2f}$".format(betas[i])
        pltLib.plotLine(hits * multiSweep * data[:, 0],
                        1 - data[:, 1],
                        clr=clrs[i // 2],
                        label=label)

    pltLib.export(exportFile, width=2)
    pltLib.endPlot()


def main(collectData=False):

    #betas = np.linspace(5.96, 6.06, 6)
    #print(betas)
    #findIkoPhaseTransition(betas, 2500, True, "export/paper.png")

    betas = np.linspace(5.5, 5.9, 5)
    print(betas)
    findIkoPhaseTransition(betas, 10000, True, "export/newStuffTwo.png")


if __name__ == "__main__":
    main()
