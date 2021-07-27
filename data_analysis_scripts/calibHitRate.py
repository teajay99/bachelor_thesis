#!/usr/bin/env python3

import executor
import pltLib
import numpy as np
import helpers
from pathlib import Path

from uncertainties import ufloat

WORK_DIR = "tmpData/calibHitRate"


def calibrateDeltas(latSize, betas, thermTime, sweeps, iters):
    a = [0.001 for k in range(len(betas))]
    b = [1 for k in range(len(betas))]
    c = [0 for k in range(len(betas))]
    fa = [0 for k in range(len(betas))]
    fb = [0 for k in range(len(betas))]
    fc = [0 for k in range(len(betas))]

    ex = executor.executor(8)

    for j in range(iters):
        ex.recordGPUData(latSize,
                         betas,
                         a,
                         sweeps + thermTime,
                         WORK_DIR + "/data",
                         verbose=False)
        for i in range(len(betas)):
            fa[i] = np.mean(
                np.loadtxt(WORK_DIR + "/data/data-{}.csv".format(i),
                           dtype=np.float64)[thermTime:, 2]) - 0.5

        ex.recordGPUData(latSize,
                         betas,
                         b,
                         sweeps + thermTime,
                         WORK_DIR + "/data",
                         verbose=False)
        for i in range(len(betas)):
            fb[i] = np.mean(
                np.loadtxt(WORK_DIR + "/data/data-{}.csv".format(i),
                           dtype=np.float64)[thermTime:, 2]) - 0.5
            if fb[i] != fa[i]:
                c[i] = a[i] - ((b[i] - a[i]) / (fb[i] - fa[i])) * fa[i]
                c[i] = min(c[i], 1)
        ex.recordGPUData(latSize,
                         betas,
                         c,
                         sweeps + thermTime,
                         WORK_DIR + "/data",
                         verbose=False)
        for i in range(len(betas)):
            fc[i] = np.mean(
                np.loadtxt(WORK_DIR + "/data/data-{}.csv".format(i),
                           dtype=np.float64)[thermTime:, 2]) - 0.5
            if (fa[i] > 0) == (fc[i] > 0):
                a[i] = c[i]
            else:
                b[i] = c[i]
        print(str(j + 1) + "/" + str(iters) + " Iterations done")
    print(betas, c, fc)
    outFile = open(WORK_DIR + "/delta_config.csv", "w")
    for i in range(len(betas)):
        outFile.write(
            str(betas[i]) + "\t" + str(c[i]) + "\t" + str(fc[i]) + "\n")
    outFile.close()


def fitDeltas():
    deltas = np.loadtxt(WORK_DIR + "/delta_config.csv")[:, 1]
    betas = np.loadtxt(WORK_DIR + "/delta_config.csv")[:, 0]
    hitRates = np.loadtxt(WORK_DIR + "/delta_config.csv")[:, 2]

    def fitFunc(p, x):
        a, b, c, d, e, f = p
        out = a / (x - e) + b / ((x - e)**2) + c / ((x - e)**3) + d / (
            (x - e)**4) + f
        return np.minimum(out, 1)

    p, perr, crs = pltLib.makeFit(
        np.array([ufloat(i, 0.001 * i) for i in betas]),
        np.array([ufloat(i, 0.01 * i) for i in deltas]), fitFunc, [
            0.3207785, -0.30052893, 0.3690416, -0.00391372, 0.08930275,
            0.0245088
        ])

    print(p, perr, crs)

    with open('tmpData/delta_fit_params.csv', 'w') as f:
        for item in p:
            f.write("%s\n" % item)

    pltLib.startNewPlot("$\\beta$", "", "")
    pltLib.setLogScale(True, False)
    pltLib.plotPoints(betas, deltas, clr="k", label="$\\delta$", marker="x")
    pltLib.plotPoints(betas,
                      hitRates + 0.5,
                      clr="b",
                      label="$\\frac{N_{\\textrm{acc}}}{N_{\\textrm{hit}}}$")
    pltLib.plotFitFunc(fitFunc,
                       p,
                       np.min(betas),
                       np.max(betas),
                       log=True,
                       label="$\\textrm{Fit}$")
    pltLib.export("export/calibHitRate.pgf",
                  width=0.54,
                  legendLoc="lower left")
    pltLib.endPlot()


def main():

    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    sweeps = 50
    latSize = 6
    thermTime = 500
    iterations = 30

    betas = helpers.getRoundedLogSpace(0.1, 100, 25)

    calibrateDeltas(latSize, betas, thermTime, sweeps, iterations)
    fitDeltas()


if __name__ == "__main__":
    main()
