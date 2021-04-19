#!/usr/bin/env python3

import numpy as np
import math
import os, shutil, subprocess, pathlib
from uncertainties import ufloat
import pltLib

WORK_DIR = "tmp_numericsCheck"


class numericsChecker:
    betas = np.array([])
    deltas = np.array([])
    latSize = 0
    sweeps = 0
    thermTime = 0

    def __init__(self, beta_min, beta_max, n_beta, latSize, sweeps, thermTime):
        self.betas = np.logspace(beta_min, beta_max, n_beta)

        def getDeltas(x):
            a, b, c, d, e, f = 0.3207785, -0.30052893, 0.3690416, -0.00391372, 0.08930275, 0.0245088
            out = a / (x - e) + b / ((x - e)**2) + c / ((x - e)**3) + d / (
                (x - e)**4) + f
            return np.minimum(out, 1)

        self.deltas = getDeltas(self.betas)

        self.latSize = latSize
        self.sweeps = sweeps
        self.thermTime = thermTime

        # Rounding to two significant digits
        for i in range(len(self.betas)):
            self.betas[i] = round(
                self.betas[i],
                1 - int(math.floor(math.log10(abs(self.betas[i])))))

        # Create Working Directory if not existing
        pathlib.Path(WORK_DIR).mkdir(exist_ok=True)

    def calibrateDeltas(self):
        a = [0.001 for k in range(len(self.betas))]
        b = [1 for k in range(len(self.betas))]
        c = [0 for k in range(len(self.betas))]
        fa = [0 for k in range(len(self.betas))]
        fb = [0 for k in range(len(self.betas))]
        fc = [0 for k in range(len(self.betas))]

        for j in range(15):
            self.recordSelfMadeData(self.thermTime + 20, a)
            for i in range(len(self.betas)):
                fa[i] = np.mean(
                    np.loadtxt(WORK_DIR + "/self_made/data-{}.csv".format(i),
                               dtype=np.float64)[self.thermTime:, 2]) - 0.5

            self.recordSelfMadeData(self.thermTime + 20, b)
            for i in range(len(self.betas)):
                fb[i] = np.mean(
                    np.loadtxt(WORK_DIR + "/self_made/data-{}.csv".format(i),
                               dtype=np.float64)[self.thermTime:, 2]) - 0.5
                if fb[i] != fa[i]:
                    c[i] = a[i] - ((b[i] - a[i]) / (fb[i] - fa[i])) * fa[i]
                    c[i] = min(c[i], 1)
            self.recordSelfMadeData(self.thermTime + 20, c)
            for i in range(len(self.betas)):
                fc[i] = np.mean(
                    np.loadtxt(WORK_DIR + "/self_made/data-{}.csv".format(i),
                               dtype=np.float64)[self.thermTime:, 2]) - 0.5
                if (fa[i] > 0) == (fc[i] > 0):
                    a[i] = c[i]
                else:
                    b[i] = c[i]
            print(str(j + 1) + " Iterations done")
        print(self.betas, c, fc)
        outFile = open(WORK_DIR + "/delta_config.csv", "w")
        for i in range(len(self.betas)):
            outFile.write(str(c[i]) + "\n")
        outFile.close()

    def fitDeltas(self):
        deltas = np.loadtxt(WORK_DIR + "/delta_config.csv")

        def fitFunc(p, x):
            a, b, c, d, e, f = p
            out = a / (x - e) + b / ((x - e)**2) + c / ((x - e)**3) + d / (
                (x - e)**4) + f
            return np.minimum(out, 1)

        p, perr, crs = pltLib.makeFit(
            np.array([ufloat(i, 0.001 * i) for i in self.betas]),
            np.array([ufloat(i, 0.01 * i) for i in deltas]), fitFunc, [
                0.3207785, -0.30052893, 0.3690416, -0.00391372, 0.08930275,
                0.0245088
            ])

        print(p, perr, crs)

        pltLib.startNewPlot("$\\beta$", "$\\delta$", "")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(self.betas,
                               np.array([ufloat(i, 0.01 * i) for i in deltas]))
        pltLib.plotFitFunc(fitFunc, p, 0.1, 10)
        pltLib.endPlot()



    def evaluateData(self):
        beta = np.array(self.betas, dtype=np.float64)
        refData = np.array([ufloat(0, 0) for i in self.betas])
        selfData = np.array([ufloat(0, 0) for i in self.betas])

        for i in range(len(beta)):
            refRaw = np.loadtxt(WORK_DIR + "/reference/data-{}.csv".format(i),
                                dtype=np.float64)[self.thermTime:, 1]
            selfRaw = np.loadtxt(WORK_DIR + "/self_made/data-{}.csv".format(i),
                                 dtype=np.float64)[self.thermTime:, 1]
            selfData[i] = ufloat(np.mean(selfRaw), np.std(selfRaw))
            refData[i] = ufloat(np.mean(refRaw), np.std(refRaw))

        def highCoupling(b, x):
            out = (-(3008 / 8192) + (112494928 / 212336640) -
                   (388403644 / 1486356480) +
                   (1474972157 / 33443020800)) * (x**15)
            out += ((320 / 2048) - (688 / 4096) + (21364 / 368640) -
                    (264497 / 40642560)) * (x**13)
            out += (-(112 / 1024) + (128524 / 1244160) -
                    (211991 / 8709120)) * (x**11)
            out += ((16 / 256) - (196 / 4608) + (1001 / 172800)) * (x**9)
            out += (-(4 / 96) + (29 / 1440)) * (x**7)
            out += (((4 / 96) - (5 / 288)) * (x**5))
            out += -((1 / 48) * (x**3))
            return out

        def lowCoupling(b, x):
            return 1 - (3 / (4 * x))

        hc_index = self.sweeps
        for i in range(len(self.betas)):
            if self.betas[i] > 0.85:
                hc_index = i
                break

        pltLib.startNewPlot("$\\beta$", "$W_{\\textrm{meas}}(1,1)$", "")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(beta[:hc_index],
                               selfData[:hc_index] - (beta[:hc_index] / 4),
                               label="Messpunkte")
        pltLib.plot1DErrPoints(beta[:hc_index],
                               refData[:hc_index] - (beta[:hc_index] / 4),
                               label="Referenzpunkte",
                               clr="r")
        pltLib.plotFitFunc(highCoupling, [],
                           0.01,
                           0.85,
                           label="High Coupling Exp.",
                           clr="b")
        pltLib.export("numCheckPlt0.pgf")
        pltLib.endPlot()

        pltLib.startNewPlot("$\\beta$", "$W_{\\textrm{meas}}(1,1)$", "")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(beta, selfData, label="Messpunkte")
        pltLib.plot1DErrPoints(beta, refData, label="Referenzpunkte", clr="r")
        pltLib.plotFitFunc(highCoupling, [],
                           0.1,
                           1,
                           label="High Coupling Exp.",
                           clr="b")
        pltLib.export("numCheckPlt1.pgf")
        pltLib.endPlot()

        pltLib.startNewPlot("$\\beta$", "$\\Delta W (1,1)$", "")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(
            beta,
            selfData - np.array([i.n for i in refData]),
            label="$W_{\\textrm{meas}}(1,1) - W_{\\textrm{ref}}(1,1)$")
        pltLib.export("numCheckPlt2.pgf")
        pltLib.endPlot()

        pltLib.printTeXTable(
            np.array([beta, refData, selfData, self.deltas]).transpose())


def main():
    sweeps = 1000
    latSize = 4
    thermTime = 500
    nc = numericsChecker(-2, 0, 30, latSize, sweeps, thermTime)

    #nc.calibrateDeltas()
    #nc.fitDeltas()

    #nc.recordReferenceData()
    #nc.recordSelfMadeDataGPU(verbose=True)
    nc.evaluateData()


if __name__ == "__main__":
    main()
