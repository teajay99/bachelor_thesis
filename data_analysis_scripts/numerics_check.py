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

    def recordSelfMadeData(self, sweeps=0, deltas=[], verbose=False):
        if sweeps == 0:
            sweeps = self.sweeps
        if len(deltas) == 0:
            deltas = self.deltas

        pathlib.Path(WORK_DIR + "/self_made").mkdir(exist_ok=True)
        for i in range(len(self.betas)):
            subprocess.call(
                [
                    "./../numerics/main",  #"--cuda"
                    "-m",
                    str(sweeps),
                    "-b",
                    str(self.betas[i]),
                    "-d",
                    str(deltas[i]),
                    "-l",
                    str(self.latSize),
                    "-o",
                    WORK_DIR + "/self_made/data-{}.csv".format(i)
                ],
                stdout=subprocess.DEVNULL)
            if verbose:
                print("({}/{}) Recorded Self Made Data for beta={}".format(
                    i + 1, len(self.betas), self.betas[i]))

    def recordReferenceData(self):

        threads = 8
        runs = (len(self.betas) + 8 - 1) // threads

        for j in range(runs):
            prcs = []
            for t in range(threads):
                i = (j * threads) + t
                if i < len(self.betas):
                    pathlib.Path(WORK_DIR + "/ref_" +
                                 str(i)).mkdir(exist_ok=True)
                    prcs.append(
                        subprocess.Popen([
                            "./../../../../su2/su2-metropolis", "-X",
                            str(self.latSize), "-Y",
                            str(self.latSize), "-Z",
                            str(self.latSize), "-T",
                            str(self.latSize), "-n",
                            str(self.sweeps), "-d",
                            str(self.deltas[i]), "-b",
                            str(self.betas[i]), "--ndims", "4"
                        ],
                                         cwd=WORK_DIR + "/ref_" + str(i),
                                         stdout=subprocess.PIPE))

            pathlib.Path(WORK_DIR + "/reference").mkdir(exist_ok=True)
            for t in range(len(prcs)):
                i = (j * threads) + t
                if i < len(self.betas):
                    #prcs[t].wait()
                    hrate = str(prcs[t].communicate()[0]).split("\\n")[-2][19:]
                    print(
                        "({}/{}) Recorded Reference Data for beta={} with hitrate {}"
                        .format(i + 1, len(self.betas), self.betas[i], hrate))
                    os.rename(
                        WORK_DIR + "/ref_" + str(i) +
                        "/output.metropolis.data",
                        WORK_DIR + "/reference/data-{}.csv".format(i))
                    shutil.rmtree(WORK_DIR + "/ref_" + str(i))

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
            out = (-(112 / 1024) + (128524 / 1244160) -
                   (211991 / 8709120)) * (x**11)
            out = ((16 / 256) - (196 / 4608) + (1001 / 172800)) * (x**9)
            out += (-(4 / 96) + (29 / 1440)) * (x**7)
            out += -((1 / 48) * (x**3)) + (((4 / 96) - (5 / 288)) * (x**5))
            return out

        def lowCoupling(b, x):
            return 1 - (3 / (4 * x))

        hc_index = self.sweeps
        for i in range(len(self.betas)):
            if self.betas[i] > 1:
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
                           0.1,
                           1,
                           label="High Coupling Exp.",
                           clr="b")
        pltLib.export("numCheckPlt1.pgf")
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
    sweeps = 600
    latSize = 4
    thermTime = 350
    nc = numericsChecker(-1, 2, 40, latSize, sweeps, thermTime)

    #nc.calibrateDeltas()
    #nc.fitDeltas()

    #nc.recordReferenceData()
    nc.recordSelfMadeData(verbose=True)
    nc.evaluateData()


if __name__ == "__main__":
    main()
