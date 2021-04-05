#!/usr/bin/env python3

import numpy as np
import math
import os, shutil, subprocess, pathlib
from uncertainties import ufloat
import pltLib

WORK_DIR = "tmp_numericsCheck"


class numericsChecker:
    betas = np.array([])
    latSize = 0
    sweeps = 0

    def __init__(self, beta_min, beta_max, n_beta, latSize, sweeps):
        self.betas = np.logspace(beta_min, beta_max, n_beta)
        self.latSize = latSize
        self.sweeps = sweeps

        # Rounding to two significant digits
        for i in range(len(self.betas)):
            self.betas[i] = round(
                self.betas[i],
                1 - int(math.floor(math.log10(abs(self.betas[i])))))

        # Create Working Directory if not existing
        pathlib.Path(WORK_DIR).mkdir(exist_ok=True)

    def recordSelfMadeData(self):
        pathlib.Path(WORK_DIR + "/self_made").mkdir(exist_ok=True)
        for i in range(len(self.betas)):
            subprocess.call([
                "./../numerics/main", "-m",
                str(self.sweeps), "-b",
                str(self.betas[i]), "-l",
                str(self.latSize), "-o",
                WORK_DIR + "/self_made/data-{}.csv".format(i)
            ],
                            stdout=subprocess.DEVNULL)
            print("({}/{}) Recorded Self Made Data for beta={}".format(
                i + 1, len(self.betas), self.betas[i]))

    def recordReferenceData(self):

        prcs = []

        for i in range(len(self.betas)):
            pathlib.Path(WORK_DIR + "/ref_" + str(i)).mkdir(exist_ok=True)
            prcs.append(
                subprocess.Popen([
                    "./../../../../su2/su2-metropolis", "-X",
                    str(self.latSize), "-Y",
                    str(self.latSize), "-Z",
                    str(self.latSize), "-T",
                    str(self.latSize), "-n",
                    str(self.sweeps), "-b",
                    str(self.betas[i]), "--ndims", "4"
                ],
                                 cwd=WORK_DIR + "/ref_" + str(i),
                                 stdout=subprocess.DEVNULL))
        pathlib.Path(WORK_DIR + "/reference").mkdir(exist_ok=True)
        for i in range(len(prcs)):
            prcs[i].wait()
            print("({}/{}) Recorded Reference Data for beta={}".format(
                i + 1, len(self.betas), self.betas[i]))
            os.rename(WORK_DIR + "/ref_" + str(i) + "/output.metropolis.data",
                      WORK_DIR + "/reference/data-{}.csv".format(i))
            shutil.rmtree(WORK_DIR + "/ref_" + str(i))

    def evaluateData(self, thermoTime):
        beta = np.array(self.betas, dtype=np.float64)
        refData = np.array([ufloat(0, 0) for i in self.betas])
        selfData = np.array([ufloat(0, 0) for i in self.betas])

        for i in range(len(beta)):
            refRaw = np.loadtxt(WORK_DIR + "/reference/data-{}.csv".format(i),
                                dtype=np.float64)[thermoTime:, 1]
            selfRaw = np.loadtxt(WORK_DIR + "/self_made/data-{}.csv".format(i),
                                 dtype=np.float64)[thermoTime:, 1]
            selfData[i] = ufloat(np.mean(selfRaw), np.std(selfRaw))
            refData[i] = ufloat(np.mean(refRaw), np.std(refRaw))

        def highCoupling(b, x):
            return (x / 4)

        def lowCoupling(b, x):
            return 1 - (3 / (4 * x))

        pltLib.startNewPlot("$\\beta$", "$W_{\\textrm{meas}}(1,1)$", "")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(beta, selfData, label="Messpunkte")
        pltLib.plotFitFunc(highCoupling, [], 0.1, 1.5, label="High Coupling Exp.")
        pltLib.plotFitFunc(lowCoupling, [], 5, 100,clr="b", label="Low Coupling Exp.")
        pltLib.export("numCheckPlt1.pgf")
        pltLib.endPlot(False)

        pltLib.startNewPlot("$\\beta$", "$\\Delta W (1,1)$","")
        pltLib.setLogScale(True, False)
        pltLib.plot1DErrPoints(beta, selfData -np.array([i.n for i in refData]),label="$W_{\\textrm{meas}}(1,1) - W_{\\textrm{ref}}(1,1)$")
        pltLib.export("numCheckPlt2.pgf")
        pltLib.endPlot(False)

        pltLib.printTeXTable(np.array([beta, refData, selfData]).transpose())

def main():
    sweeps = 5000
    latSize = 8
    nc = numericsChecker(-1, 2, 40, latSize, sweeps)


    #nc.recordSelfMadeData()

    nc.evaluateData(500)


if __name__ == "__main__":
    main()
