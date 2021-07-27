#!/usr/bin/env python3

import helpers
import executor
import numpy as np
import pltLib
import os, shutil, subprocess, pathlib
from partitions import fibonacci
from uncertainties import ufloat
from pathlib import Path

WORK_DIR = "tmpData/partitionEvaluation"

CLR = [
    "lightcoral", "firebrick", "limegreen", "green", "deepskyblue",
    "steelblue", "violet", "darkviolet"
]


class partitionTest:
    partition = ""
    partitionName = ""
    partitionOpt = ""
    latSize = 10

    folderName = ""

    def __init__(self, partition, partitionName, partitionOpt=None):
        self.partition = partition
        self.partitionName = partitionName
        self.partitionOpt = partitionOpt

        self.folderName = (partition)
        if not (partitionOpt is None):
            self.folderName += partitionOpt.split("/")[-1]
        self.folderName = self.folderName.replace("-", "")

        if partition == "fibonacci":
            fibonacci.generateLattice(
                int(partitionOpt), WORK_DIR + "/" + folderName + "-part.csv")
            self.partitionOpt = WORK_DIR + "/" + folderName + "-part.csv"

    def measure(self, betas, deltas, thermTime, cold, collectData=True):
        ex = executor.executor(8)
        path = WORK_DIR + "/" + self.folderName
        if cold:
            path += "_cold"
        if collectData:
            ex.recordGPUData(8,
                             betas,
                             deltas,
                             7000,
                             path,
                             partition=self.partition,
                             partitionOpt=self.partitionOpt,
                             cold=cold)
            ex.runEvaluator(path, path + ".csv", thermTime)
        data = np.loadtxt(path + ".csv", dtype=np.float64)
        plaquettes = np.array(
            [ufloat(data[i, 1], data[i, 2]) for i in range(len(data[:, 0]))])
        return plaquettes


def main():

    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)

    collectData = True
    collectRefData = True
    #Record Reference Data
    referenceIterations = 100000
    thermTime = 4000
    scanBetas = np.linspace(0.1, 10, 100)
    deltas = helpers.getDeltas(scanBetas)
    #print(deltas)

    ex = executor.executor(8)

    if collectRefData:
        ex.recordGPUData(8, scanBetas, deltas, referenceIterations + thermTime,
                         WORK_DIR + "/ref_data")
        ex.runEvaluator(WORK_DIR + "/ref_data", WORK_DIR + "/ref_data.csv",
                        thermTime)

    refData = np.loadtxt(WORK_DIR + "/ref_data.csv", dtype=np.float64)
    refPlaquettes = np.array([
        ufloat(refData[i, 1], refData[i, 2]) for i in range(len(refData[:, 0]))
    ])

    #Generate Fibonacci Lattices
    fibCounts = [8, 16, 32, 64, 128, 256, 512]

    partitions = [[
        partitionTest("--partition-tet", "$\\overline{T}$"),
        partitionTest("--partition-oct", "$\\overline{O}$"),
        partitionTest("--partition-ico", "$\\overline{I}$")
    ],
                  [
                      partitionTest("--partition-volley", "$V_1$", "1"),
                      partitionTest("--partition-volley", "$V_2$", "2"),
                      partitionTest("--partition-volley", "$V_3$", "3"),
                      partitionTest("--partition-volley", "$V_4$", "4")
                  ],
                  [
                      partitionTest("--partition-c5", "$C_5$"),
                      partitionTest("--partition-c16", "$C_{16}$"),
                      partitionTest("--partition-c8", "$C_8$"),
                      partitionTest("--partition-c120", "$C_{120}$")
                  ]]

    fibParts = []
    for c in fibCounts:
        partPath = WORK_DIR + "/fibList" + str(c) + ".csv"

        fibParts.append(
            partitionTest("--partition-list", "$F_{" + str(c) + "}$",
                          partPath))
        if collectData:
            fibonacci.generateLattice(c, partPath)
    partitions.append(fibParts[:4])
    partitions.append(fibParts[4:])

    titles = [
        "Subgroups", "Volleyball", "Regular Polytopes", "Fibonacci-I",
        "Fibonacci-II"
    ]

    for i in range(len(partitions)):
        pltLib.startNewPlot("$\\beta$", "$P-P_{\\textrm{ref}}$", "")
        pltLib.setSymLogScale(False, True, ythresh=1e-3)
        pltLib.ax.yaxis.grid(which='minor', alpha=0.3)

        tableData = [scanBetas]

        for j in range(len(partitions[i])):
            coldData = partitions[i][j].measure(scanBetas, deltas, thermTime,
                                                True,
                                                collectData) - refPlaquettes
            hotData = partitions[i][j].measure(scanBetas, deltas, thermTime,
                                               False,
                                               collectData) - refPlaquettes
            tableData.extend([np.array(coldData), np.array(hotData)])
            pltLib.plot1DErrPoints(scanBetas,
                                   coldData,
                                   clr=CLR[2 * j],
                                   label=partitions[i][j].partitionName +
                                   " - $\\textrm{cold}$")
            pltLib.plot1DErrPoints(scanBetas,
                                   hotData,
                                   clr=CLR[2 * j + 1],
                                   label=partitions[i][j].partitionName +
                                   " - $\\textrm{hot}$")

            pltLib.plotLine(scanBetas, [i.n for i in coldData],
                            clr=CLR[2 * j],
                            alpha=1)
            pltLib.plotLine(scanBetas, [i.n for i in hotData],
                            clr=CLR[2 * j + 1],
                            alpha=1)

        pltLib.export("export/" + titles[i].replace(" ", "") + ".pgf",
                      width=1.08,
                      height=0.9)
        pltLib.endPlot()

        pltLib.printTeXTable(np.stack(tableData).transpose())


if __name__ == "__main__":
    main()
