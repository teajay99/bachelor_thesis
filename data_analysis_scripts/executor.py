import os, shutil, subprocess, pathlib
import numpy as np


class executor:
    threads = 0

    def __init__(self, threads):
        self.threads = threads

    def fixParameters(self, parameters):
        runCount = 1
        for j in range(len(parameters)):
            if isinstance(parameters[j], list) or isinstance(
                    parameters[j], np.ndarray):
                runCount = len(parameters[j])

        out = [[] for i in parameters]

        for j in range(len(parameters)):
            if isinstance(parameters[j], list) or isinstance(
                    parameters[j], np.ndarray):
                out[j] = parameters[j]
            else:
                out[j] = [parameters[j] for i in range(runCount)]
        return out

    def recordGPUData(self,
                      latSize,
                      betas,
                      deltas,
                      sweeps,
                      dataDir,
                      verbose=True,
                      partition=None):

        latSize, betas, deltas, sweeps, partition = self.fixParameters(
            [latSize, betas, deltas, sweeps, partition])

        if os.path.exists(dataDir):
            shutil.rmtree(dataDir)

        pathlib.Path(dataDir).mkdir(exist_ok=True, parents=True)
        for i in range(len(betas)):
            callList = [
                "./../numerics/main", "--cuda", "-m",
                str(sweeps[i]), "-b",
                str(betas[i]), "-d",
                str(deltas[i]), "-l",
                str(latSize[i]), "-o", dataDir + "/data-{}.csv".format(i)
            ]
            if partition[i] != None:
                callList.append("-p")
                callList.append(partition[i])

            subprocess.check_call(callList, stdout=subprocess.DEVNULL)
            if verbose:
                print("Collecting GPU Dataset ({}/{}).".format(
                    i + 1, len(betas)))

    def recordCPUData(self,
                      latSize,
                      betas,
                      deltas,
                      sweeps,
                      dataDir,
                      verbose=True,
                      partition=None):

        latSize, betas, deltas, sweeps, partition = self.fixParameters(
            [latSize, betas, deltas, sweeps, partition])

        if os.path.exists(dataDir):
            shutil.rmtree(dataDir)
        pathlib.Path(dataDir).mkdir(exist_ok=True, parents=True)

        runs = (len(betas) + self.threads - 1) // self.threads
        for j in range(runs):
            prcs = []
            for t in range(self.threads):
                i = (j * self.threads) + t
                if i < len(betas):

                    callList = [
                        "./../numerics/main", "-m",
                        str(sweeps[i]), "-b",
                        str(betas[i]), "-d",
                        str(deltas[i]), "-l",
                        str(latSize[i]), "-o",
                        dataDir + "/data-{}.csv".format(i)
                    ]

                    if partition[i] != None:
                        callList.append("-p")
                        callList.append(partition[i])

                    prcs.append(
                        subprocess.Popen(callList, stdout=subprocess.DEVNULL))
            for t in range(self.threads):
                i = (j * self.threads) + t
                if i < len(betas):
                    prcs[t].wait()
                    print("Collecting CPU Dataset ({}/{}).".format(
                        i + 1, len(betas)))

    def recordReferenceData(self,
                            latSize,
                            betas,
                            deltas,
                            sweeps,
                            dataDir,
                            verbose=True):

        latSize, betas, deltas, sweeps = self.fixParameters(
            [latSize, betas, deltas, sweeps])

        if os.path.exists(dataDir):
            shutil.rmtree(dataDir)
        pathlib.Path(dataDir).mkdir(exist_ok=True, parents=True)

        runs = (len(betas) + self.threads - 1) // self.threads

        for j in range(runs):
            prcs = []
            for t in range(self.threads):
                i = (j * self.threads) + t
                if i < len(betas):
                    pathlib.Path(dataDir + "/ref_" +
                                 str(i)).mkdir(exist_ok=True)
                    prcs.append(
                        subprocess.Popen([
                            "/home/timo/Dropbox/Dokumente/Studium/Bachelorarbeit/git/su2/su2-metropolis", "-X",
                            str(latSize[i]), "-Y",
                            str(latSize[i]), "-Z",
                            str(latSize[i]), "-T",
                            str(latSize[i]), "-n",
                            str(sweeps[i]), "-d",
                            str(deltas[i]), "-b",
                            str(betas[i]), "--ndims", "4"
                        ],
                                         cwd=dataDir + "/ref_" + str(i),
                                         stdout=subprocess.DEVNULL))

            for t in range(len(prcs)):
                i = (j * self.threads) + t
                if i < len(betas):
                    prcs[t].wait()
                    os.rename(
                        dataDir + "/ref_" + str(i) + "/output.metropolis.data",
                        dataDir + "/data-{}.csv".format(i))
                    shutil.rmtree(dataDir + "/ref_" + str(i))

                    if verbose:
                        print("Collecting Reference Dataset ({}/{}).".format(
                            i + 1, len(betas)))

    def getPartition(self):
        1

    def runEvaluator(self, dataDir, outputFile, thermTime):

        fileCounter = 0

        for i in os.listdir(dataDir):
            if i[0:5] == "data-" and i[-4:] == ".csv":
                fileCounter += 1

        subprocess.check_call([
            "./evaluator.R", dataDir,
            str(thermTime),
            str(fileCounter), outputFile
        ],
                              stdout=subprocess.DEVNULL)
