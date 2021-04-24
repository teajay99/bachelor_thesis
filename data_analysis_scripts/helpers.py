import numpy as np
import math

DELTA_CALIB_FILE = "tmpData/delta_fit_params.csv"


def getRoundedLogSpace(min, max, count, digits=2):
    var = np.logspace(math.log10(min), math.log10(max), count)
    for i in range(len(var)):
        var[i] = round(var[i],
                       digits - 1 - int(math.floor(math.log10(abs(var[i])))))
    return var


def deltaFitFunc(p, beta):
    a, b, c, d, e, f = p
    out = a / (beta - e) + b / ((beta - e)**2) + c / ((beta - e)**3) + d / (
        (beta - e)**4) + f
    return np.minimum(out, 1)


def getDeltas(beta):
    p = np.loadtxt(DELTA_CALIB_FILE)
    return deltaFitFunc(p, beta)


def highCouplingExp(x, d=4):
    out = (-((47 * (d**3)) / 8192) + ((7030933 * (d**2)) / 212336640) -
           ((97100911 * d) / 1486356480) +
           (1474972157 / 33443020800)) * (x**15)
    out += (((5 * (d**3)) / 2048) - ((43 * (d**2)) / 4096) +
            ((5341 * d) / 368640) - (264497 / 40642560)) * (x**13)
    out += (-((7 * (d**2)) / 1024) + ((32131 * d) / 1244160) -
            (211991 / 8709120)) * (x**11)
    out += (((d**2) / 256) - ((49 * d) / 4608) + (1001 / 172800)) * (x**9)
    out += (-(d / 96) + (29 / 1440)) * (x**7)
    out += (((d / 96) - (5 / 288)) * (x**5))
    out += -((1 / 48) * (x**3))
    return out
