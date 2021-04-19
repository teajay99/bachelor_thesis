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


def highCouplingExp(x):
    out = (-(3008 / 8192) + (112494928 / 212336640) -
           (388403644 / 1486356480) + (1474972157 / 33443020800)) * (x**15)
    out += ((320 / 2048) - (688 / 4096) + (21364 / 368640) -
            (264497 / 40642560)) * (x**13)
    out += (-(112 / 1024) + (128524 / 1244160) - (211991 / 8709120)) * (x**11)
    out += ((16 / 256) - (196 / 4608) + (1001 / 172800)) * (x**9)
    out += (-(4 / 96) + (29 / 1440)) * (x**7)
    out += (((4 / 96) - (5 / 288)) * (x**5))
    out += -((1 / 48) * (x**3))
    return out
