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


#
#  Strong Coupling Expansion from
#  https://www.researchgate.net/publication/1739772_Series_expansions_of_the_density_of_states_in_SU2_lattice_gauge_theory
#
def strongCouplingExp(x, d=4):
    out = -(1 / 96) * (x**3)
    out += (7 / 1536) * (x**5)
    out += -(31 / 23040) * (x**7)
    out += (4451 / 8847360) * (x**9)
    out += -(264883 / 1486356480) * (x**11)
    out += (403651 / 5945425920) * (x**13)
    out += -(1826017873 / 68491306598400) * (x**15)

    return out


#
#  Weak Coupling Expansion for 6^4 lattice from
#  https://www.researchgate.net/publication/1739772_Series_expansions_of_the_density_of_states_in_SU2_lattice_gauge_theory
#


def weakCouplingExp6(x):
    params = [
        0.7498, 0.1511, 0.1427, 0.1747, 0.2435, 0.368, 0.5884, 0.98, 1.6839,
        2.9652, 5.326, 9.7234, 17.995, 33.690, 63.702
    ]

    out = 0

    for i in range(len(params)):
        out += (params[i] / (x**(i + 1)))

    return 1 - out


def inverseWeakCouplingExp6(x):
    return weakCouplingExp6(1/x)
