import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
from scipy.optimize import curve_fit
from scipy import odr
from scipy.stats import chisquare
import matplotlib as mpl
from uncertainties import ufloat
from scipy.optimize import curve_fit
from lmfit import minimize

#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")

#from uncertainties.umath import *

fig, ax = (0, 0)


def getCRS(data, model, yNoise, ddof):
    CR = np.sum(((data - model)**2) / (yNoise**2))
    return CR / (data.size - ddof)


def makeFitAbs(x, y, xNoise, yNoise, fitFunc, b0):
    model = odr.Model(fitFunc)
    data = odr.RealData(x, y, sx=xNoise, sy=yNoise)
    od = odr.ODR(data, model, beta0=b0, maxit=1000)

    # Run the regression.
    out = od.run()
    CRS = getCRS(y, fitFunc(out.beta, x), yNoise, len(b0))

    return (out.beta, out.sd_beta, CRS)


def makeFit(x, y, fitFunc, b0):
    return makeFitAbs(np.array([i.n for i in x]), np.array([i.n for i in y]),
                      np.array([i.std_dev for i in x]),
                      np.array([i.std_dev for i in y]), fitFunc, b0)


def makeFit1DErr(xin, yin, func, p0=None):
    x = xin
    y = np.array([i.n for i in yin])
    yerr = np.array([i.std_dev for i in yin])
    popt, pcov = curve_fit(func, x, y, sigma=yerr, p0=p0)

    #Calculate CRS
    CR, CRS = chisquare(y, func(x, *popt), ddof=len(popt))

    perr = np.sqrt(np.diag(pcov))

    return (popt, perr, CRS)


def makeParamFit(x_in, y_in, fitFunc, params):
    def residual(p, x, y):
        model = fitFunc(p, np.array([i.n for i in x]))
        data = np.array([i.n for i in y])
        std_dev_data = np.array([i.std_dev for i in y])
        return (data - model) / (std_dev_data)

    return minimize(residual,
                    params,
                    args=(x_in, y_in),
                    method="least_squares")


def makeParamFit2D(x_in, y_in, fitFunc, params):
    def residual(p, x, y):
        model = fitFunc(p, x)
        data = np.array([i.n for i in y])
        std_dev_data = np.array([i.std_dev for i in y])
        return (data - model) / (std_dev_data)

    return minimize(residual,
                    params,
                    args=(x_in, y_in),
                    method="least_squares")


def paramFitGetCovar(rslt, name):
    if rslt.params[name].stderr != None:
        i = rslt.var_names.index(name)
        return np.sqrt(rslt.covar[i][i])
    else:
        return float("nan")  #np.abs(0.01* rslt.params[name])


def setLogScale(x, y):
    if x:
        plt.xscale("log")
    if y:
        plt.yscale("log")


def export(fname, legndLoc='best', legend=True, width=None):
    if legend:
        plt.legend(loc=legndLoc, fontsize=10, frameon=True)
    if width != None:
        fig.set_size_inches(width * 6.49733,
                            width * 6.49733 * (5.0**.5 - 1.0) / 2.0)
        #6.49733

    plt.savefig(fname, dpi=400, bbox_inches="tight")


def startNewPlot(xText, yText, titleText, grid=True):
    global fig, ax
    #fig,ax = (0,0)
    mpl.rcParams['axes.formatter.use_locale'] = True
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = [
    #     r'\usepackage[detect-all,locale=DE]{siunitx}'
    # ]
    #plt.style.use('seaborn')
    fig, ax = plt.subplots()
    plt.title(titleText, fontsize=12)
    plt.xlabel(xText, fontsize=10)
    plt.ylabel(yText, fontsize=10)
    plt.grid(grid)

    #plt.legend(loc='upper right',fontsize=12,frameon=True)


def plotErrPointsAbs(x, y, xNoise, yNoise, label="", clr='k'):
    ax.errorbar(x,
                y,
                yerr=yNoise,
                xerr=xNoise,
                zorder=3,
                fmt='k.',
                markersize=0,
                ecolor=clr,
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                barsabove=True,
                label=label)


def plotErrPoints(x, y, label="", clr='k'):
    plotErrPointsAbs(np.array([i.n for i in x]),
                     np.array([i.n for i in y]),
                     np.array([i.std_dev for i in x]),
                     np.array([i.std_dev for i in y]),
                     label=label,
                     clr=clr)


def plot1DErrPoints(x, y, label="", clr='k'):
    ax.errorbar(x,
                np.array([i.n for i in y]),
                yerr=np.array([i.std_dev for i in y]),
                zorder=3,
                fmt='k.',
                marker="_",
                ecolor=clr,
                mec=clr,
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                barsabove=True,
                label=label)


def plotPoints(x, y, label="", clr='k'):
    ax.scatter(x, y, label=label, color=clr)


def plotImage(img, pixel_length, cmap='jet', cbarlbl=None):
    numcols = img.shape[1]
    numrows = img.shape[0]
    plt.imshow(img,
               extent=(-pixel_length / 2, (numcols - 0.5) * pixel_length,
                       (numrows - 0.5) * pixel_length, -pixel_length / 2),
               cmap=cmap)
    if cbarlbl != None:
        cbar = plt.colorbar()
        cbar.set_label(cbarlbl)


def plotFitFunc(func,
                p,
                xMin=-1,
                xMax=1,
                resolution=500,
                label="",
                clr="r",
                linestyle="-",
                log=False):
    x_fit = np.linspace(xMin, xMax, resolution)
    if log == True:
        x_fit = np.logspace(np.log(xMin) / np.log(10),
                            np.log(xMax) / np.log(10),
                            resolution,
                            base=10)
    y_fit = func(p, x_fit)
    ax.plot(x_fit,
            y_fit,
            clr,
            lw=2,
            label=label,
            zorder=2,
            linestyle=linestyle)


def plotFunc(func,
             xMin=-1,
             xMax=1,
             resolution=500,
             label="",
             clr="r",
             linestyle="-",
             log=False):
    x_fit = np.linspace(xMin, xMax, resolution)
    if log == True:
        x_fit = np.logspace(np.log(xMin) / np.log(10),
                            np.log(xMax) / np.log(10),
                            resolution,
                            base=10)
    y_fit = func(x_fit)
    ax.plot(x_fit,
            y_fit,
            clr,
            lw=2,
            label=label,
            zorder=2,
            linestyle=linestyle)


def plotLine(x, y, label="", clr="k", ls="-", linewidth=1, alpha=1):
    ax.plot(np.array([i for i in x]),
            np.array([i for i in y]),
            label=label,
            marker=None,
            color=clr,
            ls=ls,
            linewidth=linewidth,
            alpha=alpha)


def plotConfCurve(func,
                  p,
                  perr,
                  xMin=-1,
                  xMax=1,
                  resolution=500,
                  label="",
                  fcolor=(0, 0, 1, 0.15),
                  ecolor=(0, 0, 1, 0.3),
                  stdFak=5):
    x_fit = np.linspace(xMin, xMax, resolution)
    yu_fit = func(p + stdFak * perr, x_fit)
    yl_fit = func(p - stdFak * perr, x_fit)
    ax.fill_between(x_fit,
                    yu_fit,
                    yl_fit,
                    facecolor=fcolor,
                    edgecolor=ecolor,
                    label=label)


def plotFill(x,
             y_upper,
             y_lower,
             label="",
             fcolor=(0, 0, 1, 0.15),
             ecolor=(0, 0, 1, 0.3)):
    ax.fill_between(x,
                    y_upper,
                    y_lower,
                    facecolor=fcolor,
                    edgecolor=ecolor,
                    label=label)


def endPlot(blocking=True):
    plt.show(block=blocking)


def putInCRS(chisquared):
    textstr = 'Red. $\chi^2$: ' + format(chisquared, '.3f')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=props)


def printTeXTable(a, head=None):
    print(getAsTeXTable(a, head=head))


def getAsTeXTable(a, head=None):
    if len(a.shape) != 2:
        return -1
    output = "\\begin{tabular}{|" + "".join(
        [" L |" for i in range(a.shape[1])]) + "}\n\\hline\n"
    if (head != None):
        output += " & ".join(head) + "\\\\\n"
        output += "\\hline\n"

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            output += str(a[i][j])
            if j != (a.shape[1] - 1):
                output += " & "
        output += " \\\\" + "\n"

    output = output.replace("+/-", " \\pm ")
    output = output.replace(".", ",")
    output += "\\hline\n\\end{tabular}"
    return output
