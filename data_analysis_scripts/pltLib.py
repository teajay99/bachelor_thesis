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
from matplotlib.ticker import Locator

#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")

#from uncertainties.umath import *

fig, ax = (0, 0)


# Class stolen from https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        dmlower = majorlocs[1] - majorlocs[
            0]  # major tick difference at lower end
        dmupper = majorlocs[-1] - majorlocs[
            -2]  # major tick difference at upper end

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and (
            (majorlocs[0] != self.linthresh and dmlower > self.linthresh) or
            (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] * 10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0] - self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and (
            (np.abs(majorlocs[-1]) != self.linthresh
             and dmupper > self.linthresh) or
            (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1] * 10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1] + self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


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


def setSymLogScale(x, y, xthresh=1, ythresh=1):
    if x:
        plt.xscale("symlog", linthresh=xthresh)
        xaxis = plt.gca().xaxis
        xaxis.set_minor_locator(MinorSymLogLocator(xthresh))

    if y:
        plt.yscale("symlog", linthresh=ythresh)
        yaxis = plt.gca().yaxis
        yaxis.set_minor_locator(MinorSymLogLocator(ythresh))


def export(fname, legndLoc='best', legend=True, width=None, height=1.0):
    if legend:
        plt.legend(loc=legndLoc, fontsize=10, frameon=True)
    if width != None:
        fig.set_size_inches(width * 6.49733,
                            height * width * 6.49733 * (5.0**.5 - 1.0) / 2.0)
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


def plotErrPointsAbs(x,
                     y,
                     xNoise,
                     yNoise,
                     label="",
                     clr='k',
                     markersize=5,
                     marker="+"):
    ax.errorbar(x,
                y,
                yerr=yNoise,
                xerr=xNoise,
                zorder=3,
                fmt='k.',
                markersize=markersize,
                marker=marker,
                ecolor=clr,
                elinewidth=0.5,
                capsize=2,
                capthick=0.5,
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
    ax.errorbar(
        x,
        np.array([i.n for i in y]),
        yerr=np.array([i.std_dev for i in y]),
        zorder=3,
        fmt='k_',
        #marker="_",
        ecolor=clr,
        mec=clr,
        elinewidth=0.5,
        capsize=2,
        capthick=0.5,
        barsabove=True,
        label=label)


def plotPoints(x, y, label="", clr='k', marker="+"):
    ax.scatter(x, y, label=label, color=clr, marker=marker, s=20)


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
            lw=1,
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
            lw=1,
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


def cleanExponents(inp):
    out = re.sub(r'e\+0*(\d*)\s', r' \\cdot 10^{\1}', inp)
    out = re.sub(r'e\-0*(\d*)\s', r' \\cdot 10^{- \1}', out)
    return out


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
    return cleanExponents(output)
