import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
import matplotlib as mpl


def plotPoints(points):
    fig, ax = plt.subplots()

    arrowLength = [(i[0]**2) + (i[1]**2) + (i[2]**2) + (i[3]**2)
                   for i in points]

    ax.quiver([i[0] for i in points], [i[1] for i in points],
              [i[2] for i in points], [i[3] for i in points])
    plt.show()
