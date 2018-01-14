from rpy2 import robjects as robj
from rpy2.robjects import Formula, Environment
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr, data
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import r, pandas2ri
import warnings

# The R 'print' function
# rprint = robjects.globalenv.get("print")
stats = importr('stats')
grdevices = importr('grDevices')
base = importr('base')
datasets = importr('datasets')
grid.activate()
pandas2ri.activate()
lattice = importr('lattice')

datasets = importr('datasets')
mtcars = data(datasets).fetch('mtcars')['mtcars']
formula = Formula('mpg ~ wt')
formula.getenvironment()['mpg'] = mtcars.rx2('mpg')
formula.getenvironment()['wt'] = mtcars.rx2('wt')


# Python 
import numpy as np


xyplot = lattice.xyplot


def explore_r(df):
    # print df
    # df = pandas2ri.py2ri(df)
    print type(df)

    testData = df

    df['x'] = df['HF']
    df['y'] = df['LF']

    # Next, you make an robject containing function that makes the plot.
    # the language in the function is pure R, so it can be anything
    # note that the R environment is blank to start, so ggplot2 has to be
    # loaded
    plotFunc = robj.r("""
     library(ggplot2)
    function(df){
     p <- ggplot(df, aes(x, y)) +
     geom_point()
    ggsave('rpy2_magic.png', plot = p, width = 4, height = 3)
     }
    """)
     
    # import graphics devices. This is necessary to shut the graph off
    # otherwise it just hangs and freezes python
    gr = importr('grDevices')
     
    # convert the testData to an R dataframe
    robj.pandas2ri.activate()
    testData_R = robj.conversion.py2ri(testData)
     
    # run the plot function on the dataframe
    plotFunc(testData_R)


