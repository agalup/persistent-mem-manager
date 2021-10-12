### sample python interface - pagerank

import sys, getopt, os, time
import ctypes
from ctypes import *

from pandas import DataFrame
import statsmodels.api as sm
from numpy import *
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px
import plotly
import matplotlib.pyplot as plt
import seaborn as sns

from numba import cuda as cu
  
def main(argv):
    ### load shared libraries
    ouroboros = cdll.LoadLibrary('ouroboros_mm.so')
    halloc = cdll.LoadLibrary('halloc_mm.so')

    print("halloc test")
    test2 = halloc.pmm_init
    test2(1, 1024*1024*1024)

    device = cu.get_current_device()
    device.reset()

    print("ouroboros test")
    test = ouroboros.pmm_init
    test(1, 1024*1024*1024)

if __name__ == "__main__":
    main(sys.argv[1:])
