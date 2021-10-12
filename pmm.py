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

def run_test(device, test, use_malloc, instant_size):
    SMs = getattr(device, 'MULTIPROCESSOR_COUNT')
    size = SMs - 1;
    
    sm_app      = pointer((c_int * size)())
    sm_mm       = pointer((c_int * size)())
    allocs_size = pointer((c_int * size)())
    app_launch  = pointer((c_float * size)())
    app_finish  = pointer((c_float * size)())
    app_sync    = pointer((c_float * size)())

    test(use_malloc, instant_size, SMs, sm_app, sm_mm,
    allocs_size, app_launch, app_finish, app_sync)

  
def main(argv):
    ### load shared libraries
    ouroboros = cdll.LoadLibrary('ouroboros_mm.so')
    halloc = cdll.LoadLibrary('halloc_mm.so')

    ### GPU properties
    device = cu.get_current_device()
    instant_size = 1024*1024*1024
    
    print("halloc test")
    test = halloc.pmm_init
    run_test(device, test, 1, instant_size)

    device.reset()

    print("ouroboros test")
    test2 = ouroboros.pmm_init
    run_test(device, test2, 1, instant_size)

if __name__ == "__main__":
    main(sys.argv[1:])
