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

def draw_graph(testcase, alloc_per_thread, iteration_num, SMs, allocs_size, sm_app, sm_mm, app_launch, app_finish,
app_sync, uni_req_num):
    sm_app_list = [sm_app[0][i] for i in range(SMs-1)]
    sm_mm_list  = [ sm_mm[0][i] for i in range(SMs-1)]
    #sms_list = [[sm_app_list[i], sm_mm_list[i]] for i in range(SMs-1)]
    sms_list = ['(' + str(sm_app_list[i]) + ', ' + str(sm_mm_list[i]) + ')' for i in range(SMs -1)]
    #sms_arr = [np.array([sm_app_list[i], sm_mm_list[i]]).reshape((1,2)) for i in range(SMs - 1)]
    launch_list = [round(app_launch[0][i],2) for i in range(SMs-1)]
    finish_list = [round(app_finish[0][i],2) for i in range(SMs-1)]
    sync_list = [round(app_sync[0][i],2) for i in range(SMs-1)]
    uni_req_num_list = [round(uni_req_num[0][i]) for i in range(SMs-1)]

    #print(uni_req_num_list)

    plt.figure(figsize=(30,10))
    
    plt.subplot(131)
    plt.plot(sms_list, uni_req_num_list)
    plt.scatter(sms_list, uni_req_num_list)
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm)")
    plt.ylabel("The number of requests per a sec depended on the number SMs app/mm")

    plt.subplot(132)
    plt.plot(sms_list, sync_list)
    plt.scatter(sms_list, sync_list)
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm)")
    plt.ylabel("App sync time in ms")
   
    plt.subplot(133)
    plt.plot(sms_list, launch_list)
    plt.scatter(sms_list, launch_list)
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm)")
    plt.ylabel("App launch time in ms")
        
    plt.suptitle(str(testcase))
    plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+'.png')
    

def run_test(testcase, alloc_per_thread, device, pmm_init, use_malloc, instant_size, iteration_num):
    SMs = getattr(device, 'MULTIPROCESSOR_COUNT')
    size = SMs - 1;
    
    sm_app      = pointer((c_int * size)())
    sm_mm       = pointer((c_int * size)())
    allocs_size = pointer((c_int * size)())
    app_launch  = pointer((c_float * size)())
    app_finish  = pointer((c_float * size)())
    app_sync    = pointer((c_float * size)())
    uni_req_num = pointer((c_float * size)())

    pmm_init(use_malloc, alloc_per_thread, instant_size, iteration_num, SMs, sm_app, sm_mm, allocs_size, app_launch,
    app_finish, app_sync, uni_req_num)

    draw_graph(testcase, alloc_per_thread, iteration_num, SMs, allocs_size, sm_app, sm_mm, app_launch, app_finish,
    app_sync, uni_req_num)
  
def main(argv):
    ### load shared libraries
    ouroboros = cdll.LoadLibrary('ouroboros_mm.so')
    halloc = cdll.LoadLibrary('halloc_mm.so')

    ### GPU properties
    device = cu.get_current_device()
    instant_size = 1024*1024*1024
    alloc_per_thread = 4
    iteration_num = 1

    if len(argv) > 0:
        alloc_per_thread = argv[0]

    if len(argv) > 1:
        iteration_num = argv[1]

    if len(argv) > 2:
        instant_size = argv[2]

    print("alloc_per_thread {} iteration_num {} instant_size {}".format(alloc_per_thread, iteration_num, instant_size))
    
    print("ouroboros test")
    test2 = ouroboros.pmm_init
    run_test("OUROBOROS", int(alloc_per_thread), device, test2, 1, instant_size, int(iteration_num))

    device.reset()
    
    print("halloc test")
    test = halloc.pmm_init
    run_test("HALLOC", int(alloc_per_thread), device, test, 1, instant_size, int(iteration_num))


if __name__ == "__main__":
    main(sys.argv[1:])
