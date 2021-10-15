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

def draw_graph(testcase, alloc_per_thread, iteration_num, SMs, allocs_size, 
               sm_app, sm_mm, app_launch, app_finish,
               app_sync_pmm, uni_req_num_pmm, app_sync, uni_req_num):

    sm_app_list = [sm_app[0][i] for i in range(SMs-1)]
    sm_mm_list  = [ sm_mm[0][i] for i in range(SMs-1)]
    sms_list = ['(' + str(sm_app_list[i]) + ', ' + 
                str(sm_mm_list[i]) + ') ' for i in range(SMs -1)]

    sms_list_req = ['(' + str(sm_app_list[i]) + ', ' + 
                str(sm_mm_list[i]) + ') ' + 
                str(allocs_size[0][i]) for i in range(SMs -1)]
    
    launch_list     = [round(app_launch     [0][i],2) for i in range(SMs-1)]
    finish_list     = [round(app_finish     [0][i],2) for i in range(SMs-1)]
    sync_list_pmm   = [round(app_sync_pmm   [0][i],2) for i in range(SMs-1)]
    uni_req_num_pmm = [round(uni_req_num_pmm[0][i],1) for i in range(SMs-1)]
    
    sync_list       = [round(app_sync       [0][i],2) for i in range(SMs-1)]
    uni_req_num     = [round(uni_req_num    [0][i],1) for i in range(SMs-1)]

    plt.figure(figsize=(30,15))
    
    plt.subplot(131)
    plt.plot   (sms_list, uni_req_num, color='blue')
    plt.scatter(sms_list, uni_req_num, color='blue')
    plt.plot   (sms_list, uni_req_num_pmm, color='red')
    plt.scatter(sms_list, uni_req_num_pmm, color='red')
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm)")
    plt.ylabel("Number of requests per sec")
    plt.title("Red - persistent memory man. Blue - standard")

    plt.subplot(132)
    plt.plot    (sms_list_req, sync_list, color='blue')
    plt.scatter (sms_list_req, sync_list, color='blue')
    plt.plot    (sms_list_req, sync_list_pmm, color='red')
    plt.scatter (sms_list_req, sync_list_pmm, color='red')
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm) #requests (allocations)")
    plt.ylabel("ms")
    plt.title("Red - persistent memory man. Blue - standard")
   
    plt.subplot(133)
    plt.plot    (sms_list, launch_list)
    plt.scatter (sms_list, launch_list)
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm)")
    plt.ylabel("ms")
    plt.title("App launch time persistent man")
        
    plt.suptitle(str(testcase))
    plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+'.png')
    

def run_test(testcase, alloc_per_thread, device, pmm_init, perf_alloc, use_malloc, instant_size, iteration_num):


    SMs = getattr(device, 'MULTIPROCESSOR_COUNT')
    size = SMs - 1;
   
    instant_size    = pointer((c_size_t)(instant_size))
    sm_app          = pointer((c_int * size)())
    sm_mm           = pointer((c_int * size)())
    allocs_size     = pointer((c_int * size)())
    app_launch      = pointer((c_float * size)())
    app_finish      = pointer((c_float * size)())
    app_sync_pmm    = pointer((c_float * size)())
    app_sync        = pointer((c_float * size)())
    uni_req_num_pmm = pointer((c_float * size)())
    uni_req_num     = pointer((c_float * size)())

    print("instant_size = ", instant_size[0])

    pmm_init(use_malloc, alloc_per_thread, instant_size, iteration_num, 
             SMs, sm_app, sm_mm, allocs_size, app_launch, app_finish, 
             app_sync_pmm, uni_req_num_pmm);

    device.reset()

    perf_alloc(alloc_per_thread, instant_size, iteration_num, SMs, 
               app_sync, uni_req_num, use_malloc)

    draw_graph(testcase, alloc_per_thread, iteration_num, SMs, allocs_size, 
               sm_app, sm_mm, app_launch, app_finish, 
               app_sync_pmm, uni_req_num_pmm, app_sync, uni_req_num)
  
def main(argv):
    ### load shared libraries
    ouroboros = cdll.LoadLibrary('ouroboros_mm.so')
    halloc = cdll.LoadLibrary('halloc_mm.so')

    ### GPU properties
    device = cu.get_current_device()

    instant_size = (2 ** (3+10+10+10)) #(8*1024*1024*1024)
    print("instant_size ", instant_size)
    alloc_per_thread = 4
    iteration_num = 1

    if len(argv) > 0:
        alloc_per_thread = argv[0]

    if len(argv) > 1:
        iteration_num = argv[1]

    if len(argv) > 2:
        instant_size = argv[2]

    print("alloc_per_thread {} iteration_num {} instant_size {}".format(alloc_per_thread, iteration_num, instant_size))
    
    #print("ouroboros test")
    #pmm_init = ouroboros.pmm_init
    #perf_alloc = ouroboros.perf_alloc
    #run_test("OUROBOROS", int(alloc_per_thread), device, pmm_init, perf_alloc, 1, instant_size, int(iteration_num))

    device.reset()
    
    print("halloc test")
    pmm_init = halloc.pmm_init
    perf_alloc = halloc.perf_alloc
    run_test("HALLOC", int(alloc_per_thread), device, pmm_init, perf_alloc, 1, instant_size, int(iteration_num))


if __name__ == "__main__":
    main(sys.argv[1:])
