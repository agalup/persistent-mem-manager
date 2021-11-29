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

def draw_graph(plt, testcase, alloc_per_thread, iteration_num, SMs, allocs_size, 
               sm_app, sm_mm, sm_gc,
               #app_launch, app_finish, app_sync_pmm, uni_req_num_pmm, app_sync, 
               uni_req_num, array_size):
               #use_malloc):

    print("results size ", array_size[0])

    size = array_size[0]

    sm_app_list = [sm_app[0][i] for i in range(size)]
    sm_mm_list  = [ sm_mm[0][i] for i in range(size)]
    sm_gc_list  = [ sm_gc[0][i] for i in range(size)]
    sms_list = ['(' + str(sm_app_list[i]) + ', ' + 
                str(sm_mm_list[i]) + ', ' +
                str(sm_gc_list[i]) + ') ' for i in range(size)]

    sms_list_req = ['(' + str(sm_app_list[i]) + ', ' + 
                str(sm_mm_list[i]) + ') ' + 
                str(allocs_size[0][i]) for i in range(size)]
    
    #launch_list     = [round(app_launch     [0][i],2) for i in range(size)]
    #finish_list     = [round(app_finish     [0][i],2) for i in range(size)]
    #sync_list_pmm   = [round(app_sync_pmm   [0][i],2) for i in range(size)]
    #uni_req_num_pmm = [round(uni_req_num_pmm[0][i],1) for i in range(size)]
    
    #sync_list       = [round(app_sync       [0][i],2) for i in range(size)]
    uni_req_num     = [round(uni_req_num    [0][i],1) for i in range(size)]

    #col1 = 'darkblue'
    #col2 = 'crimson'

    #if (not use_malloc):
    #    col1 = 'blue'
    #    col2 = 'red'

    col2 = 'blue'
    col1 = 'red'

    #plt.subplot(131)
    plt.plot   (sms_list, uni_req_num, color=col1)
    plt.scatter(sms_list, uni_req_num, color=col1)
    #plt.plot   (sms_list, uni_req_num_pmm, color=col2)
    #plt.scatter(sms_list, uni_req_num_pmm, color=col2)
    plt.xticks(rotation=90)
    plt.xlabel("(SMs app, SMs mm, SMs gc)")
    plt.ylabel("Number of requests per sec")
    plt.title("Red - persistent memory man")

    #plt.subplot(132)
    #plt.plot    (sms_list_req, sync_list, color=col1)
    #plt.scatter (sms_list_req, sync_list, color=col1)
    #plt.plot    (sms_list_req, sync_list_pmm, color=col2)
    #plt.scatter (sms_list_req, sync_list_pmm, color=col2)
    #plt.xticks(rotation=90)
    #plt.xlabel("(SMs app, SMs mm) #requests (allocations)")
    #plt.ylabel("ms")
    #plt.title("Red - persistent memory man. Blue - standard")
   
    #plt.subplot(133)
    #plt.plot    (sms_list, launch_list, color=col2)
    #plt.scatter (sms_list, launch_list, color=col2)
    #plt.xticks(rotation=90)
    #plt.xlabel("(SMs app, SMs mm)")
    #plt.ylabel("ms")
    #plt.title("App launch time persistent man")
        
    plt.suptitle(str(testcase))

    use_malloc = 1

    if use_malloc:
        plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+"_"+str(size)+'.png')
    else:
        plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+
        "_malloc_off"+'.png')

def run_test(testcase, alloc_per_thread, device, pmm_init, 
            #perf_alloc, 
            instant_size, iteration_num, kernel_iter_num):


    print("instant_size = ", instant_size)
    SMs = getattr(device, 'MULTIPROCESSOR_COUNT')
    size = 600 #SMs - 1;
   
    plt.figure(figsize=(30,15))

    #use malloc:
    use_malloc = 1
    instant_size    = pointer((c_size_t)(instant_size))
    sm_app          = pointer((c_int * size)())
    sm_mm           = pointer((c_int * size)())
    sm_gc           = pointer((c_int * size)())
    requests_num    = pointer((c_int * size)())
    array_size      = pointer((c_int)())
    #app_launch      = pointer((c_float * size)())
    #app_finish      = pointer((c_float * size)())
    #app_sync_pmm    = pointer((c_float * size)())
    #app_sync        = pointer((c_float * size)())
    #uni_req_num_pmm = pointer((c_float * size)())
    uni_req_num     = pointer((c_float * size)())

    print("pmm_init, use malloc")
    pmm_init(kernel_iter_num, alloc_per_thread, instant_size, iteration_num, 
             SMs, sm_app, sm_mm, sm_gc, requests_num, uni_req_num, array_size);
             #app_launch, app_finish, 
             #app_sync_pmm, uni_req_num_pmm);

    #device.reset()

    #print("perf_alloc, use malloc")
    #perf_alloc(alloc_per_thread, instant_size, iteration_num, SMs, 
    #           app_sync, uni_req_num, use_malloc)

    ##draw both
    draw_graph(plt, testcase, alloc_per_thread, iteration_num, SMs, requests_num, 
               sm_app, sm_mm, sm_gc, uni_req_num, array_size)
  
    #device.reset()
    #
    #instant_size0    = pointer((c_size_t)(instant_size))
    #sm_app0          = pointer((c_int * size)())
    #sm_mm0           = pointer((c_int * size)())
    #allocs_size0     = pointer((c_int * size)())
    #app_launch0      = pointer((c_float * size)())
    #app_finish0      = pointer((c_float * size)())
    #app_sync_pmm0    = pointer((c_float * size)())
    #app_sync0        = pointer((c_float * size)())
    #uni_req_num_pmm0 = pointer((c_float * size)())
    #uni_req_num0     = pointer((c_float * size)())

    ##donot use malloc:
    #use_malloc = 0

    #print("pmm_init, do not use malloc")
    #pmm_init(use_malloc, alloc_per_thread, instant_size0, iteration_num, 
    #         SMs, sm_app0, sm_mm0, allocs_size0, app_launch0, app_finish0, 
    #         app_sync_pmm0, uni_req_num_pmm0);

    #device.reset()

    #print("perf_alloc, do not use malloc")
    #perf_alloc(alloc_per_thread, instant_size0, iteration_num, SMs, 
    #           app_sync0, uni_req_num0, use_malloc)

    #draw_graph(plt, testcase, alloc_per_thread, iteration_num, SMs, allocs_size0, 
    #           sm_app0, sm_mm0, app_launch0, app_finish0, app_sync_pmm0, 
    #           uni_req_num_pmm0, app_sync0, uni_req_num0, use_malloc)


def main(argv):
    ### load shared libraries
    ouroboros = cdll.LoadLibrary('ouroboros_mm.so')
    halloc = cdll.LoadLibrary('halloc_mm.so')

    ### GPU properties
    device = cu.get_current_device()

    #instant_size = (2 ** (3+10+10+10)) #(8*1024*1024*1024)
    instant_size = 7 * 1024*1024*1024
    print("instant_size ", instant_size)
    alloc_per_thread = 8
    iteration_num = 1
    kernel_iter_num = 1

    #if len(argv) > 0:
    #    alloc_per_thread = argv[0]

    if len(argv) > 0:
        iteration_num = argv[0]

    if len(argv) > 1:
        kernel_iter_num = argv[1]

    if len(argv) > 3:
        instant_size = argv[3]

    #print("alloc_per_thread {} iteration_num {} instant_size {}".format(alloc_per_thread, iteration_num, instant_size))
    
    print("ouroboros test")
    pmm_init = ouroboros.pmm_init
    #perf_alloc = ouroboros.perf_alloc
    run_test("OUROBOROS", int(alloc_per_thread), device, pmm_init, #perf_alloc, 
                instant_size, int(iteration_num), int(kernel_iter_num))

    #device.reset()
    
    #print("halloc test")
    #pmm_init = halloc.pmm_init
    #perf_alloc = halloc.perf_alloc
    #run_test("HALLOC", int(alloc_per_thread), device, pmm_init, perf_alloc, malloc_on, instant_size, int(iteration_num))


if __name__ == "__main__":
    main(sys.argv[1:])
