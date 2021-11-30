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


from pylab import *

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
    
    uni_req_num = [round(uni_req_num    [0][i],1) for i in range(size)]

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111, projection='3d')

    #color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    #color_map.set_array(uni_req_num)

    ##plt.scatter(sm_app_list, sm_mm_list, sm_gc_list, marker='s', s=200, color='green')
    #plt.scatter(sm_app_list, sm_mm_list, sm_gc_list, marker='s')
    #plt.colorbar(color_map)

    app = np.array(sm_app_list)
    mm  = np.array(sm_mm_list)
    gc  = np.array(sm_gc_list)
    req = np.array(uni_req_num)

    ax = plt.axes(projection = '3d')
    
    #colors = cm.rainbow(np.linspace(0, 1, req.shape[0]))
    #colors = cm.rainbow(req)
    #for dataset,color in zip(gc,colors):
    #    ax.scatter3D(app,mm,dataset,color=color)

    SIZE = app[size-1]+1
    SIZE2 = SIZE**2
    #gc_f = np.zeros(SIZE2).reshape(SIZE, SIZE)
    req_f = np.zeros(SIZE2).reshape(SIZE, SIZE)
    #print(gc_f.shape)
    #print(gc_f)

    app_f = np.arange(1, 35, 1)
    mm_f = np.arange(1, 35, 1)
    app_f, mm_f = np.meshgrid(app_f, mm_f)
    gc_f = SIZE - app_f - mm_f

    print(app_f)
    print(mm_f)
    print(gc_f)

    for i in range(size):
        #app[i] = app[i] - 1
        #mm[i] = mm[i] - 1
        #gc[i] = gc[i] - 1
        #gc_f[app[i]][mm[i]] = gc[i]
        req_f[app[i]][mm[i]] = req[i]
    #print(gc_f)

    #print(app.shape)
    #print(mm.shape)
    print(app_f.shape)
    print(mm_f.shape)
    print(gc_f.shape)
    #print(req_f.shape)

    import matplotlib.cm as cmx
    from mpl_toolkits.mplot3d import Axes3D
    cm = plt.get_cmap('inferno')
    cNorm = matplotlib.colors.Normalize(vmin=min(req), vmax=max(req))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure(figsize = (20,20))
    ax = Axes3D(fig)
    ax.scatter(app, mm, gc, c=scalarMap.to_rgba(req), marker="^", s=100)
    #ax.plot_surface(app_f, mm_f, gc_f, facecolors=scalarMap.to_rgba(req_f))

    ax.set_xticks(np.arange(SIZE))
    ax.set_yticks(np.arange(SIZE))
    ax.set_zticks(np.arange(SIZE))

    scalarMap.set_array(req)
    fig.colorbar(scalarMap,label='Req per sec')

    #C = np.linspace(min(req), max(req), req_f.size).reshape(req_f.shape)
    ##C = np.linspace(min(req), max(req), req.size).reshape(req.shape)
    #print(C.shape)
    #scamap = plt.cm.ScalarMappable(cmap='inferno')
    #fcolors = scamap.to_rgba(C)
    #print(fcolors.shape)
    #ax.plot_surface(mm, app, gc_f, facecolors=fcolors, cmap='inferno')
    ##ax.plot_surface(mm, app, gc, facecolors=fcolors, cmap='inferno')

    #plt.scatter(app, mm, gc)
   
    #print(sms)
    #ax = sns.heatmap(sms)

    #ax.set_title("3D Heatmap")
    ax.set_xlabel('app')
    ax.set_ylabel('mm')
    ax.set_zlabel('gc')

    plt.suptitle(str(testcase))

    from numpy import arange

    ii = -135
    jj = -35.265
    ax.view_init(azim=ii, elev=jj)
    plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+"_"+str(size)+"movie_"+str(ii)+"_"+str(jj)+".pdf")

       #for ii in arange(-146,-144,0.1):
    #    for jj in arange(43, 45, 0.1):
    #        ax.view_init(ii, jj)
    #        #ax.view_init(azim-45, elev=42)
    #        plt.savefig(str(testcase)+"_"+str(SMs)+"SMs_"+str(alloc_per_thread)+"_"+str(iteration_num)+"_"+str(size)+"movie_"+str(ii)+"_"+str(jj)+".pdf")

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
    #halloc = cdll.LoadLibrary('halloc_mm.so')

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
