#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "PerformanceMeasure.cuh"
#include "Utility.h"
#include "cuda.h"
#include "pmm-utils.cuh"

using namespace std;

//#define DEBUG
#ifdef DEBUG
#define debug(a...) printf(a)
#else
#define debug(a...)
#endif

//#define HALLOC__
//#define OUROBOROS__

/*
#ifndef HALLOC__
    #ifndef OUROBOROS__
        #define OUROBOROS__
    #endif
#endif*/

#ifdef HALLOC__
#include "Instance.cuh"
#endif

template <typename MemoryManagerType>
__global__
void mem_free(volatile int** d_memory, 
#ifdef OUROBOROS__
              MemoryManagerType* mm, 
#else 
    #ifdef HALLOC__
              MemoryManagerType mm,
    #endif
#endif
              volatile int* requests_num
        ){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num[0]){
        return;
    }
#ifdef OUROBOROS__
    mm->free((void*)d_memory[thid]);
#else 
    #ifdef HALLOC__
              mm.free((void*)d_memory[thid]);
    #endif
#endif
}

//producer
template <typename MemoryManagerType>
__global__
void mem_manager(volatile int* exit_signal, 
                volatile int* requests_number, 
                volatile int* request_iter,
                volatile int* request_signal, 
                volatile int* request_ids, 
#ifdef OUROBOROS__
                MemoryManagerType* mm,
#else
    #ifdef HALLOC__
                MemoryManagerType mm,
    #endif
#endif
                volatile int** d_memory,
                volatile int* request_mem_size,
                volatile int* lock, 
                int ouroboros_on){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            if (request_signal[request_id] == 1){

                // SEMAPHORE
                acquire_semaphore((int*)lock, request_id);
                debug("mm: request recieved %d\n", request_id); 
                int req_id = atomicAdd((int*)&request_iter[0], 1);
                request_ids[request_id] = req_id;

                if (ouroboros_on){
                    d_memory[req_id] = reinterpret_cast<volatile int*>
#ifdef HALLOC__
                        (mm.malloc(request_mem_size[request_id]));
#else
#ifdef OUROBOROS__
                        (mm->malloc(request_mem_size[request_id]));
#endif
#endif
                    assert(d_memory[req_id]);
                }

                // SIGNAL update
                atomicExch((int*)&request_signal[request_id], 2);

                release_semaphore((int*)lock, request_id);
                // SEMAPHORE

                debug("mm: request done %d\n", request_id);
            }
        }
    }
}

//consumer
__global__
void app(volatile int* exit_signal,
         volatile int** d_memory, 
         volatile int* request_signal, 
         volatile int* request_mem_size,
         volatile int* request_id, 
         volatile int* exit_counter, 
         volatile int* lock, 
         int ouroboros_on){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    request_mem_size[thid] = 4;
    request_id[thid] = -1;
    int req_id = -1;
    // SIGNAL update
    atomicExch((int*)&request_signal[thid], 1);
    __threadfence();
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
    
    // wait for success
    while (!exit_signal[0]){
        __threadfence();
        if (request_signal[thid] == 2){
            
            // SEMAPHORE
            acquire_semaphore((int*)lock, thid);
            req_id = request_id[thid];
            if (req_id >= 0 && ouroboros_on) {
                assert(d_memory[req_id]);
                d_memory[req_id][0] = thid;
            }
            request_signal[thid] = 0;
            __threadfence();
            debug("app: request %d success\n", thid);
            release_semaphore((int*)lock, thid);
            // SEMAPHORE
        
            break;
        }
    }
    atomicAdd((int*)&exit_counter[0], 1);
}


int main(int argc, char *argv[]){

#ifdef OUROBOROS__
    //Ouroboros initialization
    size_t instantitation_size = 7168ULL * 1024ULL * 1024ULL;
    using MemoryMangerType = OuroPQ;
    MemoryMangerType memory_manager;
    memory_manager.initialize(instantitation_size);
#else
#ifdef HALLOC__
    //Halloc initialization
    size_t instantitation_size = 2048ULL * 1024ULL * 1024ULL;
    using MemoryManagerType = MemoryManagerHalloc;
    MemoryManagerType memory_manager(instantitation_size);
#endif
#endif
    
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t mm_stream, app_stream;
    GUARD_CU(cudaStreamCreateWithFlags( &mm_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    
    int* exit_signal;
    GUARD_CU(cudaMallocManaged(&exit_signal, sizeof(int32_t)));

    int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int max_block_number = deviceProp.multiProcessorCount;
    printf("max block number %d\n", max_block_number);

    int ouroboros_on = 1;
    if (argc > 1){
        ouroboros_on = atoi(argv[1]);
    }

    int block_size = 1024;
    
    std::cout << "\t\t#allocs\t\t" << "#sm app\t\t" << "#sm mm\t\t" << "app launch\t" << "app finished\t" << "app finish sync\n";

    for (int mm_grid_size = 1; mm_grid_size < max_block_number; ++mm_grid_size){

        *exit_signal = 0;
        *exit_counter = 0;

        int app_grid_size = (max_block_number) - mm_grid_size;
        
        int requests_num{app_grid_size*block_size};
        
        //Request auxiliary
        RequestType requests;
        requests.init(requests_num);
        requests.memset();

        //Timing variables
        PerfMeasure timing_app, timing_mm, timing_total, timing_total_sync;

//        printf("mm starts\n");
        timing_mm.startMeasurement();
        //Run presistent kernel (Memory Manager)
        mem_manager<<<mm_grid_size, block_size, 0, mm_stream>>>(exit_signal,
                requests.requests_number, 
                requests.request_iter, 
                requests.request_signal, 
                requests.request_id,
#ifdef OUROBOROS__
                memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                memory_manager,
#endif
#endif
                requests.d_memory,
                requests.request_mem_size,
                requests.lock, 
                ouroboros_on);
        timing_mm.stopMeasurement();

        GUARD_CU(cudaPeekAtLastError());
//        printf("mm stop\n");

//        printf("app start\n");
        timing_app.startMeasurement();
        timing_total.startMeasurement();
        timing_total_sync.startMeasurement();
        //Run application
        //app<<<grid_size, block_size, 0, app_stream>>>(exit_signal, 
        app<<<app_grid_size, block_size, 0, app_stream>>>(exit_signal, 
                requests.d_memory, 
                requests.request_signal, 
                requests.request_mem_size, 
                requests.request_id, 
                exit_counter, 
                requests.lock, 
                ouroboros_on);
        timing_app.stopMeasurement();

        GUARD_CU(cudaPeekAtLastError());
//        printf("app stop\n");

        // Check results
        int old_counter = -1;
        int iter = 0;
        int iter2 = 0;
        long long iter_mean = 0;
        int time_limit = 1000000000;
        while (iter2 < time_limit){
            if (exit_counter[0] == block_size*app_grid_size){
                timing_total.stopMeasurement();
                *exit_signal = 1;
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
                timing_total_sync.stopMeasurement();
                if (ouroboros_on){
                    test1<<<app_grid_size, block_size, 0, app_stream>>>(requests.d_memory);
                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    mem_test((int**)requests.d_memory, requests_num, app_grid_size, block_size, mm_stream);
                }
                break;
            }else{
                if (exit_counter[0] != old_counter){
                    old_counter = exit_counter[0];
                
                    ++iter;
                    iter_mean += iter2;
                    iter2 = 0;

                    if (iter%1000 == 0){
                        //debug("no break, exit_counter = %d\n", exit_counter[0]);
                        //printf("no break, exit_counter = %d, change after %d iterations\n", exit_counter[0],\
                        iter_mean/iter);
                    }
                }
                ++iter2;
            }
            if (iter2 >= time_limit){
                printf("time limit exceed, break\n");
                *exit_signal = 1;
            }
        }

        iter_mean /= iter;
        //printf("new change each %d iterations\n", iter_mean);

        //Deallocate device memory
        mem_free<<<app_grid_size, block_size, 0, app_stream>>>(
                requests.d_memory, 
#ifdef OUROBOROS__
                memory_manager.getDeviceMemoryManager(),
#else
    #ifdef HALLOC__
                memory_manager,
    #endif
#endif
                requests.requests_number);

        requests.free();

        // Output
        auto app_time = timing_app.generateResult();
        auto total_time = timing_total.generateResult();
        auto total_sync_time = timing_total_sync.generateResult();
        //printf("%lf %lf %lf\n", app_time, total_time, total_sync_time);
        //std::cout << "#allocs\t" << "#sm app\t" << "#sm mm\t" << "app launch time\t" << "app finished time\n";
        printf("\t\t%d\t\t| %d\t\t| %d\t\t| %.2lf\t\t| %.2lf\t\t| %.2lf\n", 
                requests_num, app_grid_size, mm_grid_size, app_time.mean_, total_time.mean_, total_sync_time.mean_);
    }

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
    printf("DONE!\n");
    return 0;
}

