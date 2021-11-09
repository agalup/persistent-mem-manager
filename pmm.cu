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

/*
#ifndef HALLOC__
    #ifndef OUROBOROS__
        #define OUROBOROS__
    #endif
#endif*/

#ifdef HALLOC__
#include "Instance.cuh"
#endif

extern "C"{

#ifdef OUROBOROS__
    //Ouroboros initialization
    #define MemoryManagerType OuroPQ
#endif
#ifdef HALLOC__
    //Halloc initialization
    #define MemoryManagerType MemoryManagerHalloc
#endif

__global__
void mem_free(volatile int** d_memory, 
              volatile int* request_id, 
#ifdef OUROBOROS__
              //OuroPQ* mm,
              MemoryManagerType* mm, 
#else 
    #ifdef HALLOC__
              //MemoryManagerHalloc mm,
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
    if (request_id[thid] > -1)
        mm->free((void*)d_memory[thid]);
#else 
    #ifdef HALLOC__
    assert(request_id);
    assert(d_memory);
    if (request_id[thid] > -1 && d_memory[thid]){
        assert(d_memory[thid]);
        mm.free((void*)d_memory[thid]);
    }
    #endif
#endif
}

__global__
void mem_free_perf(volatile int** d_memory, 
#ifdef OUROBOROS__
              MemoryManagerType* mm, 
#else 
    #ifdef HALLOC__
              MemoryManagerType mm, 
    #endif
#endif
              int requests_num, 
              int turn_on
        ){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num){
        return;
    }
    if (turn_on){
#ifdef OUROBOROS__
            if (d_memory[thid])
                mm->free((void*)d_memory[thid]);
#else 
    #ifdef HALLOC__
            if (d_memory[thid])
                mm.free((void*)d_memory[thid]);
    #endif
#endif
    }
}


__device__
void _request_processing(
        int request_id, 
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* request_iter,
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
        int turn_on
        ){

    // SEMAPHORE
    acquire_semaphore((int*)lock, request_id);
    debug("mm: request recieved %d\n", request_id); 
    int req_id = atomicAdd((int*)&request_iter[0], 1);
    request_ids[request_id] = req_id;

    if (turn_on){
        
        d_memory[req_id] = reinterpret_cast<volatile int*>
#ifdef HALLOC__
            (mm.malloc(request_mem_size[request_id]));
#else
#ifdef OUROBOROS__
        (mm->malloc(request_mem_size[request_id]));
#endif
#endif
        __threadfence();
        if (!exit_signal[0]){
            if (!d_memory[req_id]){
                printf("request failed\n");
                assert(d_memory[req_id]);
            }
        }
    }

    // SIGNAL update
    atomicExch((int*)&request_signal[request_id], 2);

    release_semaphore((int*)lock, request_id);
    // SEMAPHORE
}

//producer
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
        int turn_on){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            if (request_signal[request_id] == 1){
                _request_processing(request_id, exit_signal, request_signal, 
                                request_iter, request_ids, mm, d_memory, 
                                request_mem_size, lock, turn_on);
                debug("mm: request done %d\n", request_id);
            }
        }
    }
}

__device__
void post_request(volatile int* lock,
                  volatile int* request_mem_size,
                  volatile int* request_id,
                  volatile int* request_signal,
                  int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    request_mem_size[thid] = size_to_alloc;
    request_id[thid] = -1;
    // SIGNAL update
    atomicExch((int*)&request_signal[thid], 1);
    __threadfence();
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
}

__device__
void request_recieved(volatile int* lock,
                      volatile int* request_id,
                      volatile int* exit_signal,
                      volatile int** d_memory,
                      volatile int* request_signal,
                      int& req_id,
                      int turn_on
                      ){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    req_id = request_id[thid];
    if (req_id >= 0 && turn_on && !exit_signal[0]) {
        assert(d_memory[req_id]);
        d_memory[req_id][0] = thid;
    }
    request_signal[thid] = 0;
    __threadfence();
    debug("app: request %d success\n", thid);
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
}

__device__
void mm_malloc(volatile int* exit_signal,
        volatile int** d_memory,
        volatile int* request_signal,
        volatile int* request_mem_size, 
        volatile int* request_id,
        volatile int* lock,
        int size_to_alloc,
        int turn_on
        ){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    post_request(lock, request_mem_size, request_id, request_signal, size_to_alloc);

    // wait for success
    while (!exit_signal[0]){
        __threadfence();
        if (request_signal[thid] == 2){
            request_recieved(lock, request_id, exit_signal, d_memory, request_signal, req_id, turn_on);
            break;
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
        int size_to_alloc,
        int turn_on){

    mm_malloc(exit_signal, d_memory, request_signal, request_mem_size, request_id, lock, size_to_alloc, turn_on);
    
    atomicAdd((int*)&exit_counter[0], 1);
}

__global__
void simple_alloc(volatile int** d_memory, 
        volatile int* exit_counter,
#ifdef OUROBOROS__
        //OuroPQ* mm,
        MemoryManagerType* mm, 
#else 
#ifdef HALLOC__
        //MemoryManagerHalloc mm,
        MemoryManagerType mm, 
#endif
#endif
        int size_to_alloc,
        int turn_on
        ){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (turn_on){
        d_memory[thid] = reinterpret_cast<volatile int*>
#ifdef HALLOC__
            (mm.malloc(size_to_alloc));
#else
#ifdef OUROBOROS__
        (mm->malloc(size_to_alloc));
#endif
#endif
    }
    atomicAdd((int*)&exit_counter[0], 1);
}

void perf_alloc(int size_to_alloc, size_t* ins_size, size_t num_iterations, 
            int SMs, float* app_sync, float* uni_req_num, int turn_on){

    auto instant_size = *ins_size;

#ifdef OUROBOROS__
    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);
#else
#ifdef HALLOC__
    //Halloc initialization
    //size_t instantitation_size = 2048ULL * 1024ULL * 1024ULL;
    MemoryManagerType memory_manager(instant_size);
#endif
#endif
 
    int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    GUARD_CU(cudaPeekAtLastError());

    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t app_stream;
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    
    int block_size = 1024;
    printf("size to alloc per thread %d, num iterations %d\n", size_to_alloc, num_iterations);
    std::cout << "\t\t#allocs\t\t" << "#sm app\t\t" << "#req per sec\t\t" << "app finish sync\n";

    for (int app_grid_size = 1; app_grid_size < SMs; ++app_grid_size){

        int requests_num{app_grid_size*block_size};
        //Timing variables
        PerfMeasure timing_total_sync;
        
        for (int iteration = 0; iteration < num_iterations; ++iteration){
            *exit_counter = 0;

            volatile int** d_memory{nullptr};
            GUARD_CU(cudaMalloc(&d_memory, requests_num * sizeof(volatile int*)));
            GUARD_CU(cudaPeekAtLastError());

            timing_total_sync.startMeasurement();
            //Run application
            simple_alloc<<<app_grid_size, block_size, 0, app_stream>>>(
                    d_memory,
                    exit_counter, 
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    memory_manager,
#endif
#endif                   
                    size_to_alloc, 
                    turn_on);
            GUARD_CU(cudaPeekAtLastError());

            // Check results
            int old_counter = -1;
            long long iter = 0;
            long long iter2 = 0;
            long long iter_mean = 0;
            long long  time_limit = 10000000000;
            while (iter2 < time_limit){
                if (exit_counter[0] == requests_num){
                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    timing_total_sync.stopMeasurement();
                    if (turn_on){
                        test1<<<app_grid_size, block_size, 0, app_stream>>>(d_memory);
                        GUARD_CU(cudaDeviceSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        mem_test((int**)d_memory, requests_num, app_grid_size, block_size);
                    }
                    break;
                }else{
                    if (exit_counter[0] != old_counter){
                        old_counter = exit_counter[0];
                        ++iter;
                        iter_mean += iter2;
                        iter2 = 0;
                    }
                    ++iter2;
                }
                //was no change
                if (iter2 >= time_limit){
                    printf("time limit exceed, break\n");
                }
            }

            if (iter != 0)
                iter_mean /= iter;
        
            if (iter2 >= time_limit){
                printf("%d: sync\n", __LINE__);
            }
            GUARD_CU(cudaDeviceSynchronize());
            if (iter2 >= time_limit){
                printf("%d: sync done\n", __LINE__);
            }
            GUARD_CU(cudaPeekAtLastError());
            //Deallocate device memory
            mem_free_perf<<<app_grid_size, block_size, 0, app_stream>>>(
                    d_memory, 
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    memory_manager,
#endif
#endif
                    requests_num, turn_on);
            GUARD_CU(cudaFree((void*)d_memory));
            GUARD_CU(cudaPeekAtLastError());
        }
        // Output
        auto total_sync_time = timing_total_sync.generateResult();
        app_sync  [app_grid_size - 1] = (total_sync_time.mean_);
        // The number of requests done per a second
        uni_req_num[app_grid_size - 1] = 
                            (requests_num * 1000.0)/total_sync_time.mean_;

        printf("\t\t%d\t\t| %d\t\t| %.2lf\t\t| %.2lf\n", 
                requests_num, app_grid_size, uni_req_num[app_grid_size - 1], 
                total_sync_time.mean_);
    }

    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
}

void pmm_init(int turn_on, int size_to_alloc, size_t* ins_size, size_t num_iterations, 
            int SMs, int* sm_app, int* sm_mm, int* allocs, float* app_launch, 
            float* app_finish, float* app_sync, float* uni_req_num){

    auto instant_size = *ins_size;

#ifdef OUROBOROS__
    //Ouroboros initialization
    MemoryManagerType memory_manager;
    memory_manager.initialize(instant_size);
#else
#ifdef HALLOC__
    //Halloc initialization
    MemoryManagerType memory_manager(instant_size);
#endif
#endif
 
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    //Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t mm_stream, app_stream;
    GUARD_CU(cudaStreamCreateWithFlags( &mm_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    
    int* exit_signal;
    GUARD_CU(cudaMallocManaged(&exit_signal, sizeof(int32_t)));
    GUARD_CU(cudaPeekAtLastError());

    int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    
    int block_size = 1024;
    printf("size to alloc per thread %d, num iterations %d, instantsize %ld\n", 
                size_to_alloc, num_iterations, instant_size);
    std::cout << "\t\t#allocs\t\t" << "#sm app\t\t" << "#sm mm\t\t" << 
                "#req per sec\t\t" << "app finish sync\n";

    //for (int app_grid_size = 1; app_grid_size < SMs; ++app_grid_size){

        int app_grid_size = 33;
        int mm_grid_size = SMs - app_grid_size;
        int requests_num{app_grid_size*block_size};

        //output
        sm_app[app_grid_size - 1] = app_grid_size;
        sm_mm [app_grid_size - 1] = mm_grid_size;
        allocs[app_grid_size - 1] = requests_num;

        //Timing variables
        PerfMeasure timing_app, timing_mm, timing_total, timing_total_sync;

        for (int iteration = 0; iteration < num_iterations; ++iteration){

            *exit_signal = 0;
            *exit_counter = 0;
            RequestType requests;
            requests.init(requests_num);
            requests.memset();

            //Run presistent kernel (Memory Manager)
            timing_mm.startMeasurement();
            mem_manager<<<mm_grid_size, block_size, 0, mm_stream>>>(exit_signal, requests.requests_number, 
                    requests.request_iter, requests.request_signal, requests.request_id,
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    memory_manager,
#endif
#endif
                    requests.d_memory, requests.request_mem_size, requests.lock, turn_on);
            timing_mm.stopMeasurement();
            GUARD_CU(cudaPeekAtLastError());

            //Run application
            timing_app.startMeasurement();
            timing_total.startMeasurement();
            timing_total_sync.startMeasurement();
            app<<<app_grid_size, block_size, 0, app_stream>>>(exit_signal, requests.d_memory, 
                    requests.request_signal, requests.request_mem_size, requests.request_id, 
                    exit_counter, requests.lock, size_to_alloc, turn_on);
            timing_app.stopMeasurement();
            GUARD_CU(cudaPeekAtLastError());

            // Check results
            int old_counter = -1;
            long long iter = 0;
            long long iter2 = 0;
            long long iter_mean = 0;
            long long  time_limit = 100000000;
            while (iter2 < time_limit){
                if (exit_counter[0] == block_size*app_grid_size){
                    timing_total.stopMeasurement();
                    *exit_signal = 1;
                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                    timing_total_sync.stopMeasurement();
                    if (turn_on){
                        //test1<<<app_grid_size, block_size, 0, app_stream>>>(requests.d_memory);
                        test1<<<app_grid_size, block_size>>>(requests.d_memory);
                        GUARD_CU(cudaDeviceSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                        //mem_test((int**)requests.d_memory, requests_num, app_grid_size, block_size, mm_stream);
                        mem_test((int**)requests.d_memory, requests_num, app_grid_size, block_size);
                        GUARD_CU(cudaDeviceSynchronize());
                        GUARD_CU(cudaPeekAtLastError());
                    }
                    printf("break!\n");
                    break;
                }else{
                    if (exit_counter[0] != old_counter){
                        old_counter = exit_counter[0];
                        ++iter;
                        iter_mean += iter2;
                        iter2 = 0;
                        //if (iter%1000 == 0){}
                    }
                    ++iter2;
                }
                if (iter2 >= time_limit){
                    printf("time limit exceed, break\n");
                    *exit_signal = 1;
                    GUARD_CU(cudaDeviceSynchronize());
                    GUARD_CU(cudaPeekAtLastError());
                }
            }
            if (iter != 0)
                iter_mean /= iter;
         
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            //Deallocate device memory
            mem_free<<<app_grid_size, block_size>>>(requests.d_memory, requests.request_id, 
#ifdef OUROBOROS__
                    memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
                    memory_manager,
#endif
#endif
                    requests.requests_number);

            GUARD_CU(cudaPeekAtLastError());
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            requests.free();
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
        }
        // Output
        auto app_time = timing_app.generateResult();
        auto total_time = timing_total.generateResult();
        auto total_sync_time = timing_total_sync.generateResult();
        app_launch[app_grid_size - 1] = (app_time.mean_);
        app_finish[app_grid_size - 1] = (total_time.mean_);
        app_sync  [app_grid_size - 1] = (total_sync_time.mean_);
        // The number of requests done per a second
        uni_req_num[app_grid_size - 1] = (requests_num * 1000.0)/total_sync_time.mean_;

        printf("\t\t%d\t\t| %d\t\t| %d\t\t| %.2lf\t\t| %.2lf\n", requests_num, 
            app_grid_size, mm_grid_size, uni_req_num[app_grid_size - 1], total_sync_time.mean_);
    //}

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
}

}
