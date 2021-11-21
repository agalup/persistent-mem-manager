#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>
#include <any>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
#include "PerformanceMeasure.cuh"
#include "Utility.h"
#include "cuda.h"
#include "pmm-utils.cuh"
//#include "src/gpu_hash_table.cuh"

using namespace std;

extern "C" {

__global__
    void mem_free(volatile int** d_memory, 
            volatile int* request_id, 
            MemoryManagerType* mm, 
            volatile int* requests_num){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num[0]){
        return;
    }
    if (d_memory[thid]){
        printf("sync error: %d was not released before\n", thid);
        mm->free((void*)d_memory[thid]);
    }
}

__device__
void _request_processing(
        int request_id, 
        volatile int* exit_signal,
        volatile int* request_signal,
        volatile int* request_iter,
        volatile int* request_ids, 
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock, 
        int call_on
        ){

    // SEMAPHORE
    acquire_semaphore((int*)lock, request_id);
    debug("mm: request recieved %d\n", request_id); 
    
    if (call_on){
        switch (request_signal[request_id]){

            case MALLOC:
                int req_id = atomicAdd((int*)&request_iter[0], 1);
                request_ids[request_id] = req_id;
                d_memory[req_id] = reinterpret_cast<volatile int*>
                    (mm->malloc(4+request_mem_size[request_id]));
                __threadfence();
                if (!exit_signal[0]){
                    if (!d_memory[req_id]){
                        printf("request failed\n");
                        assert(d_memory[req_id]);
                    }
                }
                // SIGNAL update
                atomicExch((int*)&request_signal[request_id], request_done);
                break;

            case FREE:
                assert(d_memory[request_ids[request_id]]);
                auto request_status = d_memory[request_ids[request_id]][0] - 1;
                d_memory[request_ids[request_id]][0] -= 1;
                /*printf("d_memory[%d] = %d\n", request_ids[request_id], 
                    d_memory[request_ids[request_id]]);*/
                //Moved to GC:
                //mm->free((void*)d_memory[request_ids[request_id]]);
                // SIGNAL update
                if (request_status < 0){
                    atomicExch((int*)&request_signal[request_id], request_gc);
                }else{
                    atomicExch((int*)&request_signal[request_id], request_done);
                }
                __threadfence();
                break;

            case GC:
                auto addr_id = request_ids[request_id];
                if (d_memory[addr_id] != NULL){
                    __threadfence();
                    if (d_memory[addr_id][0] == -1){
     //                   printf(". %d\n", addr_id);
                        mm->free((void*)d_memory[addr_id]);
                        d_memory[addr_id] = NULL;
                    }
                }
                atomicExch((int*)&request_signal[request_id], request_done);
                break;

            default:
                printf("request processing fail\n");
        }
    }

    release_semaphore((int*)lock, request_id);
    // SEMAPHORE
}

__global__
void garbage_collector(volatile int** d_memory,
                       volatile int* requests_number, 
                       volatile int* request_iter,
                       volatile int* request_signal, 
                       volatile int* request_ids, 
                       volatile int* request_mem_size,
                       volatile int* lock,
                       volatile int* exit_signal,
                       //const int size,
                       MemoryManagerType* mm){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __threadfence();
    while (! exit_signal[0]){
        //for (int addr_id = thid; !exit_signal[0] && addr_id < size; 
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                //addr_id += (gridDim.x * blockDim.x)){
                request_id += blockDim.x*gridDim.x){
            if (request_signal[request_id] == GC){
                //printf("%d %d\n", request_id, d_memory[request_id][0]);
                _request_processing(request_id, exit_signal, request_signal,
                                    request_iter, request_ids, mm, d_memory, 
                                    request_mem_size, lock, 1);
            }
        }
    }
}


//producer
__global__
void mem_manager(volatile int* exit_signal, 
        volatile int* requests_number, 
        volatile int* request_iter,
        volatile int* request_signal, 
        volatile int* request_ids, 
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock, 
        int call_on){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            if (request_signal[request_id] == MALLOC or 
                request_signal[request_id] == FREE){
                _request_processing(request_id, exit_signal, request_signal, 
                                    request_iter, request_ids, mm, d_memory, 
                                    request_mem_size, lock, call_on);
            
            debug("mm: request done %d\n", request_id);
            }
        }
    }
}

__device__
void post_request(request_type type, 
                  volatile int* lock,
                  volatile int* request_mem_size,
                  volatile int* request_id,
                  volatile int* request_signal,
                  int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    if (type == MALLOC){
        request_mem_size[thid] = size_to_alloc;
        request_id[thid] = -1;
    }
    // SIGNAL update
    atomicExch((int*)&request_signal[thid], type);
    __threadfence();
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
}

__device__
void request_processed(request_type type,
                      volatile int* lock,
                      volatile int* request_id,
                      volatile int* exit_signal,
                      volatile int** d_memory,
                      volatile int** new_ptr,
                      volatile int* request_signal,
                      int& req_id,
                      int call_on
                      ){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    switch (type){
        case MALLOC:
            req_id = request_id[thid];
            if (req_id >= 0 && call_on && !exit_signal[0]) {
                assert(d_memory[req_id]);
                d_memory[req_id][0] = 0;
                *new_ptr = &d_memory[req_id][1];
            }
            break;
        case FREE:
            //d_memory[request_id[thid]] = NULL;
            break;
        case GC:
            assert(d_memory[req_id] == NULL);
            break;
        default:
            printf("error\n");
            break;
    }
    request_signal[thid] = request_empty;
    __threadfence();
    debug("app: request %d success\n", thid);
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
}

__device__
void request(request_type type,
        volatile int* exit_signal,
        volatile int** d_memory,
        volatile int** new_ptr,
        volatile int* request_signal,
        volatile int* request_mem_size, 
        volatile int* request_id,
        volatile int* lock,
        int size_to_alloc,
        int call_on
        ){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    post_request(type, lock, request_mem_size, request_id, request_signal, size_to_alloc);

    // wait for success
    while (!exit_signal[0]){
        __threadfence();
        if (request_signal[thid] == request_done){
            request_processed(type, lock, request_id, exit_signal, d_memory, new_ptr, request_signal, req_id, call_on);
            if (type == MALLOC)
                assert(d_memory[request_id[thid]]);
            break;
        }
    }
}

//consumer
__global__
void malloc_app_test(volatile int* exit_signal,
        volatile int** d_memory, 
        volatile int* request_signal, 
        volatile int* request_mem_size,
        volatile int* request_id, 
        volatile int* exit_counter, 
        volatile int* lock,
        int size_to_alloc,
        int call_on){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
   
    int* new_ptr;
    request((request_type)MALLOC, exit_signal, d_memory, &new_ptr, 
            request_signal, request_mem_size, request_id, lock, 
            size_to_alloc, call_on);

    new_ptr[0] = thid;

    assert(d_memory[request_id[thid]]);
    int value = d_memory[request_id[thid]][0];
    if (value != 0) printf("val = %d\n", value);
    //assert(d_memory[request_id[thid]][0] == 0);
    assert(new_ptr[0] == thid);
    
    atomicAdd((int*)&exit_counter[0], 1);
}

//consumer2
__global__
void free_app_test(volatile int* exit_signal, 
              volatile int** d_memory, 
              volatile int* request_signal, 
              volatile int* request_mem_size,
              volatile int* request_id, 
              volatile int* exit_counter, 
              volatile int* lock,
              int size_to_alloc,
              int call_on){
    
    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    __threadfence();
   
    int* new_ptr;
    request((request_type)FREE, exit_signal, d_memory, &new_ptr, 
            request_signal, request_mem_size, request_id, lock, 
            size_to_alloc, call_on);

    atomicAdd((int*)&exit_counter[0], 1);
}

void check_persistent_kernel_results(int* exit_signal, 
                   int* exit_counter, 
                   int block_size, 
                   int app_grid_size, 
                   cudaStream_t& app_stream,
                   int call_on, 
                   RequestType& requests, 
                   int requests_num,
                   bool& finish){

    // Check results
    int old_counter = -1;
    long long iter = 0;
    long long time_limit = 10000000000;
    //printf("start\n");
    while (iter < time_limit){
        // Check if all allocations are done
        if (exit_counter[0] == block_size*app_grid_size){
            GUARD_CU(cudaStreamSynchronize(app_stream));
            GUARD_CU(cudaPeekAtLastError());
            finish = true;
            break;
        }else{
            if (exit_counter[0] != old_counter){
                old_counter = exit_counter[0];
                //printf("%d\n", old_counter);
                iter = 0;
            }
            ++iter;
        }
        if (iter >= time_limit){
            // Start mm and app again
            printf("time limit exceed, break\n");
            fflush(stdout);
            *exit_signal = 1;
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
        }
    }
    printf("all requests are posted\n");

    GUARD_CU(cudaPeekAtLastError());
}

void createStreams(cudaStream_t& gc_stream, 
                   cudaStream_t& mm_stream, 
                   cudaStream_t& app_stream){
    GUARD_CU(cudaStreamCreateWithFlags( &gc_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaStreamCreateWithFlags( &mm_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaPeekAtLastError());
}

void allocManaged(int** ptr, size_t size){
    GUARD_CU(cudaMallocManaged(ptr, size));
    GUARD_CU(cudaPeekAtLastError());
    GUARD_CU(cudaDeviceSynchronize());
}

void start_memory_manager(PerfMeasure& timing_mm, 
                          uint32_t mm_grid_size,
                          uint32_t block_size, 
                          cudaStream_t& mm_stream,
                          int* exit_signal,
                          RequestType& requests,
                          MemoryManagerType& memory_manager,
                          int call_on){

    timing_mm.startMeasurement();
    mem_manager<<<mm_grid_size, block_size, 0, mm_stream>>>(exit_signal, 
            requests.requests_number, 
            requests.request_iter, requests.request_signal, requests.request_id,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
            memory_manager,
#endif
#endif
            requests.d_memory, requests.request_mem_size, requests.lock, call_on);
    timing_mm.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());

}

void start_garbage_collector(PerfMeasure& timing_gc, 
                          uint32_t gc_grid_size,
                          uint32_t block_size, 
                          cudaStream_t& gc_stream,
                          int* exit_signal,
                          RequestType& requests,
                          MemoryManagerType& memory_manager){
    timing_gc.startMeasurement();
    garbage_collector<<<gc_grid_size, block_size, 0, gc_stream>>>(
            requests.d_memory, 
            requests.requests_number,
            requests.request_iter,
            requests.request_signal,
            requests.request_id,
            requests.request_mem_size, 
            requests.lock,
            exit_signal,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager()
#else
#ifdef HALLOC__
            memory_manager
#endif
#endif
            );
    timing_gc.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());

}

void clean_memory(uint32_t grid_size,
                  uint32_t block_size, 
                  RequestType& requests,
                  MemoryManagerType& memory_manager,
                  int* exit_signal){

    *exit_signal = 1;
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
    mem_free<<<grid_size, block_size>>>(requests.d_memory, 
            requests.request_id, 
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
    requests.free();
    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}


void start_application(int type, 
                       PerfMeasure& timing_launch, 
                       PerfMeasure& timing_sync, 
                       uint32_t grid_size,
                       uint32_t block_size, 
                       cudaStream_t& stream,
                       int* exit_signal,
                       RequestType& requests,
                       int* exit_counter,
                       int size_to_alloc, 
                       int call_on,
                       bool& kernel_complete){
    // Run application
    timing_launch.startMeasurement();
    timing_sync.startMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    auto kernel = malloc_app_test;
    if (type == FREE){
        kernel = free_app_test;
    }
    printf("start kernel\n");
    kernel<<<grid_size, block_size, 0, stream>>>(exit_signal, requests.d_memory, 
            requests.request_signal, requests.request_mem_size, 
            requests.request_id, exit_counter, requests.lock, size_to_alloc, 
            call_on);
    timing_launch.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    printf("done\n");
    fflush(stdout);


    // Check resutls: test
    printf("check results\n");
    check_persistent_kernel_results(exit_signal, exit_counter, block_size, 
            grid_size, stream, call_on, requests, requests.size, kernel_complete);
    timing_sync.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    printf("results done\n");
    fflush(stdout);

    if (kernel_complete){
        if (type == MALLOC && call_on){
            printf("test1!\n");
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            test1<<<grid_size, block_size, 0, stream>>>(requests.d_memory, 
                                                        requests.size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            mem_test((int**)requests.d_memory, requests.size, grid_size, block_size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            printf("test done\n");
        }else if (type == FREE){
            printf("test2!\n");
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            test2<<<grid_size, block_size, 0, stream>>>(requests.d_memory, 
                                                        requests.size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
        }
    }
}

void sync_streams(cudaStream_t& gc_stream, 
                  cudaStream_t& mm_stream, 
                  cudaStream_t& app_stream){

    printf("waiting for streams\n");
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
    printf("app stream synced\n");
    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaPeekAtLastError());
    printf("mm stream synced\n");
    GUARD_CU(cudaStreamSynchronize(gc_stream));
    GUARD_CU(cudaPeekAtLastError());
    printf("gc stream synced\n");
    GUARD_CU(cudaPeekAtLastError());

}


void pmm_init(int call_on, int size_to_alloc, size_t* ins_size, 
              size_t num_iterations, int SMs, int* sm_app, int* sm_mm, int* sm_gc, 
              int* allocs, float* malloc_sync, float* malloc_per_sec, 
              float* free_sync, float* free_per_sec){

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
    cudaStream_t gc_stream, mm_stream, app_stream;
    createStreams(gc_stream, mm_stream, app_stream);
    
    int* exit_signal;
    allocManaged(&exit_signal, sizeof(int32_t));

    int* exit_counter;
    allocManaged(&exit_counter, sizeof(uint32_t));

    int block_size = 1024;
    printf("size to alloc per thread %d, num iterations %d, instantsize %ld\n", 
                size_to_alloc, num_iterations, instant_size);
    std::cout << "\t\t#allocs\t\t" << "#sm app\t\t" << "#sm mm\t\t" << 
                "#malloc req per sec\t" << "malloc\t\t" << "#free req per sec\t" << "free\n";

    int gc_grid_size = 1;

    for (int app_grid_size = 1; app_grid_size < (SMs - gc_grid_size); ++app_grid_size){

        int mm_grid_size = SMs - gc_grid_size - app_grid_size;

        printf("SMs: gc %d, mm %d, app %d, total %d\n", gc_grid_size, mm_grid_size, app_grid_size, SMs);
        int requests_num{app_grid_size*block_size};

        //output
        sm_app[app_grid_size - 1] = app_grid_size;
        sm_mm [app_grid_size - 1] = mm_grid_size;
        sm_gc [app_grid_size - 1] = gc_grid_size;
        allocs[app_grid_size - 1] = requests_num;

        //Timing variables
        PerfMeasure timing_malloc_app, timing_free_app, timing_mm, timing_gc, 
                    malloc_total_sync, free_total_sync;

        for (int iteration = 0; iteration < num_iterations; ++iteration){

            *exit_signal = 0;
            *exit_counter = 0;
            RequestType requests;
            requests.init(requests_num);
            requests.memset();
            
            GUARD_CU(cudaPeekAtLastError());

            // Run Memory Manager (Presistent kernel)
            start_memory_manager(timing_mm, mm_grid_size, block_size, mm_stream,
                                 exit_signal, requests, memory_manager, call_on);

            
            // Run Garbage Collector (persistent kernel)
            start_garbage_collector(timing_gc, gc_grid_size, block_size, gc_stream,
                                 exit_signal, requests, memory_manager);
            

            // Run APP (all threads do malloc)
            bool kernel_complete = false;
            start_application(MALLOC, timing_malloc_app, malloc_total_sync, 
                              app_grid_size, block_size, app_stream, exit_signal,
                              requests, exit_counter, size_to_alloc, call_on,
                              kernel_complete);

            if (not kernel_complete){
                // Deallocate device memory
                clean_memory(app_grid_size, block_size, requests, memory_manager,
                             exit_signal);
                continue;
            }

            // Run APP (all threads do free)
            *exit_counter = 0;
            kernel_complete = false;
            start_application(FREE, timing_free_app, free_total_sync, 
                              app_grid_size, block_size, app_stream, exit_signal,
                              requests, exit_counter, 0, call_on, kernel_complete);

            *exit_signal = 1;
            sync_streams(app_stream, mm_stream, gc_stream);

            // Deallocate device memory
            clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);

        }
        // Output
        auto malloc_time            = timing_malloc_app.generateResult();
        auto free_time              = timing_free_app.generateResult();
        auto malloc_total_sync_time = malloc_total_sync.generateResult();
        auto free_total_sync_time   = free_total_sync.generateResult();

        malloc_sync[app_grid_size - 1]      = (malloc_total_sync_time.mean_);
        free_sync[app_grid_size - 1]        = (free_total_sync_time.mean_);
        // The number of requests done per a second
        malloc_per_sec[app_grid_size - 1]   = (requests_num * 1000.0)/malloc_total_sync_time.mean_;
        free_per_sec[app_grid_size - 1]     = (requests_num * 1000.0)/free_total_sync_time.mean_;

        printf("\t\t%d\t\t| %d\t\t| %d\t\t| %.2lf\t\t| %.2lf\t\t| %.2lf\t\t | %.2lf\n", requests_num, 
            app_grid_size, mm_grid_size, malloc_per_sec[app_grid_size - 1], malloc_total_sync_time.mean_,
            free_per_sec[app_grid_size - 1], free_total_sync_time.mean_);
    }

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
}

}
