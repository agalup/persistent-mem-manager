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
        volatile int* request_counter,
        volatile int* request_ids, 
        volatile int** request_dest, 
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){

    // SEMAPHORE
    acquire_semaphore((int*)lock, request_id);
    debug("mm: request recieved %d\n", request_id); 
    auto addr_id = request_ids[request_id];
    __threadfence();
    
    switch (request_signal[request_id]){

        case MALLOC:
            if (addr_id == -1){
                addr_id = atomicAdd((int*)&request_counter[0], 1);
                request_ids[request_id] = addr_id;
            }else{
                assert(d_memory[addr_id] == NULL);
            }
            d_memory[addr_id] = reinterpret_cast<volatile int*>
                (mm->malloc(4+request_mem_size[request_id]));
            __threadfence();
            assert(d_memory[addr_id]);
            d_memory[addr_id][0] = 0;
            request_dest[request_id] = &d_memory[addr_id][1];
            atomicExch((int*)&request_signal[request_id], request_done);
            break;

        case FREE:
            assert(d_memory[addr_id]);
            if (d_memory[addr_id][0] != 0)
                printf("d_memory{%d} = %d\n", addr_id, d_memory[addr_id][0]);
            assert(d_memory[addr_id][0] == 0);
            auto request_status = d_memory[addr_id][0] - 1;
            d_memory[addr_id][0] -= 1;
            request_dest[request_id] = NULL;
            assert(d_memory[addr_id][0] == -1);
            if (request_status < 0){
                atomicExch((int*)&request_signal[request_id], request_gc);
            }else{
                assert(1);
                printf("should not be here!\n");
                atomicExch((int*)&request_signal[request_id], request_done);
            }
            break;

        case GC:
            assert(d_memory[addr_id]);
            assert(d_memory[addr_id][0] == -1);
            mm->free((void*)d_memory[addr_id]);
            d_memory[addr_id] = NULL;
            atomicExch((int*)&request_signal[request_id], request_done);
            break;

        default:
            printf("request processing fail\n");

    }
    __threadfence();

    release_semaphore((int*)lock, request_id);
    // SEMAPHORE
}

__global__
void garbage_collector(volatile int** d_memory,
                       volatile int* requests_number, 
                       volatile int* request_counter,
                       volatile int* request_signal, 
                       volatile int* request_ids, 
                       volatile int* request_mem_size,
                       volatile int* lock,
                       volatile int* exit_signal,
                       MemoryManagerType* mm){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __threadfence();
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){
            if (request_signal[request_id] == GC){
                _request_processing(request_id, exit_signal, request_signal,
                                    request_counter, request_ids, NULL, mm, 
                                    d_memory, request_mem_size, lock);
            }
        }
    }
}


//producer
__global__
void mem_manager(volatile int* exit_signal, 
        volatile int* requests_number, 
        volatile int* request_counter,
        volatile int* request_signal, 
        volatile int* request_ids, 
        volatile int** request_dest,
        MemoryManagerType* mm, 
        volatile int** d_memory,
        volatile int* request_mem_size,
        volatile int* lock){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; !exit_signal[0] && request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            if (request_signal[request_id] == MALLOC or 
                request_signal[request_id] == FREE){
                _request_processing(request_id, exit_signal, request_signal, 
                                    request_counter, request_ids, request_dest,
                                    mm, d_memory, request_mem_size, lock);
            
            debug("mm: request done %d\n", request_id);
            }
        }
    }
}

__device__
void post_request(request_type type,
                  volatile int** dest,
                  volatile int* lock,
                  volatile int* request_mem_size,
                  volatile int* request_id,
                  volatile int* request_signal,
                  volatile int** request_dest,
                  volatile int* exit_signal,
                  int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    if (type == MALLOC){
        request_mem_size[thid] = size_to_alloc;
    }
    __threadfence();
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
                      volatile int** dest,
                      volatile int* request_signal,
                      volatile int** request_dest,
                      int& req_id){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    switch (type){
        case MALLOC:
            req_id = request_id[thid];
            if (req_id >= 0 && !exit_signal[0]) {
                __threadfence();
                *dest = request_dest[thid];
                assert(d_memory[req_id] != NULL);
                if (d_memory[req_id][0] != 0)
                    printf("d_memory[%d] = %d\n", req_id, d_memory[req_id][0]);
                //assert(d_memory[req_id][0] == 0);
                assert(*dest != NULL);
                assert(request_dest[thid] == *dest);
            }
            break;
        case FREE:
            assert(d_memory[req_id] == NULL);
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
    release_semaphore((int*)lock, thid);
    // SEMAPHORE
}

__device__
void request(request_type type,
        volatile int* exit_signal,
        volatile int** d_memory,
        volatile int** dest,
        volatile int* request_signal,
        volatile int* request_mem_size, 
        volatile int* request_id,
        volatile int** request_dest,
        volatile int* lock,
        int size_to_alloc){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int req_id = -1;
    // wait for success
    while (!exit_signal[0]){
        __threadfence();
        if (request_signal[thid] == request_empty){
            post_request(type, dest, lock, request_mem_size, request_id, request_signal, request_dest, exit_signal, size_to_alloc);
            break;
        }
    }

    // wait for success
    while (!exit_signal[0]){
        __threadfence();
        if (request_signal[thid] == request_done){
            request_processed(type, lock, request_id, exit_signal, d_memory, dest, request_signal, request_dest, req_id);
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
        volatile int** request_dest, 
        volatile int* exit_counter, 
        volatile int* lock,
        int size_to_alloc,
        int iter_num){

    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i=0; i<iter_num; ++i){

        volatile int* new_ptr = NULL;
        request((request_type)MALLOC, exit_signal, d_memory, &new_ptr, 
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc);
        new_ptr[0] = thid;

        assert(d_memory[request_id[thid]]);
        int value = d_memory[request_id[thid]][0];
        if (value != 0) printf("val = %d\n", value);
        assert(new_ptr[0] == thid);

        request((request_type)FREE, exit_signal, d_memory, &new_ptr,
                request_signal, request_mem_size, request_id, request_dest,
                lock, size_to_alloc);
    }
    
    atomicAdd((int*)&exit_counter[0], 1);
}

//consumer2
__global__
void free_app_test(volatile int* exit_signal, 
              volatile int** d_memory, 
              volatile int* request_signal, 
              volatile int* request_mem_size,
              volatile int* request_id, 
              volatile int** request_dest, 
              volatile int* exit_counter, 
              volatile int* lock,
              int size_to_alloc,
              int iter_num){
    
    __threadfence();
   
    request((request_type)FREE, exit_signal, d_memory, NULL, 
            request_signal, request_mem_size, request_id, request_dest,
            lock, 0);

    atomicAdd((int*)&exit_counter[0], 1);
}

void check_persistent_kernel_results(int* exit_signal, 
                   int* exit_counter, 
                   int block_size, 
                   int app_grid_size, 
                   cudaStream_t& app_stream,
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
            GUARD_CU(cudaPeekAtLastError());
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
    debug("all requests are posted\n");

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
                          MemoryManagerType& memory_manager){

    timing_mm.startMeasurement();
    mem_manager<<<mm_grid_size, block_size, 0, mm_stream>>>(exit_signal, 
            requests.requests_number, requests.request_counter, requests.request_signal, 
            requests.request_id, requests.request_dest,
#ifdef OUROBOROS__
            memory_manager.getDeviceMemoryManager(),
#else
#ifdef HALLOC__
            memory_manager,
#endif
#endif
            requests.d_memory, requests.request_mem_size, requests.lock);
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
            requests.request_counter,
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
                       int iter_num,
                       bool& kernel_complete){
    // Run application
    timing_launch.startMeasurement();
    timing_sync.startMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    auto kernel = malloc_app_test;
    if (type == FREE){
        kernel = free_app_test;
    }
    debug("start kernel\n");
    kernel<<<grid_size, block_size, 0, stream>>>(exit_signal, requests.d_memory, 
            requests.request_signal, requests.request_mem_size, 
            requests.request_id, requests.request_dest, exit_counter, requests.lock, 
            size_to_alloc, iter_num);
    timing_launch.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    debug("done\n");
    fflush(stdout);


    // Check resutls: test
    debug("check results\n");
    check_persistent_kernel_results(exit_signal, exit_counter, block_size, 
            grid_size, stream, requests, requests.size, kernel_complete);
    timing_sync.stopMeasurement();
    GUARD_CU(cudaPeekAtLastError());
    debug("results done\n");
    fflush(stdout);

    if (kernel_complete){
        if (type == MALLOC){
            /*printf("test1!\n");
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            test1<<<grid_size, block_size, 0, stream>>>(requests.d_memory, 
                                                        requests.size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            mem_test((int**)requests.d_memory, requests.size, grid_size, block_size);
            GUARD_CU(cudaStreamSynchronize(stream));
            GUARD_CU(cudaPeekAtLastError());
            printf("test done\n");*/
        }else if (type == FREE){
            debug("test2!\n");
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

    debug("waiting for streams\n");
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("app stream synced\n");
    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("mm stream synced\n");
    GUARD_CU(cudaStreamSynchronize(gc_stream));
    GUARD_CU(cudaPeekAtLastError());
    debug("gc stream synced\n");
    GUARD_CU(cudaPeekAtLastError());

}


void pmm_init(int kernel_iteration_num, int size_to_alloc, size_t* ins_size, 
              size_t num_iterations, int SMs, int* sm_app, int* sm_mm, int* sm_gc, 
              int* allocs, float* uni_req_per_sec, int* array_size){

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
    printf("size to alloc per thread %d, num iterations %d, kernel iterations %d, instantsize %ld\n", 
                size_to_alloc, num_iterations, kernel_iteration_num, instant_size);
    std::cout << "#requests\t" << "#sm app\t\t" << "#sm mm\t\t" << "#sm gc\t\t" <<
                "#malloc and free per sec\n";

    //int gc_grid_size = 1;

    int it = 0;

    for (int app_grid_size = 1; app_grid_size < SMs; ++app_grid_size){
    //for (int app_grid_size = 1; app_grid_size < 5; ++app_grid_size){

    for (int mm_grid_size = 1; mm_grid_size < (SMs - app_grid_size); ++mm_grid_size){
    //for (int mm_grid_size = 1; mm_grid_size < 5; ++mm_grid_size){

        int gc_grid_size = SMs - app_grid_size - mm_grid_size;
        if (gc_grid_size <= 0)
            continue;

        debug("SMs: app %d, mm %d, gc %d, total %d\n", app_grid_size, mm_grid_size, gc_grid_size, SMs);
        int requests_num{app_grid_size*block_size};

        //output
        sm_app[it] = app_grid_size;
        sm_mm [it] = mm_grid_size;
        sm_gc [it] = gc_grid_size;
        allocs[it] = requests_num;

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
                                 exit_signal, requests, memory_manager);

            
            // Run Garbage Collector (persistent kernel)
            start_garbage_collector(timing_gc, gc_grid_size, block_size, gc_stream,
                                 exit_signal, requests, memory_manager);
            

            // Run APP (all threads do malloc)
            bool kernel_complete = false;
            start_application(MALLOC, timing_malloc_app, malloc_total_sync, 
                              app_grid_size, block_size, app_stream, exit_signal,
                              requests, exit_counter, size_to_alloc, 
                              kernel_iteration_num, kernel_complete);

            if (not kernel_complete){
                // Deallocate device memory
                clean_memory(app_grid_size, block_size, requests, memory_manager,
                             exit_signal);
                continue;
            }

     /*       // Run APP (all threads do free)
            *exit_counter = 0;
            kernel_complete = false;
            start_application(FREE, timing_free_app, free_total_sync, 
                              app_grid_size, block_size, app_stream, exit_signal,
                              requests, exit_counter, 0, kernel_iteration_num, kernel_complete);*/

            *exit_signal = 1;
            sync_streams(app_stream, mm_stream, gc_stream);

            // Deallocate device memory
            clean_memory(app_grid_size, block_size, requests, memory_manager, exit_signal);

        }
        // Output
        //auto malloc_time            = timing_malloc_app.generateResult();
        //auto free_time              = timing_free_app.generateResult();
        auto malloc_total_sync_time = malloc_total_sync.generateResult();
        //auto free_total_sync_time   = free_total_sync.generateResult();

        //malloc_sync[app_grid_size - 1]      = (malloc_total_sync_time.mean_);
        //free_sync[app_grid_size - 1]        = (free_total_sync_time.mean_);
        // The number of requests done per a second
        uni_req_per_sec[it]   = (requests_num * 1000.0)/malloc_total_sync_time.mean_;
        //free_per_sec[app_grid_size - 1]     = (requests_num * 1000.0)/free_total_sync_time.mean_;

        printf("  %d\t\t %d\t\t %d\t\t %d\t\t %.2lf\t\t \n", requests_num, 
            app_grid_size, mm_grid_size, gc_grid_size, 
            uni_req_per_sec[it]);


        ++it;
    }
    }
    
    *array_size = it;
    //printf("it %d/ SMs %d\n", it, SMs);
    //assert(it == SMs);

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
}

}
