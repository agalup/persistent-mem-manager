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
#include "Instance.cuh"

using namespace std;

//#define DEBUG
#ifdef DEBUG
#define debug(a...) printf(a)
#else
#define debug(a...)
#endif


//producer
template <typename MemoryManagerType>
__global__
void mem_manager(volatile int* exit_signal, 
                volatile int* requests_number, 
                volatile int* request_iter,
                volatile int* request_signal, 
                volatile int* request_ids, 
                //MemoryManagerType* mm,
                MemoryManagerType mm,
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
                        (mm.malloc(request_mem_size[request_id]));
                        //(mm->malloc(request_mem_size[request_id]));
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

__global__
void mem_free(volatile int** d_memory){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    /*TODO*/
}


int main(int argc, char *argv[]){

    //Ouroboros initialization
    //size_t instantitation_size = 7168ULL * 1024ULL * 1024ULL;
    //using MemoryMangerType = OuroPQ;
    //MemoryMangerType memory_manager;
    //memory_manager.initialize(instantitation_size);

    //Halloc initialization
    size_t instantitation_size = 2048ULL * 1024ULL * 1024ULL;
    using MemoryManagerType = MemoryManagerHalloc;
    MemoryManagerType memory_manager(instantitation_size);
    
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

    for (int mm_grid_size = 1; mm_grid_size < max_block_number; ++mm_grid_size){

        *exit_signal = 0;
        *exit_counter = 0;

        int app_grid_size = max_block_number - mm_grid_size;

        int requests_num{app_grid_size*block_size};
        std::cout << "Number of Allocations: " << requests_num << "\n";
        std::cout << "APP: " << app_grid_size << ", " << block_size << "\n";
        std::cout << "MM:  " << mm_grid_size << ", " << block_size << "\n";

        //Request auxiliary
        RequestType requests;
        requests.init(requests_num);
        requests.memset();

        //Timing variables
        PerfMeasure timing_app, timing_mm, timing_total;

        printf("mm starts\n");
        timing_mm.startMeasurement();
        //Run presistent kernel (Memory Manager)
        mem_manager<<<mm_grid_size, block_size, 0, mm_stream>>>(exit_signal,
                requests.requests_number, 
                requests.request_iter, 
                requests.request_signal, 
                requests.request_id,
                //memory_manager.getDeviceMemoryManager(),
                memory_manager,
                requests.d_memory,
                requests.request_mem_size,
                requests.lock, 
                ouroboros_on);
        timing_mm.stopMeasurement();

        GUARD_CU(cudaPeekAtLastError());
        printf("mm stop\n");

        printf("app start\n");
        timing_app.startMeasurement();
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
        //timing_app.stopMeasurement();

        GUARD_CU(cudaPeekAtLastError());
        printf("app stop\n");

        // Check results
        int old_counter = -1;
        int iter = 0;
        while (1){
            if (exit_counter[0] == block_size*app_grid_size){
                *exit_signal = 1;
                GUARD_CU(cudaDeviceSynchronize());
                GUARD_CU(cudaPeekAtLastError());
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
                    if (iter%1000 == 0)
                        //debug("no break, exit_counter = %d\n", exit_counter[0]);
                        printf("no break, exit_counter = %d\n", exit_counter[0]);
                    ++iter;
                }
            }
        }

        //Deallocate device memory
        mem_free<<<app_grid_size, block_size, 0, app_stream>>>(
                requests.d_memory);
        requests.free();
    }

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    GUARD_CU(cudaPeekAtLastError());
    printf("DONE!\n");
    return 0;
}

