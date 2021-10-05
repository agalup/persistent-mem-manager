#include <iostream>
#include <string>
#include <cassert>

#include "device/Ouroboros_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "InstanceDefinitions.cuh"
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


//producer
template <typename MemoryManagerType>
__global__
void mem_manager(volatile int* exit_signal, 
                volatile int* requests_number, 
                volatile int* request_iter,
                volatile int* request_signal, 
                volatile int* request_ids, 
                MemoryManagerType* mm,
                volatile int** d_memory,
                volatile int* request_mem_size,
                volatile int* lock){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (! exit_signal[0]){
        for (int request_id=thid; request_id<requests_number[0]; 
                request_id += blockDim.x*gridDim.x){

            if (request_signal[request_id] == 1){

                // SEMAPHORE
                acquire_semaphore((int*)lock, request_id);
                printf("mm: request recieved %d\n", request_id); 
                int req_id = atomicAdd((int*)&request_iter[0], 1);
                request_ids[request_id] = req_id;

                //__threadfence();
                //d_memory[req_id] = reinterpret_cast<volatile int*>
                //        (mm->malloc(request_mem_size[request_id]));
                //__threadfence();
                //assert(d_memory[req_id]);

                // SIGNAL update
                atomicExch((int*)&request_signal[request_id], 2);

                //__threadfence();
                release_semaphore((int*)lock, request_id);
                // SEMAPHORE

                printf("mm: request done %d\n", request_id, req_id);
                //break;
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
         volatile int* lock){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    // SEMAPHORE
    acquire_semaphore((int*)lock, thid);
    request_mem_size[thid] = 4;
    request_id[thid] = -1;
    int req_id = -1;
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
            if (req_id >= 0) {
                //assert(d_memory[req_id]);
                //d_memory[req_id][0] = thid;
            }
            request_signal[thid] = 0;
            __threadfence();
            //printf("app: request %d success\n", thid);
            release_semaphore((int*)lock, thid);
            // SEMAPHORE
        
            //printf("done request(%d) = %d, request_id = %d\n", thid, req_id, request_id[thid]);
            break;
        }
    }
    atomicAdd((int*)&exit_counter[0], 1);
    
}

int main(int argc, char *argv[]){

    //Ouroboros initialization
    size_t instantitation_size = 7168ULL * 1024ULL * 1024ULL;
    using MemoryMangerType = OuroVACQ;
    MemoryMangerType memory_manager;
    memory_manager.initialize(instantitation_size);

//Creat two asynchronous streams which may run concurrently with the default stream 0.
    //The streams are not synchronized with the default stream.
    cudaStream_t mm_stream, app_stream;
    //cudaError_t retval;
    GUARD_CU(cudaStreamCreateWithFlags( &mm_stream, cudaStreamNonBlocking));
    GUARD_CU(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));
    
    int* exit_signal;
    GUARD_CU(cudaMallocManaged(&exit_signal, sizeof(int32_t)));
    *exit_signal = 0;

    int* exit_counter;
    GUARD_CU(cudaMallocManaged(&exit_counter, sizeof(uint32_t)));
    *exit_counter = 0;

    //int grid_size = 1;
    //int block_size = 32;

    int grid_size = 1;
    int block_size = 64;

    if (argc > 1){
        printf("args: %s %s\n", argv[1], argv[2]);
        grid_size = atoi(argv[1]);
        block_size = atoi(argv[2]);
    }
   /* 
    int man_grid_size, man_block_size;
    GUARD_CU(cudaOccupancyMaxPotentialBlockSize(&man_grid_size, &man_block_size, mem_manager<MemoryMangerType>));
    printf("Manager: Max Potential Occupancy: G:%d, B:%d, gives %d threads\n", \
    man_grid_size, man_block_size, man_grid_size*man_block_size); 
     
    int app_grid_size, app_block_size;
    GUARD_CU(cudaOccupancyMaxPotentialBlockSize(&app_grid_size, &app_block_size, app));
    printf("APP: Max Potential Occupancy: G:%d, B:%d, gives %d threads\n", \
    app_grid_size, app_block_size, app_grid_size*app_block_size); 
  
    int grid_size = min(man_grid_size, app_grid_size);
    int block_size = min(man_block_size, app_block_size);
*/
    int requests_num{grid_size*block_size};
    std::cout << "Number of Allocations: " << requests_num << "\n";

    volatile int** d_memory{nullptr};
    GUARD_CU(cudaMalloc(&d_memory, sizeof(volatile int*) * requests_num));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());

    //Request auxiliary
    RequestType requests;
    requests.init(requests_num);
    requests.memset();
    GUARD_CU(cudaPeekAtLastError());

    //Run presistent kernel (Memory Manager)
    mem_manager<<<grid_size, block_size, 0, mm_stream>>>(exit_signal,
    requests.requests_number, 
    requests.request_iter, 
    requests.request_signal, 
    requests.request_id,
    memory_manager.getDeviceMemoryManager(),
    d_memory,
    requests.request_mem_size,
    /*requests.request_success,*/
    requests.lock);

    //GUARD_CU(cudaStreamSynchronize( mm_stream));
    GUARD_CU(cudaPeekAtLastError());

    int isRunning = 0;
    int old_counter = 0;
    while (1){
        if (exit_counter[0] == block_size*grid_size){
            //printf("break because of exit_counter = %d\n", exit_counter[0]);
            *exit_signal = 1;
            /*GUARD_CU(cudaStreamSynchronize(app_stream));
            GUARD_CU(cudaStreamSynchronize(mm_stream));
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            test1<<<grid_size, block_size, 0, app_stream>>>(d_memory);
            GUARD_CU(cudaDeviceSynchronize());
            GUARD_CU(cudaPeekAtLastError());
            mem_test((int**)d_memory, requests_num, grid_size, block_size, mm_stream);*/
            break;
        }else{
            if (exit_counter[0] != old_counter){
                old_counter = exit_counter[0];
                printf("no break, exit_counter = %d\n", exit_counter[0]);
            }
        }
        if (!isRunning){
            //Run application
            app<<<grid_size, block_size, 0, app_stream>>>(exit_signal, d_memory, 
            requests.request_signal, 
            requests.request_mem_size, 
            requests.request_id, 
            exit_counter, 
            requests.lock);

            GUARD_CU(cudaPeekAtLastError());
            isRunning = 1;
        }
    }

    GUARD_CU(cudaStreamSynchronize(mm_stream));
    GUARD_CU(cudaStreamSynchronize(app_stream));
    printf("DONE!\n");
    return 0;
}

