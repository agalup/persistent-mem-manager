cudaError_t GRError(cudaError_t error, const char *message,
                    const char *filename, int line, bool print) {
  if (error && print) {
    int gpu;
    cudaGetDevice(&gpu);
    fprintf(stderr, "[%s, %d @ gpu %d] %s (CUDA error %d: %s)\n", filename,
            line, gpu, message, error, cudaGetErrorString(error));
    fflush(stderr);
  }
  return error;
}

#define GUARD_CU(cuda_call)                                                   \
  {                                                                           \
    if (cuda_call != (enum cudaError) CUDA_SUCCESS){  \
        printf("--- ERROR(%d:%s) --- %s:%d\n", cuda_call, cudaGetErrorString(cuda_call), __FILE__, __LINE__);\
    } \
  }\

struct RequestType{

    volatile int* requests_number; 
    volatile int* request_iter;
    volatile int* request_signal; 
    volatile int* request_id; 
    volatile int* request_mem_size;
    volatile int* lock;
    int size;

    void init(size_t Size);
    void memset();
};

void RequestType::init(size_t Size){

    size = Size;

    GUARD_CU(cudaMallocManaged(&requests_number,         sizeof(volatile int)));
    
    GUARD_CU(cudaMallocManaged(&request_iter,            sizeof(volatile int)));

    GUARD_CU(cudaMallocManaged(&request_signal,   size * sizeof(volatile int)));
    
    GUARD_CU(cudaMallocManaged(&request_id,       size * sizeof(volatile int)));
 
    GUARD_CU(cudaMallocManaged(&request_mem_size, size * sizeof(volatile int)));

    GUARD_CU(cudaMallocManaged(&lock,             size * sizeof(volatile int)));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}

void RequestType::memset(){

    *requests_number = size;

    *request_iter = 0;

    GUARD_CU(cudaMemset((void*)request_signal, 0,   size * sizeof(volatile int)));

    GUARD_CU(cudaMemset((void*)request_id, -1,      size * sizeof(volatile int)));

    GUARD_CU(cudaMemset((void*)request_mem_size, 0, size * sizeof(volatile int)));

    GUARD_CU(cudaMemset((void*)lock, 0,             size * sizeof(volatile int)));

    GUARD_CU(cudaDeviceSynchronize());
    GUARD_CU(cudaPeekAtLastError());
}


__global__
void copy(int** d_memory0, int* d_memory, int size){
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    d_memory[thid] = d_memory0[thid][0];
}
//mem test
void mem_test(int** d_memory0, int requests_num, int blocks, int threads, 
                cudaStream_t& mm_stream){
    //create array
    int* d_memory{nullptr};
    cudaError_t retval;
    retval = cudaMalloc(&d_memory, sizeof(int) * requests_num);
    if (retval){
        printf("malloc does not work %d\n", retval);
    }
    retval = cudaDeviceSynchronize();
    if (retval){
        printf("sync does not work\n");
    }
    copy<<<blocks, threads, 0, mm_stream>>>(d_memory0, d_memory, requests_num);
    retval = cudaDeviceSynchronize();
    if (retval){
        printf("sync does not work\n");
    }
    int* h_memory = (int*)malloc(requests_num* sizeof(int));
    retval = cudaMemcpy(h_memory, d_memory, sizeof(int)*requests_num, cudaMemcpyDeviceToHost);
    if (retval){
        printf("memcpy does not work\n");
    }
    retval = cudaDeviceSynchronize();
    if (retval){
        printf("sync does not work\n");
    }
    /*for (int i=0; i< requests_num; ++i){
        if (h_memory[i] != i){//1234321){//i){
            printf("h_memory[%d] = %d != %d\n", i, h_memory[i], i);
        }
    }*/
}
template <typename MemoryManagerType>
__global__
void mem_free(MemoryManagerType* mm, int** d_memory, int requests_num){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    if (thid >= requests_num){
        return;
    }
    mm->free(d_memory[thid]);
}

__device__ void acquire_semaphore(int* lock, int i){
    while (atomicCAS(&lock[i], 0, 1) != 0);
    __threadfence();
}

__device__ void release_semaphore(int* lock, int i){
    __threadfence();
    lock[i] = 0;
}


//test
__global__
void test1(volatile int** d_memory){
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    assert(d_memory[thid]);
    d_memory[thid][0] *= 100;
}

