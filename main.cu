#include "pmm.cu"
using namespace std;

int main(int argc, char *argv[]){

    size_t instant_size = 8 * 1024ULL * 1024ULL * 1024ULL;
    int size_to_alloc = 4;
    int iteration_num = 1;

    int turn_on = 0;//1;
    /*if (argc > 1){
        turn_on = atoi(argv[1]);
    }*/
    if (argc > 1){
        size_to_alloc = atoi(argv[1]);
    }
    if (argc > 2){
        iteration_num = atoi(argv[2]);
    }
    if (argc > 3){
        instant_size = atoi(argv[3]);
    }

   // int* app_launch = (int*)malloc
    cudaDeviceProp deviceProp;
    GUARD_CU(cudaGetDeviceProperties(&deviceProp, 0));
    int SMs = deviceProp.multiProcessorCount;
    printf("max block number %d\n", SMs);
    printf("instant size %ld\n", instant_size);
    int size = SMs - 1;

    int* sm_app            = (int*)malloc(sizeof(int)*size);
    int* sm_mm             = (int*)malloc(sizeof(int)*size);
    int* allocs_size       = (int*)malloc(sizeof(int)*size);
    float* app_launch      = (float*)malloc(sizeof(float)*size);
    float* app_finish      = (float*)malloc(sizeof(float)*size);
    float* app_sync_pmm    = (float*)malloc(sizeof(float)*size);
    float* uni_req_num_pmm = (float*)malloc(sizeof(float)*size);
    float* app_sync        = (float*)malloc(sizeof(float)*size);
    float* uni_req_num     = (float*)malloc(sizeof(float)*size);
    
    pmm_init(turn_on, size_to_alloc, &instant_size, iteration_num, SMs, 
            sm_app, sm_mm, allocs_size, app_launch, app_finish, 
            app_sync_pmm, uni_req_num_pmm);

    GUARD_CU(cudaDeviceReset());
    GUARD_CU(cudaPeekAtLastError());

    perf_alloc(size_to_alloc, &instant_size, iteration_num, SMs, 
            app_sync, uni_req_num, turn_on);

    printf("DONE!\n");
    return 0;
}

