#include "pmm.cu"
using namespace std;

int main(int argc, char *argv[]){

    size_t instant_size = 1024ULL * 1024ULL * 1024ULL;
    int size_to_alloc = 4;
    int iteration_num = 1;

    int turn_on = 1;
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
    cudaGetDeviceProperties(&deviceProp, 0);
    int SMs = deviceProp.multiProcessorCount;
    printf("max block number %d\n", SMs);
    int size = SMs - 1;

    int* sm_app        = (int*)malloc(sizeof(int)*size);
    int* sm_mm         = (int*)malloc(sizeof(int)*size);
    int* allocs_size   = (int*)malloc(sizeof(int)*size);
    float* app_launch  = (float*)malloc(sizeof(float)*size);
    float* app_finish  = (float*)malloc(sizeof(float)*size);
    float* app_sync    = (float*)malloc(sizeof(float)*size);
    float* uni_req_num = (float*)malloc(sizeof(float)*size);
    
    pmm_init(turn_on, size_to_alloc, instant_size, iteration_num, SMs, sm_app, sm_mm, 
            allocs_size, app_launch, app_finish, app_sync, uni_req_num);

    printf("DONE!\n");
    return 0;
}

