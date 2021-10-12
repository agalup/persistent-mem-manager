#include "pmm.cu"
using namespace std;

int main(int argc, char *argv[]){

    size_t instant_size = 1024ULL * 1024ULL * 1024ULL;

    int turn_on = 1;
    if (argc > 1){
        turn_on = atoi(argv[1]);
    }

   // int* app_launch = (int*)malloc
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int SMs = deviceProp.multiProcessorCount;
    printf("max block number %d\n", SMs);
    int size = SMs - 1;

    int* sm_app      = (int*)malloc(sizeof(int)*size);
    int* sm_mm       = (int*)malloc(sizeof(int)*size);
    int* allocs_size = (int*)malloc(sizeof(int)*size);
    float* app_launch = (float*)malloc(sizeof(float)*size);
    float* app_finish = (float*)malloc(sizeof(float)*size);
    float* app_sync   = (float*)malloc(sizeof(float)*size);
    
    pmm_init(turn_on, instant_size, SMs, sm_app, sm_mm, allocs_size, 
            app_launch, app_finish, app_sync);

    printf("DONE!\n");
    return 0;
}

