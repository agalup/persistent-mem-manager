#include "pmm.cu"
using namespace std;


int main(int argc, char *argv[]){

    size_t instant_size = 7168ULL * 1024ULL * 1024ULL;

    int turn_on = 1;
    if (argc > 1){
        turn_on = atoi(argv[1]);
    }

   // int* app_launch = (int*)malloc

    pmm_init(turn_on, instant_size);
    
    printf("DONE!\n");
    return 0;
}

