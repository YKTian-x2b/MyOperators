#include <iostream>

#define M 4
#define N 4
#define K 8

int main(int argc, char **argv){
    size_t tid = 0;
    // while(tid < 1024){
    //     const size_t warp_id = tid / 32;
    //     const size_t lane_id = tid % 32;
    //     if(warp_id != (tid >> 5)){
    //         std::cout << " warp_id != (tid >> 5)" << std::endl;
    //     }
    //     if(lane_id != (tid & 31)){
    //         std::cout << "lane_id != (tid & 31)" << std::endl;
    //     }
    //     tid++;
    // }
    std::cout << sizeof(size_t) << std::endl;
}