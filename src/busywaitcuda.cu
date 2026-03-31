#include "busywaitcuda.hpp"
#include <cuda_runtime.h>


__global__ void busyWaitKernel(unsigned long long delay_cycles) {
    unsigned long long start_time = clock64();
    
    // Spin until the required number of cycles has elapsed
    while ((clock64() - start_time) < delay_cycles) {
        // Do nothing, just keep checking the clock
    }
}

// Host function to calculate cycles and launch
void busyWaitGPU(float milliseconds, cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // prop.clockRate is returned in kilohertz (kHz), 
    // which is conveniently exactly "cycles per millisecond".
    unsigned long long delay_cycles = static_cast<unsigned long long>(milliseconds * prop.clockRate);

    // Launch with a single thread. 
    // (Launch with more blocks/threads if you need to lock up the whole GPU)
    busyWaitKernel<<<1, 1, 0, stream>>>(delay_cycles);
}

#if __CUDA_ARCH__ >= 700
__global__ void sleepKernel(unsigned int milliseconds) {
    // __nanosleep takes an unsigned int (max ~4.29 seconds per call).
    // We break it into 1-second chunks to safely support longer delays.
    unsigned int chunk_ms = 1000;
    unsigned int full_chunks = milliseconds / chunk_ms;
    unsigned int remainder_ms = milliseconds % chunk_ms;

    for (unsigned int i = 0; i < full_chunks; ++i) {
        __nanosleep(chunk_ms * 1000000); // Convert ms to ns
    }
    
    if (remainder_ms > 0) {
        __nanosleep(remainder_ms * 1000000);
    }
}

// Host function to calculate cycles and launch
void sleepWaitGPU(unsigned int milliseconds, cudaStream_t stream) {
    sleepKernel<<<1, 1, 0, stream>>>(milliseconds);
}
#endif