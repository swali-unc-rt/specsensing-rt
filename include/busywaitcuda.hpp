#pragma once

#include <cuda_runtime_api.h>

void busyWaitGPU(float milliseconds, cudaStream_t stream);

#if __CUDA_ARCH__ >= 700
void sleepWaitGPU(unsigned int milliseconds, cudaStream_t stream);
#endif