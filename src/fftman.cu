#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h> // Provides cuComplex, make_cuComplex, etc.
#include <stdio.h>
#include <math.h>       // For M_PI

#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ inline float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

/**
 * @brief Complex subtraction: a - b
 */
__device__ inline float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

/**
 * @brief Complex multiplication: a * b
 */
__device__ inline float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/**
 * @brief Reverses the bottom 'log_n' bits of 'num'.
 * * This is used for the bit-reversal permutation required by the
 * iterative Cooley-Tukey FFT algorithm.
 */
__device__ inline int reverse_bits(int num, int log_n) {
    int reversed_num = 0;
    for (int i = 0; i < log_n; i++) {
        if ((num >> i) & 1) {
            reversed_num |= 1 << (log_n - 1 - i);
        }
    }
    return reversed_num;
}

__global__ void iterative_fft_kernel(float2* data, int n, int log_n) {
    
    // Externally allocated shared memory.
    // We need 'n' complex numbers per block.
    // This is declared as `extern` because its size depends on 'n',
    // which is a runtime parameter (passed during kernel launch).
    extern __shared__ float2 s_data[];

    // ---
    // 1. Bit-Reversal Load from Global to Shared Memory
    // ---
    
    // Each thread loads two elements
    int t = threadIdx.x;            // Thread ID within the block (0 to n/2 - 1)
    int batch_id = blockIdx.x;      // Block ID, corresponds to the batch index
    
    int idx1 = t;
    int idx2 = t + (n / 2);
    
    // Calculate the bit-reversed indices
    int rev_idx1 = reverse_bits(idx1, log_n);
    int rev_idx2 = reverse_bits(idx2, log_n);

    // Get the base pointer for the current batch in global memory
    float2* batch_data = data + batch_id * n;

    // Load from global memory (with bit-reversal) into shared memory
    s_data[rev_idx1] = batch_data[idx1];
    s_data[rev_idx2] = batch_data[idx2];
    
    // Synchronize all threads in the block to ensure s_data is fully loaded
    __syncthreads();

    // ---
    // 2. Iterative Butterfly Operations
    // ---

    // Loop over the FFT stages (from s=1 to log_n)
    for (int s = 1; s <= log_n; s++) {
        int m = 1 << s;     // "Span" of the butterfly
        int m_half = m >> 1;  // Stride
        
        // Calculate indices for this thread's butterfly
        // 't' is still threadIdx.x
        int k = t % m_half;                 // Index of the twiddle factor (0 to m_half - 1)
        int a_idx = (t / m_half) * m + k;   // Index of the 'a' element
        int b_idx = a_idx + m_half;         // Index of the 'b' element

        // Calculate the twiddle factor W_m^k = exp(-2 * PI * i * k / m)
        float angle = -2.0f * M_PI * k / (float)m;
        float2 W = { cosf(angle), sinf(angle) };

        // Perform the butterfly operation:
        // a = a + W*b
        // b = a - W*b
        float2 t_val = s_data[b_idx];  // t_val = b
        float2 W_t = cmul(W, t_val); // W_t = W*b
        float2 a_val = s_data[a_idx];  // a_val = a
        
        s_data[a_idx] = cadd(a_val, W_t);
        s_data[b_idx] = csub(a_val, W_t);
        
        // Synchronize after each stage to ensure all butterflies
        // in this stage are complete before starting the next.
        __syncthreads();
    }

    // ---
    // 3. Write-Back from Shared to Global Memory
    // ---
    
    // Each thread writes its two corresponding elements back to global memory
    // (t is still threadIdx.x)
    batch_data[t] = s_data[t];
    batch_data[t + (n / 2)] = s_data[t + (n / 2)];
}


// ---
// Host Helper Functions
// ---

/**
 * @brief Simple check to see if a number is a power of 2.
 */
bool isPowerOfTwo(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}

/**
 * @brief Simple host-side error checking for CUDA calls.
 */
void checkCudaErr(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Host-side launcher for the batched FFT kernel.
 *
 * @param d_data        Device pointer to the data.
 * @param N             FFT size (must be power of 2, <= 2048).
 * @param batch_size    Number of FFTs to run.
 */
int run_batched_fft(float2* d_data, int NUM_BATCHES, int FFT_SIZE_PER_BATCH,cudaStream_t stream)
{
    if (!isPowerOfTwo(FFT_SIZE_PER_BATCH)) {
        std::cerr << "Error: FFT_SIZE_PER_BATCH (" << FFT_SIZE_PER_BATCH 
                  << ") must be a power of 2." << std::endl;
        return 1;
    }
    
    if (FFT_SIZE_PER_BATCH > 2048) {
         std::cerr << "Error: FFT_SIZE_PER_BATCH (" << FFT_SIZE_PER_BATCH 
                  << ") cannot exceed 2048 (due to max threads per block)." << std::endl;
        return 1;
    }

    // Total number of complex samples
    //int N_COMPLEX = FFT_SIZE_PER_BATCH * NUM_BATCHES;
    // Total number of floats (real + imag)
    //int N_FLOATS = N_COMPLEX * 2;
    int LOG_N = static_cast<int>(log2(FFT_SIZE_PER_BATCH));

    //size_t host_bytes = N_COMPLEX * sizeof(float2);

    dim3 threadsPerBlock(FFT_SIZE_PER_BATCH / 2); // n/2 threads per block
    dim3 blocksPerGrid(NUM_BATCHES);              // One block per FFT batch

    size_t shared_mem_bytes = FFT_SIZE_PER_BATCH * sizeof(float2);

    iterative_fft_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_bytes, stream>>>(d_data, FFT_SIZE_PER_BATCH, LOG_N);

    return 0;
}