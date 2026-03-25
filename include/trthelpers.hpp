#pragma once

#include <iostream>
#include <memory>
#include <list>
#include <atomic>
#include <utility>

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <mutex>

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

size_t getTensorSize(const nvinfer1::Dims& dims);

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

class NvInferLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;

public:
    static NvInferLogger* _instance;
    static NvInferLogger& Instance();
    static void deleteInstance();
};

class TRTWorkload;

class TRTContextPool {
public:
    void addToPool(int count = 1);
    
    nvinfer1::IExecutionContext* getContext(cudaStream_t** stream);
    void returnContext(nvinfer1::IExecutionContext* ctx, cudaStream_t* stream);

    TRTContextPool(nvinfer1::ICudaEngine* baseEngine);
    ~TRTContextPool();
private:
    nvinfer1::ICudaEngine* baseEngine;
    
    void getpoollock();
    void exitpoollock();

    std::mutex trt_lock;

    std::list<std::pair<nvinfer1::IExecutionContext*,cudaStream_t*>> contexts;
};