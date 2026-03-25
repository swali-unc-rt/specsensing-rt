#include "trthelpers.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <litmus.h>
#include <immintrin.h>

size_t getTensorSize(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) return 0;
    size_t totalSize = 1;
    for( int i = 0; i < dims.nbDims; ++i ) {
        if( dims.d[i] == -1 ) continue;
        totalSize *= dims.d[i];
    }
    return totalSize;
}

void NvInferLogger::log(Severity severity, const char* msg) noexcept {
    // Suppress info-level messages
    if (severity == Severity::kINFO || severity == Severity::kVERBOSE) {
        return;
    }

    // switch (severity) {
    //     case Severity::kINTERNAL_ERROR: fprintf(stderr, "INTERNAL_ERROR: "); break;
    //     case Severity::kERROR:   fprintf(stderr, "ERROR: ");   break;
    //     case Severity::kWARNING: fprintf(stderr, "WARNING: "); break;
    //     case Severity::kINFO:    fprintf(stderr, "INFO: ");    break;
    //     case Severity::kVERBOSE: fprintf(stderr, "VERBOSE: "); break;
    //     default:                 fprintf(stderr, "UNKNOWN: "); break;
    // }
    // fprintf(stderr, "%s\n", msg);
}

NvInferLogger* NvInferLogger::_instance = nullptr;
NvInferLogger& NvInferLogger::Instance() {
    if (!_instance) {
        _instance = new NvInferLogger();
    }
    return *_instance;
}

void NvInferLogger::deleteInstance() {
    if (_instance) {
        delete _instance;
        _instance = nullptr;
    }
}

void TRTContextPool::returnContext(nvinfer1::IExecutionContext* ctx, cudaStream_t* stream) {
    getpoollock();
    contexts.push_back(std::make_pair(ctx, stream));
    exitpoollock();
}

TRTContextPool::TRTContextPool(nvinfer1::ICudaEngine* baseEngine) : baseEngine(baseEngine) {
}

TRTContextPool::~TRTContextPool() {
    for( auto ctx : contexts )
        CHECK_CUDA( cudaStreamDestroy(*ctx.second) );
}

nvinfer1::IExecutionContext* TRTContextPool::getContext(cudaStream_t** stream) {
    getpoollock();
    if( contexts.size() > 0 ) {
        std::pair<nvinfer1::IExecutionContext*,cudaStream_t*> ctx = contexts.front();
        contexts.pop_front();
        exitpoollock();
        *stream = ctx.second;
        return ctx.first;
    }

    fprintf(stderr,"TRTContextPool: No available contexts, creating new one\n");

    // Create a new one
    auto ctx = baseEngine->createExecutionContext();
    exitpoollock();
    cudaStream_t* newStream = new cudaStream_t;
    CHECK_CUDA( cudaStreamCreate(newStream) );
    *stream = newStream;
    return ctx;
}

void TRTContextPool::addToPool(int count) {
    getpoollock();
    for( int i = 0; i < count; ++i ) {
        nvinfer1::IExecutionContext* ctx = baseEngine->createExecutionContext();
        cudaStream_t* newStream = new cudaStream_t;
        CHECK_CUDA( cudaStreamCreate(newStream) );
        contexts.push_back( std::make_pair(ctx, newStream) );
    }
    exitpoollock();
}

void TRTContextPool::getpoollock() {
    trt_lock.lock();
}

void TRTContextPool::exitpoollock() {
    trt_lock.unlock();
}