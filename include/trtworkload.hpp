#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "trthelpers.hpp"

#include <string>

class TRTWorkload {
public:
    // When the ctx is already built
    TRTWorkload(nvinfer1::IExecutionContext* ctx);
    // When the engine is already built
    TRTWorkload(nvinfer1::ICudaEngine* engine);
    // When the engine needs to be deserialized
    TRTWorkload(std::string filename);
    virtual ~TRTWorkload();

    virtual bool inference(std::string input_name,
        void* deviceInputBuffer,
        std::string output_name,
        void* deviceOutputBuffer,
        nvinfer1::Dims inputDims,
        cudaStream_t streamToUse,
        bool doStreamSync
    );

    bool validateTensorInput(std::string input_name) const;
    bool validateTensorOutput(std::string output_name) const;
    void outputTensors() const;

    nvinfer1::IRuntime* getRuntime() { return runtime; }
    nvinfer1::ICudaEngine* getCudaEngine() { return engine; }
    nvinfer1::IExecutionContext* getContext() { return ctx; }
private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* ctx;
    bool nofreectx;
};

bool validateTensorInput(nvinfer1::ICudaEngine* engine, std::string name);
bool validateTensorOutput(nvinfer1::ICudaEngine* engine, std::string name);