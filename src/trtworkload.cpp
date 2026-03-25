#include "trtworkload.hpp"

#include <stdexcept>
#include <fstream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#include "libsmctrl.h"

using std::string;
using std::ifstream;
using std::vector;

bool TRTWorkload::inference(
    string input_name, void* deviceInputBuffer,
    string output_name, void* deviceOutputBuffer,
    nvinfer1::Dims inputDims,
    cudaStream_t streamToUse,
    bool doStreamSync
) {
    if( !ctx ) {
        fprintf(stderr,"No execution context available for inference\n");
        return false;
    }
    
    if( !ctx->setInputTensorAddress(input_name.c_str(), deviceInputBuffer) ) {
        fprintf(stderr,"Failed to set input tensor address for %s\n", input_name.c_str());
        return false;
    }

    if( !ctx->setOutputTensorAddress(output_name.c_str(), deviceOutputBuffer) ) {
        fprintf(stderr,"Failed to set output tensor address for %s\n", output_name.c_str());
        return false;
    }

    ctx->setInputShape(input_name.c_str(), inputDims);

    ctx->enqueueV3(streamToUse);

    if( doStreamSync )
        cudaStreamSynchronize(streamToUse);

    return true;
}

bool validateTensorInput(nvinfer1::ICudaEngine* engine, string name) {
    return engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
}

bool validateTensorOutput(nvinfer1::ICudaEngine* engine, string name) {
    return engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kOUTPUT;
}

bool TRTWorkload::validateTensorInput(string name) const {
    return ::validateTensorInput(engine,name);
}

bool TRTWorkload::validateTensorOutput(string name) const {
    return ::validateTensorOutput(engine,name);
}

void TRTWorkload::outputTensors() const {
    for( int i = 0; i < engine->getNbIOTensors(); ++i) {
        printf( "Tensor %d: %s, dimensions: ", i, engine->getIOTensorName(i) );
        auto dims = engine->getTensorShape(engine->getIOTensorName(i));
        if( dims.nbDims == 0 ) {
            printf("\n");
            continue;
        }

        printf("%ld", dims.d[0]);
        for( int i = 1; i < dims.nbDims; ++i )
            printf("x%ld", dims.d[i]);
        printf("\n");
    }
}

TRTWorkload::TRTWorkload(string filename) {
    // Initialize pointers
    this->runtime = nullptr;
    this->engine = nullptr;
    this->ctx = nullptr;
    nofreectx = false;

    // Read the engine file
    ifstream engineFile( filename, std::ios::binary );
    if( !engineFile.good() )
        throw std::invalid_argument("Could not open engine file");
    
    // Grab the size of the engine file
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    // Read it into a buffer
    vector<char> engineBuffer(engineSize);
    engineFile.read(engineBuffer.data(), engineSize);
    engineFile.close();

    // Deserialize
    this->runtime = nvinfer1::createInferRuntime( NvInferLogger::Instance() );
    if( !this->runtime )
        throw std::runtime_error("Could not create inference runtime");
    
    this->engine = runtime->deserializeCudaEngine(engineBuffer.data(), engineSize);
    if( !this->engine ) {
        delete this->runtime;
        throw std::runtime_error("Could not deserialize cuda engine");
    }

    // Create execution context
    this->ctx = this->engine->createExecutionContext();
    if( !this->ctx ) {
        delete this->runtime;
        delete this->engine;
        throw std::runtime_error("Could not create execution context");
    }

    //CHECK_CUDA(cudaStreamCreate(&this->stream));
}

TRTWorkload::TRTWorkload(nvinfer1::ICudaEngine* engine) {
    this->runtime = nullptr;
    this->engine = nullptr;
    this->ctx = engine->createExecutionContext();
    nofreectx = false;
    if( !this->ctx )
        throw std::runtime_error("Could not create execution context");
    //CHECK_CUDA(cudaStreamCreate(&this->stream));
}

TRTWorkload::TRTWorkload(nvinfer1::IExecutionContext* ctx) {
    this->runtime = nullptr;
    this->engine = nullptr;
    this->ctx = ctx;
    nofreectx = true;
    //CHECK_CUDA(cudaStreamCreate(&this->stream));
}

TRTWorkload::~TRTWorkload() {
    if( ctx && !nofreectx ) {
        //cudaStreamDestroy(stream);
        delete ctx;
    }
    if( engine ) delete engine;
    if( runtime ) delete runtime;
}