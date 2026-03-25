#include "rfmlp.hpp"
#include "trtworkload.hpp"

#include <stdexcept>

RFML::RFML(std::string filename, std::string input_name, std::string output_name)
    : TRTWorkload(filename), inputname(input_name), outputname(output_name) {
    if( !validateTensorInput(inputname) )
        throw std::invalid_argument("bad input tensor name");
    if( !validateTensorOutput(outputname) )
        throw std::invalid_argument("bad output tensor name");
}

RFML::RFML(nvinfer1::ICudaEngine* engine, std::string input_name, std::string output_name)
    : TRTWorkload(engine), inputname(input_name), outputname(output_name) {
    if( !::validateTensorInput(engine,inputname) )
        throw std::invalid_argument("bad input tensor name");
    if( !::validateTensorOutput(engine,outputname) )
        throw std::invalid_argument("bad output tensor name");
}

RFML::RFML(nvinfer1::IExecutionContext* ctx, std::string input_name, std::string output_name)
    : TRTWorkload(ctx), inputname(input_name), outputname(output_name) {
    if( !::validateTensorInput((nvinfer1::ICudaEngine*)&ctx->getEngine(),inputname) )
        throw std::invalid_argument("bad input tensor name");
    if( !::validateTensorOutput((nvinfer1::ICudaEngine*)&ctx->getEngine(),outputname) )
        throw std::invalid_argument("bad output tensor name");
}

bool RFML::inference(void* deviceInput, void* deviceOutput, size_t numInputs, cudaStream_t stream, bool doStreamSync) {
    return TRTWorkload::inference(inputname, deviceInput, outputname, deviceOutput, getInputDimension(numInputs), stream, doStreamSync );
}

bool FSSEI::inference(void* deviceInputBuffer, void* deviceOutputBuffer, size_t numInputs, cudaStream_t stream, bool doStreamSync) {
    if( numInputs > FSSEI_MAX_BATCH_SIZE ) {
        fprintf(stderr,"!!!!! Batch size %zu exceeds FSSEI max batch size of %d\n", numInputs, FSSEI_MAX_BATCH_SIZE);
        return false;
    }

    auto ctx = getContext();

    if( !ctx ) {
        fprintf(stderr,"No execution context available for inference\n");
        return false;
    }
    
    if( !ctx->setInputTensorAddress(inputname.c_str(), deviceInputBuffer) ) {
        fprintf(stderr,"Failed to set input tensor address for %s\n", inputname.c_str());
        return false;
    }

    if( !ctx->setInputTensorAddress("input_2", input2DeviceBuffer ) ) {
        fprintf(stderr,"Failed to set input tensor address for input_2\n");
        return false;
    }

    if( !ctx->setOutputTensorAddress(outputname.c_str(), deviceOutputBuffer) ) {
        fprintf(stderr,"Failed to set output tensor address for %s\n", outputname.c_str());
        return false;
    }

    if( !ctx->setOutputTensorAddress("dense_1", output2DeviceBuffer) ) {
        fprintf(stderr,"Failed to set output tensor address for dense_1\n");
        return false;
    }

    if( !ctx->setOutputTensorAddress("Center", output3DeviceBuffer) ) {
        fprintf(stderr,"Failed to set output tensor address for Center\n");
        return false;
    }

    ctx->setInputShape(inputname.c_str(), getInputDimension(numInputs) );
    ctx->setInputShape("input_2", nvinfer1::Dims2{ (int64_t)numInputs, 1 } );

    ctx->enqueueV3(stream);

    if( doStreamSync )
        cudaStreamSynchronize(stream);

    return true;
}

void FSSEI::allocExtraBuffers() {
    auto ctx = getContext();

    size_t input2Size = sizeof(float) * 1 * FSSEI_MAX_BATCH_SIZE;
    size_t output2Size = sizeof(float) * 1024 * FSSEI_MAX_BATCH_SIZE;
    size_t output3Size = sizeof(float) * 1 * FSSEI_MAX_BATCH_SIZE;

    CHECK_CUDA(cudaMalloc(&input2DeviceBuffer, input2Size));
    CHECK_CUDA(cudaMalloc(&output2DeviceBuffer, output2Size));
    CHECK_CUDA(cudaMalloc(&output3DeviceBuffer, output3Size));

    if( !::validateTensorInput((nvinfer1::ICudaEngine*)&ctx->getEngine(),"input_2") )
        throw std::invalid_argument("bad input tensor name input_2 for FSSEI");
    if( !::validateTensorOutput((nvinfer1::ICudaEngine*)&ctx->getEngine(),"dense_1") )
        throw std::invalid_argument("bad output tensor name dense_1 for FSSEI");
    if( !::validateTensorOutput((nvinfer1::ICudaEngine*)&ctx->getEngine(),"Center") )
        throw std::invalid_argument("bad output tensor name Center for FSSEI");
}

void FSSEI::freeExtraBuffers() {
    if( input2DeviceBuffer ) {
        CHECK_CUDA(cudaFree(input2DeviceBuffer));
        input2DeviceBuffer = nullptr;
    }
    if( output2DeviceBuffer ) {
        CHECK_CUDA(cudaFree(output2DeviceBuffer));
        output2DeviceBuffer = nullptr;
    }
    if( output3DeviceBuffer ) {
        CHECK_CUDA(cudaFree(output3DeviceBuffer));
        output3DeviceBuffer = nullptr;
    }
}