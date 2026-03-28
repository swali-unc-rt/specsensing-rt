#pragma once

#include <string>

#define REPLACE_FSSEI

//#define AMC_FILE_LOCATION "/workspace/amc_rt.trt"
//#define AMC_FILE_LOCATION "/mnt/onr/amc_1x-5120x.trt"
#define AMC_FILE_LOCATION "/mnt/onr/trt2/amc_1_5.trt"
//#define AMC_FILE_LOCATION "/mnt/onr/trt/amc_201_205.trt"
//#define AMC_FILE_LOCATION "/mnt/onr/amc_32x-128x.trt"
//#define AMC_FILE_LOCATION "/mnt/onr/amc_1x-16x.trt"
#define AMC_INPUT_NAME "input_1"
#define AMC_OUTPUT_NAME "activation_8"
#define AMC_MAX 5

//#define SEI_FILE_LOCATION "/workspace/sei_rt.trt"
//#define SEI_FILE_LOCATION "/mnt/onr/sei_1x-5120x.trt"
//#define SEI_FILE_LOCATION "/mnt/onr/trt/sei2_6_10.trt"
#define SEI_FILE_LOCATION "/mnt/onr/trt2/sei2_1_5.trt"
//#define SEI_FILE_LOCATION "/mnt/onr/sei_32x-128x.trt"
//#define SEI_FILE_LOCATION "/mnt/onr/sei_1x-20x.trt"
#define SEI_INPUT_NAME "input"
#define SEI_OUTPUT_NAME "output"
#define SEI_MAX 5

//#define GEO_FILE_LOCATION "/mnt/onr/trt/localization_16_20.trt"
//#define GEO_FILE_LOCATION "/mnt/onr/trt/localization_61_65.trt"
#define GEO_FILE_LOCATION "/mnt/onr/trt2/localization_1_5.trt"
//#define GEO_FILE_LOCATION "/mnt/onr/ecrts26/Cellular-Position-Estimation-Using-Deep-Learning/localization.trt"
#define GEO_INPUT_NAME "x"         // size x 1 x 96
#define GEO_OUTPUT_NAME "linear_2" // size x 2
#define GEO_MAX 5

// //#define FSSEI_FILE_LOCATION "/mnt/onr/trt/fssei_1_5.trt"
// #define FSSEI_FILE_LOCATION "/mnt/onr/trt/fssei_41_45.trt"
// //#define FSSEI_FILE_LOCATION "/mnt/onr/fssei_1_2048.trt"
// #define FSSEI_INPUT1_NAME "input_1" // (-1, 4800, 2)
// #define FSSEI_INPUT2_NAME "input_2" // (-1, 1)
// #define FSSEI_OUTPUT1_NAME "Classifier" // (-1, 90)
// #define FSSEI_OUTPUT2_NAME "dense_1" // (-1, 1024)
// #define FSSEI_OUTPUT3_NAME "Center" // (-1,1)
// #define FSSEI_MAX_BATCH_SIZE 2048


#include <litmus.h>

struct marginalCost {
    lt_t initCost;
    lt_t costPerInput;
    unsigned int samplesPerInput;
    unsigned int floatOutsPerInput;
    int maxBatchSize;
    int maxTPCcount;
};

constexpr auto maxsms_ed = 2;
constexpr auto maxsms_sei = 4;
constexpr auto maxsms_amc = 4;
constexpr auto maxsms_geo = 4;
//constexpr auto maxsms_fssei = 8;

constexpr auto ed_mcost = marginalCost { .initCost = ms2ns(193), .costPerInput = ms2ns(11), .samplesPerInput = 1024, .floatOutsPerInput = 0, .maxBatchSize = 0, .maxTPCcount = maxsms_ed };
constexpr auto amc_mcost = marginalCost { .initCost = ms2ns(507), .costPerInput = ms2ns(505), .samplesPerInput = 1024, .floatOutsPerInput = 24, .maxBatchSize = AMC_MAX, .maxTPCcount = maxsms_amc };
constexpr auto sei_mcost = marginalCost { .initCost = ms2ns(1151), .costPerInput = ms2ns(1556), .samplesPerInput = 512, .floatOutsPerInput = 4, .maxBatchSize = SEI_MAX, .maxTPCcount = maxsms_sei };
constexpr auto geo_mcost = marginalCost { .initCost = ms2ns(699), .costPerInput = ms2ns(47), .samplesPerInput = 96, .floatOutsPerInput = 2, .maxBatchSize = GEO_MAX, .maxTPCcount = maxsms_geo };
//constexpr auto fssei_mcost = marginalCost { .initCost = us2ns(1090), .costPerInput = us2ns(1957), .samplesPerInput = 4800, .floatOutsPerInput = 90, .maxBatchSize = FSSEI_MAX, .maxTPCcount = maxsms_fssei };


enum class RFMLType {
    ED,
    AMC,
    SEI,
    FSSEI,
    GEO,
};


#include "trtworkload.hpp"
//#include "ltdag.hpp"

#include <cuda_runtime_api.h>

// All this class does is wrap around TRTWorkload and
//  set the input names, tensor sizes, and validate.
class RFML : public TRTWorkload {
public:
    virtual bool inference(void* deviceInput, void* deviceOutput, size_t numInputs, cudaStream_t stream, bool doStreamSync);

    virtual size_t getOutputSize(size_t inputSize) const = 0;
    virtual nvinfer1::Dims getInputDimension(int64_t numInputs) const = 0;
    virtual std::string getModelName() const = 0;

    RFML(std::string filename, std::string input_name, std::string output_name);
    RFML(nvinfer1::ICudaEngine* engine, std::string input_name, std::string output_name);
    RFML(nvinfer1::IExecutionContext* ctx, std::string input_name, std::string output_name);
    virtual ~RFML() {}
protected:
    std::string inputname;
    std::string outputname;
};

class AMC : public RFML {
public:
    virtual size_t getOutputSize(size_t inputSize) const override { return inputSize * 24 * sizeof(float); }
    virtual nvinfer1::Dims getInputDimension(int64_t numInputs) const override { return nvinfer1::Dims3{numInputs, 1024, 2}; }

    AMC() : RFML(AMC_FILE_LOCATION, AMC_INPUT_NAME, AMC_OUTPUT_NAME) {}
    AMC(nvinfer1::ICudaEngine* amcEngine) : RFML(amcEngine, AMC_INPUT_NAME, AMC_OUTPUT_NAME) {}
    AMC(nvinfer1::IExecutionContext* ctx) : RFML(ctx, AMC_INPUT_NAME, AMC_OUTPUT_NAME) {}
    virtual ~AMC() {}
    virtual std::string getModelName() const override { return "AMC"; }
protected:
};

class SEI : public RFML {
public:
    virtual size_t getOutputSize(size_t inputSize) const override { return inputSize * 4 * sizeof(float); }
    virtual nvinfer1::Dims getInputDimension(int64_t numInputs) const override { return nvinfer1::Dims3{numInputs, 2, 512}; }

    SEI() : RFML(SEI_FILE_LOCATION, SEI_INPUT_NAME, SEI_OUTPUT_NAME) {}
    SEI(nvinfer1::ICudaEngine* seiEngine) : RFML(seiEngine, SEI_INPUT_NAME, SEI_OUTPUT_NAME) {}
    SEI(nvinfer1::IExecutionContext* ctx) : RFML(ctx, SEI_INPUT_NAME, SEI_OUTPUT_NAME) {}
    virtual ~SEI() {}
    virtual std::string getModelName() const override { return "SEI"; }
protected:
};

class GEO : public RFML {
public:
    virtual size_t getOutputSize(size_t inputSize) const override { return inputSize * 2 * sizeof(float); }
    virtual nvinfer1::Dims getInputDimension(int64_t numInputs) const override { return nvinfer1::Dims3{numInputs, 1, 96}; }

    GEO() : RFML(GEO_FILE_LOCATION, GEO_INPUT_NAME, GEO_OUTPUT_NAME) {}
    GEO(nvinfer1::ICudaEngine* geoEngine) : RFML(geoEngine, GEO_INPUT_NAME, GEO_OUTPUT_NAME) {}
    GEO(nvinfer1::IExecutionContext* ctx) : RFML(ctx, GEO_INPUT_NAME, GEO_OUTPUT_NAME) {}
    virtual ~GEO() {}
    virtual std::string getModelName() const override { return "GEO"; }
};

// class FSSEI : public RFML {
// public:
//     virtual size_t getOutputSize(size_t inputSize) const override { return inputSize * 90 * sizeof(float); }
//     virtual nvinfer1::Dims getInputDimension(int64_t numInputs) const override { return nvinfer1::Dims3{numInputs, 4800, 2}; }

//     virtual bool inference(void* deviceInput, void* deviceOutput, size_t numInputs, cudaStream_t stream, bool doStreamSync) override;

//     FSSEI() : RFML(FSSEI_FILE_LOCATION, FSSEI_INPUT1_NAME, FSSEI_OUTPUT1_NAME) { allocExtraBuffers(); }
//     FSSEI(nvinfer1::ICudaEngine* fsseiEngine) : RFML(fsseiEngine, FSSEI_INPUT1_NAME, FSSEI_OUTPUT1_NAME) { allocExtraBuffers(); }
//     FSSEI(nvinfer1::IExecutionContext* ctx) : RFML(ctx, FSSEI_INPUT1_NAME, FSSEI_OUTPUT1_NAME) { allocExtraBuffers(); }
//     virtual ~FSSEI() { freeExtraBuffers(); }
// private:
//     void* input2DeviceBuffer;
//     void* output2DeviceBuffer;
//     void* output3DeviceBuffer;

//     void allocExtraBuffers();
//     void freeExtraBuffers();
// };

// in fftman.cu
int run_batched_fft(float2* d_data, int NUM_BATCHES, int FFT_SIZE_PER_BATCH, cudaStream_t stream);