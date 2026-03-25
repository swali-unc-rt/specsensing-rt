#include <stdio.h>
#include <stdlib.h>

#include <litmus.h>
#include "litmushelper.hpp"

#include "lrt_dag.hpp"
#include "rfmlp.hpp"

#include "SignalGenerator.hpp"

constexpr unsigned int INPUTS_TO_USE = 4;
constexpr float fs = 10e9;

using std::make_shared;
using std::shared_ptr;
using std::stop_token;
using std::stop_source;

void initED(Node* node, void** processData);
void initAMC(Node* node, void** processData);
void initSEI(Node* node, void** processData);
void initGEO(Node* node, void** processData);

void jobED(Node* node, void* processData);
void jobAMC(Node* node, void* processData);
void jobSEI(Node* node, void* processData);
void jobGEO(Node* node, void* processData);

void cleanupED(Node* node, void* processData);
void cleanupAMC(Node* node, void* processData);
void cleanupSEI(Node* node, void* processData);
void cleanupGEO(Node* node, void* processData);

SignalGenerator* channel = nullptr;
void* channelBuffer = nullptr;

int main( int argc, char* argv[] ) {
    // Initialize non-RT program parts
    SignalGenerator siggen(10, 2 * INPUTS_TO_USE * 1024, fs, fs/100, fs/3, fs/300, fs/200, 10, 20, 20.0);
    siggen.setSample(0, 1.0);
    siggen.setSample(1, 0.0);
    CHECK_CUDA( cudaMalloc(&channelBuffer, sizeof(float2) * INPUTS_TO_USE * 1024) );
    CHECK_CUDA( cudaMemcpy(channelBuffer, siggen.getSamples(), sizeof(float2) * INPUTS_TO_USE * 1024, cudaMemcpyHostToDevice) );
    channel = &siggen;

    // Set schedule to GSN-EDF
    system(LIBLITMUS_LIB_DIR "/setsched GSN-EDF");
    sleep(3);

    // Initialize litmus
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_litmus() );
    litmus_releasegroup_envinit();

    // Initialize RT program
    std::stop_source stopper;
    auto dag = make_shared<DAG>( 1, nullptr, stopper.get_token() );
    dag->setPeriod( ms2ns(1000) );
    dag->setReleaserCost( ms2ns(100) );

    // Create structure:
    //     1
    //    / \
    //   2   3
    //    \ /
    //     4
    {
        auto node1 = dag->createNode( 1, nullptr, NodeFNs{jobED, initED, cleanupED}, stopper.get_token() );
        auto node2 = dag->createNode( 2, nullptr, NodeFNs{jobAMC, initAMC, cleanupAMC}, stopper.get_token() );
        auto node3 = dag->createNode( 3, nullptr, NodeFNs{jobSEI, initSEI, cleanupSEI}, stopper.get_token() );
        auto node4 = dag->createNode( 4, nullptr, NodeFNs{jobGEO, initGEO, cleanupGEO}, stopper.get_token() );

        node1->setCost( ms2ns(100) ); node1->setPeriod( ms2ns(1000) ); node1->setDeadline( ms2ns(1000) );
        node2->setCost( ms2ns(100) ); node2->setPeriod( ms2ns(1000) ); node2->setDeadline( ms2ns(1000) );
        node3->setCost( ms2ns(100) ); node3->setPeriod( ms2ns(1000) ); node3->setDeadline( ms2ns(1000) );
        node4->setCost( ms2ns(100) ); node4->setPeriod( ms2ns(1000) ); node4->setDeadline( ms2ns(1000) );
    }
    dag->addEdge( 1, 2 );
    dag->addEdge( 1, 3 );
    dag->addEdge( 2, 4 );
    dag->addEdge( 3, 4 );
    dag->startReleaser();
    dag->startAllNodes();

    while( get_nr_ts_release_waiters() < 5 ) {
        // Wait for threads
    }

    release_taskset( ms2ns(100), ms2ns(100) );

    sleep(10);

    printf("Stopping..\n");
    stopper.request_stop();
    sleep(4);

    dag->release_nodes(); // ensure all nodes have been released before we begin cleaning up the release group environment
    sleep(4);
    dag.reset(); // ensure DAG is cleaned up before we clean up the release group environment

    printf("Destroying releasegroups..\n");
    litmus_releasegroup_envdestroy();
    sleep(4);

    // Switch back to Linux scheduler (will clean up GSN-EDF as well)
    system(LIBLITMUS_LIB_DIR "/setsched Linux");
    sleep(3);

    CHECK_CUDA( cudaFree(channelBuffer) );
    return 0;
}

struct jobData {
    void* deviceInput;
    void* deviceOutput;
    cudaStream_t stream;
    RFML* model;
};

void initED(Node* node, void** processData) {
    jobData *data = new jobData;
    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * ed_mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * ed_mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    *processData = data;
}
void initAMC(Node* node, void** processData) {
    jobData *data = new jobData;
    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * amc_mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * amc_mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->model = new AMC();
    *processData = data;
}
void initSEI(Node* node, void** processData) {
    jobData *data = new jobData;
    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * sei_mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * sei_mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->model = new SEI();
    *processData = data;
}
void initGEO(Node* node, void** processData) {
    jobData *data = new jobData;
    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * geo_mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * geo_mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->model = new GEO();
    *processData = data;
}

void jobED(Node* node, void* processData) {
    printf("ED job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * ed_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    run_batched_fft( (float2*)data->deviceInput, INPUTS_TO_USE, ed_mcost.samplesPerInput, data->stream );
    CHECK_CUDA( cudaStreamSynchronize(data->stream) );
    printf("ED job done\n");
}
void jobAMC(Node* node, void* processData) {
    printf("AMC job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * amc_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("AMC job done\n");
}
void jobSEI(Node* node, void* processData) {
    printf("SEI job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * sei_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("SEI job done\n");
}
void jobGEO(Node* node, void* processData) {
    printf("GEO job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * geo_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("GEO job done\n");
}


void cleanupED(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    delete data;
}
void cleanupAMC(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    delete data->model;
    delete data;
}
void cleanupSEI(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    delete data->model;
    delete data;
}
void cleanupGEO(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    delete data->model;
    delete data;
}