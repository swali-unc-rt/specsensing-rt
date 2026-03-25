#include <stdio.h>
#include <stdlib.h>

#include "benchmark.hpp"

#include <chrono>
#include <thread>
#include <random>
#include <string.h>

#include <libsmctrl.h>
#include <cuda_runtime_api.h>

#include "litmushelper.hpp"

#include "trthelpers.hpp"
#include "SignalGenerator.hpp"

#include "rfmlp.hpp"

constexpr float fs = 10e9;

using hrclock = std::chrono::high_resolution_clock;
using namespace std::chrono;

uint64_t getRandomSetBits(int bitsToSet, int maxBitPosition);
void usage(char* argv0);

static int ed_main(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    if( argc < 2 ) {
        usage(argv[0]);
        return 1;
    }

    // Just in case, but you can fix this to a seed to obtain repeatable results.
    // But you will need to change SignalGenerator too.
    srand(time(nullptr));

    if( !strcasecmp(argv[1], "ED") ) {
        return ed_main(argc, argv);
    } else {
        printf("Error unknown benchmark name: %s\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    return 0;
}

void usage_ed(char* argv0);

static int ed_main(int argc, char* argv[]) {
    // Args needed:
    //  (1) benchmark name (already parsed)
    //  (2) jobperiod
    //  (3) startNumSMs
    //  (4) endNumSMs
    //  (5) startInputSize
    //  (6) endInputSize
    //  (7) inputSizeStep
    //  (8) numTrials
    if( argc < 9 ) {
        usage_ed(argv[0]);
        return 1;
    }

    auto demandpositiveint = [argv0 = argv[0]](char* arg) -> int {
        int val = atoi(arg);
        if( val <= 0 ) {
            fprintf(stderr, "Error: argument %s must be a positive integer\n", arg);
            usage_ed(argv0);
            exit(1);
        }
        return val;
    };

    lt_t jobPeriod = us2ns( demandpositiveint(argv[2]) );
    int startNumSMs = demandpositiveint(argv[3]);
    int endNumSMs = demandpositiveint(argv[4]);
    int startInputSize = demandpositiveint(argv[5]);
    int endInputSize = demandpositiveint(argv[6]);
    int inputSizeStep = demandpositiveint(argv[7]);
    int numTrials = demandpositiveint(argv[8]);
    constexpr int samplesPerInput = 1024;

    // Create our analyzed signals
    printf("Generating signals...\n");
    SignalGenerator siggen(10, 2 * endInputSize * samplesPerInput, fs, fs/100, fs/3, fs/300, fs/200, 10, 20, 20.0);
    cudaStream_t stream;
    CHECK_CUDA( cudaStreamCreate(&stream) );

    // Initialize our channel buffer
    void* channelBuffer = nullptr;
    size_t channelBufferSize = sizeof(float2) * endInputSize * samplesPerInput;

    siggen.setSample(0, 1.0);
    siggen.setSample(1, 0.0);
    CHECK_CUDA(cudaMalloc(&channelBuffer, channelBufferSize));
    CHECK_CUDA(cudaMemcpy(channelBuffer, (void*)siggen.getSamples(), channelBufferSize, cudaMemcpyHostToDevice));

    // This buffer will be used by the job
    void* jobBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&jobBuffer, channelBufferSize));

    {
        // First, we need to do a dry run, otherwise the first run will always
        // have a bunch of CUDA initialization overhead that skews the results.
        void* deviceInput = nullptr;
        constexpr int dryRunInputSize = 2;
        CHECK_CUDA(cudaMalloc(&deviceInput, dryRunInputSize * samplesPerInput * sizeof(float2)));

        CHECK_CUDA(cudaMemcpy(deviceInput, channelBuffer, dryRunInputSize * samplesPerInput * sizeof(float2), cudaMemcpyDeviceToDevice));
        run_batched_fft( (float2*)deviceInput, dryRunInputSize, samplesPerInput, stream );
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaFree(deviceInput));
    }

    unsigned long totalRuns = (endNumSMs - startNumSMs + 1) * ( (endInputSize - startInputSize) / inputSizeStep + 1 ) * numTrials;
    int intPercentage = 0;
    unsigned long runsCompleted = 0;

    // Initialize litmus
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_litmus() );

    become_periodic( jobPeriod-1, jobPeriod );

    LITMUS_CALL_TID( wait_for_ts_release() );

    for( int numSMs = startNumSMs; numSMs <= endNumSMs; ++numSMs ) {
        for( int numInputs = startInputSize; numInputs <= endInputSize; numInputs += inputSizeStep ) {
            size_t jobBufferSize = sizeof(float2) * numInputs * samplesPerInput;

            for( int i = 0; i < numTrials; ++i ) {
                sleep_next_period();

                libsmctrl_set_stream_mask(stream, ~getRandomSetBits(numSMs, NUM_SMs) );
                cudaMemcpyAsync(jobBuffer, channelBuffer, jobBufferSize, cudaMemcpyDeviceToDevice, stream);
                run_batched_fft( (float2*)jobBuffer, numInputs, samplesPerInput, stream );
                ++runsCompleted;
                CHECK_CUDA(cudaStreamSynchronize(stream));
            }

            int newIntPercentage = (int)( (runsCompleted * 100) / totalRuns );
            if( newIntPercentage != intPercentage ) {
                intPercentage = newIntPercentage;
                printf("Progress: %d%%\r", intPercentage);
                fflush(stdout);
            }

        }
    }

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
    return 0;
}

uint64_t getRandomSetBits(int bitsToSet, int maxBitPosition) {
    uint64_t mask = 0;
    while( bitsToSet > 0 ) {
        int bitPos = rand() % maxBitPosition;
        for( ; bitPos < maxBitPosition; bitPos++ ) {
            if( (mask & (1ULL << bitPos)) == 0 ) {
                mask |= (1ULL << bitPos);
                --bitsToSet;
                break;
            }
        }
    }
    return mask;
}

void usage_ed(char* argv0) {
    // Args needed:
    //  (1) benchmark name (already parsed)
    //  (2) jobperiod
    //  (3) startNumSMs
    //  (4) endNumSMs
    //  (5) startInputSize
    //  (6) endInputSize
    //  (7) inputSizeStep
    //  (8) numTrials
    fprintf(stderr,
        "Usage: %s ED <jobperiod_us> <startNumSMs> <endNumSMs> <startInputSize> <endInputSize> <inputSizeStep> <numTrials>\n"
        "\n"
        "Will run the ed benchmark at the specified period, sweeping:\n"
        "   [startNumSMs, endNumSMs] for the number of active SMs\n"
        "   [startInputSize, endInputSize] for the input size\n"
        "   * numTrials for each configuration above.\n"
        "   Each trial has a random set of TPCs selected.\n"
        "1 input = 1024 samples\n"
        "\n"
        "Note, this program by itself does nothing. Make sure to use\n"
        "it with feather-trace in the python directory.\n"
        "\n"
    , argv0 );
}

void usage(char* argv0) {
    fprintf(stderr,
        "Usage: %s <benchmark_name>\n"
        "Available benchmarks:\n"
        "   ED   energy detection\n"
        "   AMC  automatic modulation classification\n"
        "   SEI  specific emitter identification\n"
        "   GEO  geolocation\n"
        "\n"
        "Note, this program by itself does nothing. Make sure to use\n"
        "it with feather-trace in the python directory.\n"
        "\n"
    , argv0);
}