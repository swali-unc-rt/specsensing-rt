#include <stdio.h>
#include <stdlib.h>

#include <litmus.h>
#include "litmushelper.hpp"

#include "lrt_dag.hpp"
#include "rfmlp.hpp"
#include "Logger.hpp"
#include "taskdefs.hpp"
#include "RTSystem.hpp"
#include "taskset.hpp"

#include "SignalGenerator.hpp"

#include <thread>

using std::make_shared;
using std::shared_ptr;
using std::stop_token;
using std::stop_source;

constexpr unsigned int INPUTS_TO_USE = 5;
constexpr float fs = 10e9;

void do_dry_gpu_init();

int main(int argc, char* argv[]) {
    amc = new AMC();
    sei = new SEI();
    geo = new GEO();

    unlink( SMLP_NAMESPACE );

    // Set schedule to GSN-EDF
    //system(LIBLITMUS_LIB_DIR "/setsched GSN-EDF");
    sleep(3);
    
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_litmus() );
    LITMUS_CALL_TID( litmus_releasegroup_envinit() );

    int smlp_od = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, SMLP_GPCUSE, 1 );

    printf("Generating signal samples...\n");
    SignalGenerator siggen(10, 2 * INPUTS_TO_USE * 1024, fs, fs/100, fs/3, fs/300, fs/200, 10, 20, 20.0);
    siggen.setSample(0, 1.0);
    siggen.setSample(1, 0.0);
    channel = &siggen;
    printf("Dry run channel initailization..\n");
    CHECK_CUDA( cudaMalloc(&channelBuffer, sizeof(float2) * INPUTS_TO_USE * 1024) );
    CHECK_CUDA( cudaMemcpy(channelBuffer, siggen.getSamples(), sizeof(float2) * INPUTS_TO_USE * 1024, cudaMemcpyHostToDevice) );
    do_dry_gpu_init();

    // Now that we're done with that, we can begin..
    auto rt = RTSystem::Instance();
    printf("Creating system..\n");
    rt->createSystem(optEntries[TEST_ITER].numChannels);

    // this will create all of the tasks and threads for the system
    printf("Starting system..\n");
    rt->start();

    for( int i = 0; i < 4; ++i ) {
        printf("Threads: %d\n", get_nr_ts_release_waiters() );
        sleep(1);
    }

    //sleep(10);
    printf("Releasing taskset..\n");
    release_taskset(ms2ns(100),ms2ns(100));

    // Sleep for the needed duration
    printf("Sleeping for %d seconds to let the system run..\n", SECONDS_TO_RUN);
    sleep(SECONDS_TO_RUN);

    // End the system
    printf("Stopping system..\n");
    //exit(1);
    rt->stop();
    sleep(5);

    printf("Cleaning up..\n");

    //delete amc;
    //delete sei;
    //delete geo;
    
    //LITMUS_CALL_TID( litmus_releasegroup_envdestroy() );

    printf("Switching back to Linux scheduler..\n");
    system(LIBLITMUS_LIB_DIR "/setsched Linux");
    sleep(3);

    unlink( SMLP_NAMESPACE );
    return 0;
}

void do_dry_gpu_init() {
    void *pd_ed, *pd_amc, *pd_sei, *pd_geo;
    printf("Doing dry-run GPU jobs to initialize CUDA context and TensorRT engines...\n");
    printf("Initializing..\n");
    simple_initED(nullptr, &pd_ed);
    printf("init ED done\n");
    simple_initAMC(nullptr, &pd_amc);
    simple_initSEI(nullptr, &pd_sei);
    simple_initGEO(nullptr, &pd_geo);
    printf("Running CUDA jobs..\n");
    simple_jobED(nullptr, pd_ed);
    simple_jobAMC(nullptr, pd_amc);
    simple_jobSEI(nullptr, pd_sei);
    simple_jobGEO(nullptr, pd_geo);
    printf("Cleaning up..\n");
    simple_cleanupED(nullptr, pd_ed);
    simple_cleanupAMC(nullptr, pd_amc);
    simple_cleanupSEI(nullptr, pd_sei);
    simple_cleanupGEO(nullptr, pd_geo);
    printf("Done with dry-run initialization.\n");
}

