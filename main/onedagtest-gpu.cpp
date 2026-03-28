#include <stdio.h>
#include <stdlib.h>

#include <litmus.h>
#include "litmushelper.hpp"

#include "lrt_dag.hpp"
#include "rfmlp.hpp"

#include "SignalGenerator.hpp"
#include "RTSystem.hpp"
#include "taskdefs.hpp"

constexpr unsigned int INPUTS_TO_USE = 4;
constexpr float fs = 10e9;

using std::make_shared;
using std::shared_ptr;
using std::stop_token;
using std::stop_source;

//SignalGenerator* channel = nullptr;
//void* channelBuffer = nullptr;

int main( int argc, char* argv[] ) {
    amc = new AMC();
    sei = new SEI();
    geo = new GEO();
    
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
    //     1  (root)
    //    / \ 
    //   2   3
    //    \ / 
    //     4  
    {
        auto node1 = dag->createNode( 1, nullptr, getNodeFnsSimple(RFMLType::ED), stopper.get_token() );
        auto node2 = dag->createNode( 2, nullptr, getNodeFnsSimple(RFMLType::AMC), stopper.get_token() );
        auto node3 = dag->createNode( 3, nullptr, getNodeFnsSimple(RFMLType::SEI), stopper.get_token() );
        auto node4 = dag->createNode( 4, nullptr, getNodeFnsSimple(RFMLType::GEO), stopper.get_token() );

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
    delete amc;
    delete sei;
    delete geo;
    return 0;
}