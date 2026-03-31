// Just a quick thing I'm testing unrelated to the fsmlp microbench in the benchmarks git, this is separate from that

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#include <thread>
#include <barrier>
#include <vector>

#include <litmus.h>
#include <libsmctrl.h>
#include "litmushelper.hpp"

#include "busywaitcuda.hpp"
#include "fsmlp.hpp"
#include "taskdefs.hpp"
#include "mcslock.hpp"

#define SMLP_OD_ID 1
#define SMLP_NAMESPACE "./smlp_lock_od"
#define NUM_CPU_TASKS 70
#define CPU_PERIOD_MS 100
// NUM_CPU_TASKS/100 represents the total utilization
// so that means each CPU_TASK will take up 1% of the CPU
// as such, each CPU task's cost will be 1ms
#define CPU_COST_MS 1

#define NUM_SMLP_TASKS 10
#define SMLP_COST_MS 20
#define SMLP_PERIOD_MS 100
#define SMLP_GPU_MASK GPC1

#define FSMLP_GPU_MASK GPC3
#define FSMLP_GPUMGR_PERIOD_US 500
#define FSMLP_GPUMGR_COST_US 10
#define NUM_GPUMGR_TASKS 10
#define NUM_FSMLP_TASKS NUM_SMLP_TASKS

using std::thread;
using std::stop_token;
using std::stop_source;
using std::barrier;
using std::vector;
using std::shared_ptr;
using std::make_shared;

static FILE* threadnames = nullptr;

void dummy_cpu_task(stop_token st, lt_t cost, lt_t period, lt_t deadline, barrier<>& jobInitialized);
void dummy_smlp_task(lt_t cost, lt_t period, barrier<>& jobInitialized);
void dummy_fsmlp_task(lt_t cost, lt_t period, shared_ptr<FSMLP> fsmlp, barrier<>& jobInitialized);
void rec_threadname(const char* name);

int main( int argc, char* argv[] ) {
    unlink( SMLP_NAMESPACE );

    threadnames = fopen("tncsv.csv", "w");
    fprintf(threadnames, "threadid, ThreadType\n");
    fflush(threadnames);
    sleep(1);

    barrier jobInit(2);
    vector<thread> cpuThreads;
    vector<thread> smlpThreads;
    vector<thread> fsmlpThreads;
    vector<thread> gpumgrThreads;

    printf("Set scheduler to GSN-EDF.\n");
    system(LIBLITMUS_LIB_DIR "/setsched GSN-EDF");
    sleep(3);
    
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_litmus() );

    // It always seems to work better when the lock is initialized in the main thread, even if not used here
    int smlpod = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, 1, 1 );

    // To stop our CPU tasks
    stop_source stopsrc;

    printf("Creating CPU tasks..\n");
    // Create our dummy CPU tasks
    for( int i = 0; i < NUM_CPU_TASKS; ++i ) {
        cpuThreads.emplace_back( dummy_cpu_task, stopsrc.get_token(), ms2ns( CPU_COST_MS ), ms2ns( CPU_PERIOD_MS ), ms2ns( CPU_PERIOD_MS ), std::ref(jobInit) );
        jobInit.arrive_and_wait(); // One CPU thread at a time
    }

    printf("Creating SMLP tasks..\n");
    // Create our SMLP tasks
    for( int i = 0; i < NUM_SMLP_TASKS; ++i ) {
        smlpThreads.emplace_back( dummy_smlp_task, ms2ns( SMLP_COST_MS ), ms2ns( SMLP_PERIOD_MS ), std::ref(jobInit) );
        jobInit.arrive_and_wait();
    }

    // Create our FSMLP GPU manager and FSMLP-using tasks
    shared_ptr<FSMLP> fsmmlp = make_shared<FSMLP>( FSMLP_GPU_MASK, 10000 );
    printf("Creating FSMLP tasks..\n");
    for( int i = 0; i < NUM_FSMLP_TASKS; ++i ) {
        fsmlpThreads.emplace_back( dummy_fsmlp_task, ms2ns( SMLP_COST_MS ), ms2ns( SMLP_PERIOD_MS ), fsmmlp, std::ref(jobInit) );
        jobInit.arrive_and_wait();
    }

    //thread gpumgr = thread( &GPUManagementTask, stopsrc.get_token(), us2ns( FSMLP_GPUMGR_COST_US ), us2ns( FSMLP_GPUMGR_PERIOD_US ), 0, fsmmlp );
    for( int i = 0; i < NUM_GPUMGR_TASKS; ++i ) {
        gpumgrThreads.emplace_back( GPUManagementTask, stopsrc.get_token(), us2ns( FSMLP_GPUMGR_COST_US ), us2ns( FSMLP_GPUMGR_PERIOD_US ), i * us2ns(100), fsmmlp );
    }

    printf("All tasks created, waiting for release..\n");
    //sleep(1);

    become_periodic( ms2ns(100), ms2ns(100) );

    // Now that everything is initialized, we should check to make sure all of the threads are waiting for release_ts
    while( get_nr_ts_release_waiters() < NUM_CPU_TASKS + NUM_SMLP_TASKS + NUM_FSMLP_TASKS + NUM_GPUMGR_TASKS ) {
        printf("Waiting for all threads to be ready for release, currently %d/%d\n", get_nr_ts_release_waiters(), NUM_CPU_TASKS + NUM_SMLP_TASKS + NUM_FSMLP_TASKS + NUM_GPUMGR_TASKS );
        //_mm_pause();
        //sleep(1);
    }
    // Threads are ready, release!
    printf("releasing!\n");
    //sleep(1);
    release_taskset( ms2ns(100), ms2ns(100) );
    //sleep(1);
    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );

    // Run until all of the smlp and fsmlp threads are done
    for( auto& t : smlpThreads ) {
        t.join();
    }
    for( auto& t : fsmlpThreads ) {
        t.join();
    }

    // Cleanup
    printf("Stopping system..\n");
    stopsrc.request_stop();
    printf("Joining CPU threads..\n");
    for( auto& t : cpuThreads )
        t.join();
    printf("join gpu man..\n");
    for( auto& t : gpumgrThreads )
        t.join();
    sleep(1);
    close(smlpod);
    sleep(1);

    printf("Switching back to Linux scheduler..\n");
    system(LIBLITMUS_LIB_DIR "/setsched Linux");
    sleep(3);

    unlink( SMLP_NAMESPACE );
    return 0;
}

void busywait_launch_fn(unsigned int ms, cudaStream_t stream) {
    //printf("b\n");
    busyWaitGPU( ms, stream );
}

void fsmlp_busywait_launch_fn(GPURequest* req) {
    //printf("f\n");
    busyWaitGPU( ns2ms( req->abs_deadline ), *(req->stream) );
}

void dummy_fsmlp_task(lt_t cost, lt_t period, shared_ptr<FSMLP> fsmlp, barrier<>& jobInitialized) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // cuda warmup
    busywait_launch_fn( ns2ms( cost ), stream );
    cudaStreamSynchronize(stream);

    GPURequest req {
        .abs_deadline = cost, // a hack for this microbenchmark
        .smctrl_tpcs_allowed = ( 1 << 6 ) - 1,
        .stream = &stream,
        .gpu_launch_fn = fsmlp_busywait_launch_fn,
        .job_barrier = std::barrier(2),
        .lock_node = MCSNode{},
        .jobData = nullptr,
        .nodeData = nullptr,
    };

    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    rec_threadname("fsmlp");

    become_periodic(cost, period);
    jobInitialized.arrive_and_wait();
    wait_for_ts_release();

    fsmlp->submitRequest(&req);
    sleep_next_period();
    
    cudaStreamDestroy(stream);
    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}


void dummy_smlp_task(lt_t cost, lt_t period, barrier<>& jobInitialized) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //libsmctrl_set_stream_mask(&stream, ~SMLP_GPU_MASK);

    // CUDA warmup
    busywait_launch_fn( ns2ms( cost ), stream );
    cudaStreamSynchronize(stream);

    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    rec_threadname("smlp");

    become_periodic(cost, period);
    int smlpod = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, 1, 1 );
    jobInitialized.arrive_and_wait();

    wait_for_ts_release();
    
    uint64_t assigned_mask;
    LITMUS_CALL_TID( litmus_smlp_lock( smlpod, 1, &assigned_mask ) );
    //sleepWaitGPU( ns2ms( cost ), stream );
    //busyWaitGPU( ns2ms( cost ), stream );
    busywait_launch_fn( ns2ms( cost ), stream );
    cudaStreamSynchronize(stream);
    LITMUS_CALL_TID( litmus_smlp_gpu_done( smlpod ) );
    LITMUS_CALL_TID( litmus_smlp_unlock( smlpod ) );
    sleep_next_period();

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );

    cudaStreamDestroy(stream);
}

void dummy_cpu_task(stop_token st, lt_t cost, lt_t period, lt_t deadline, barrier<>& jobInitialized) {
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    rec_threadname("cpu");

    become_periodic( cost, period, deadline );
    jobInitialized.arrive_and_wait();
    wait_for_ts_release();

    while(!st.stop_requested()) {
        lt_t start = litmus_clock();
        while( litmus_clock() - start < cost ) {
            if( st.stop_requested() )
                break;
            // Elite Dangerous did this
            _mm_pause();
            _mm_pause();
            _mm_pause();
            _mm_pause();
            _mm_pause();
            _mm_pause();
            _mm_pause();
            _mm_pause();
        }

        if( st.stop_requested() )
            break;
        sleep_next_period();
    }

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}

void rec_threadname(const char* name) {
    fprintf(threadnames, "%ld, %s\n", syscall(SYS_gettid), name);
    fflush(threadnames);
}