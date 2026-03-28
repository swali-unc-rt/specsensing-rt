#include "taskdefs.hpp"
#include "Logger.hpp"
#include "RTSystem.hpp"
#include "fsmlp.hpp"

#include <litmus.h>
#include "litmushelper.hpp"

constexpr unsigned int INPUTS_TO_USE = 4;
constexpr float fs = 10e9;
extern void* channelBuffer;
extern AMC* amc;
extern SEI* sei;
extern GEO* geo;

NodeFNs getNodeFns(RFMLType fntype) {
    switch( fntype ) {
    case RFMLType::ED: return {jobED, initED, simple_cleanupED};
    case RFMLType::AMC: return {jobAMC, initAMC, simple_cleanupAMC};
    case RFMLType::SEI: return {jobSEI, initSEI, simple_cleanupSEI};
    //case RFMLType::FSSEI: return {jobSEI, initSEI, simple_cleanupSEI}; // FSSEI is just SEI for now
    case RFMLType::GEO: return {jobGEO, initGEO, simple_cleanupGEO};
    default:
        Log::logerror("Error: unknown RFMLType %d\n", (int)fntype);
        fprintf(stderr, "Error: unknown RFMLType %d\n", (int)fntype);
        exit(1);
    }
}

NodeFNs getNodeFnsSimple(RFMLType fntype) {
    switch( fntype ) {
    case RFMLType::ED: return {simple_jobED, simple_initED, simple_cleanupED};
    case RFMLType::AMC: return {simple_jobAMC, simple_initAMC, simple_cleanupAMC};
    case RFMLType::SEI: return {simple_jobSEI, simple_initSEI, simple_cleanupSEI};
    //case RFMLType::FSSEI: return {simple_jobSEI, simple_initSEI, simple_cleanupSEI}; // FSSEI is just SEI for now
    case RFMLType::GEO: return {simple_jobGEO, simple_initGEO, simple_cleanupGEO};
    default:
        Log::logerror("Error: unknown RFMLType %d\n", (int)fntype);
        fprintf(stderr, "Error: unknown RFMLType %d\n", (int)fntype);
        exit(1);
    }
}

void RFMLLaunchFn(GPURequest* req) {
    auto data = static_cast<jobData*>(req->jobData);
    auto nodeData = static_cast<NodeData*>(req->nodeData);

#ifndef DO_NO_GPU_WORK
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, nodeData->batchCount, data->stream, false);
#endif
}

void jobED(Node* node, void* processData) {
    jobData* data = static_cast<jobData*>(processData);
    NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());
    // ED is the only job type that uses the kernel SMLP always
    uint64_t assignedMask = 0;
    //printf("job start od %d\n", data->smlp_od);
    
    int rv = litmus_smlp_lock(data->smlp_od, ( 1UL << nodeData->maxTPCcount ) - 1, &assignedMask);
    if( rv ) {
        Log::logerror("ED Error: litmus_smlp_lock %lx returned %d %m\n", ( 1UL << nodeData->maxTPCcount ) - 1, rv);
    }
#ifndef DO_NO_GPU_WORK
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    run_batched_fft( (float2*)data->deviceInput, nodeData->batchCount, nodeData->mcost.samplesPerInput, data->stream );
    CHECK_CUDA( cudaStreamSynchronize(data->stream) );
#endif
    litmus_smlp_gpu_done(data->smlp_od);
    litmus_smlp_unlock(data->smlp_od);
}

void jobAMC(Node* node, void* processData) {
    jobData* data = static_cast<jobData*>(processData);

    //printf("job start od %d\n", data->smlp_od);

    struct control_page* cpage = get_ctrl_page();
    if( !cpage ) {
        fprintf(stderr, "worker node has no control page?\n");
        exit(1);
    }

    if( RTSystem::Instance()->getLockType() == LockType::SMLP ) {
        NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());
        uint64_t assignedMask = 0;
        int rv = litmus_smlp_lock(data->smlp_od, ( 1UL << nodeData->maxTPCcount ) - 1, &assignedMask);
        if( rv ) {
            Log::logerror("%s Error: litmus_smlp_lock %lx returned %d %m\n", data->model->getModelName().c_str(), ( 1UL << nodeData->maxTPCcount ) - 1, rv);
        } else {
            libsmctrl_set_stream_mask(data->stream, ~assignedMask);
        }
#ifndef DO_NO_GPU_WORK
        CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
        data->model->inference(data->deviceInput, data->deviceOutput, nodeData->batchCount, data->stream, true);
#endif
        litmus_smlp_gpu_done(data->smlp_od);
        litmus_smlp_unlock(data->smlp_od);
    } else {
        data->req.abs_deadline = cpage->deadline;
        RTSystem::Instance()->getFSMLP()->submitRequest( &data->req );
    }

    //printf("job end\n");
}

void jobSEI(Node* node, void* processData) {
    jobAMC(node,processData);
}

void jobGEO(Node* node, void* processData) {
    jobAMC(node,processData);
}

void initED(Node* node, void** processData) {
    Log::logthreadname("ED", node->getDAG()->getId(), node->getId(), node->isSink());
    NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());
    
    jobData *data = new jobData {
        .model = nullptr,
        .mcost = nodeData->mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << 1 ) - 1,
            .gpu_launch_fn = nullptr,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
            .nodeData = (void*)nodeData,
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * nodeData->batchCount * nodeData->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->req.stream = &data->stream;
    data->smlp_od = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, SMLP_GPCUSE, 1 );
    printf("%d ED (%llu,%llu,%llu): Opened SMLP semaphore with id %d %lX\n", litmus_gettid(), node->getCost() / 1000, node->getPeriod() / 1000, node->getDeadline() / 1000, data->smlp_od, SMLP_GPCUSE );

    *processData = data;
}

void initAMC(Node* node, void** processData) {
    Log::logthreadname("AMC", node->getDAG()->getId(), node->getId(), node->isSink());
    NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());
    
    jobData *data = new jobData {
        .model = new AMC(amc->getCudaEngine()),
        .mcost = nodeData->mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << nodeData->maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
            .nodeData = (void*)nodeData,
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * nodeData->batchCount * nodeData->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->req.stream = &data->stream;
    data->smlp_od = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, SMLP_GPCUSE, 1 );
    printf("%d AMC (%llu,%llu,%llu): Opened SMLP semaphore with id %d %lX\n", litmus_gettid(), node->getCost() / 1000, node->getPeriod() / 1000, node->getDeadline() / 1000, data->smlp_od, SMLP_GPCUSE );

    *processData = data;
}

void initSEI(Node* node, void** processData) {
    Log::logthreadname("SEI", node->getDAG()->getId(), node->getId(), node->isSink());
    NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());

    jobData *data = new jobData {
        .model = new SEI(sei->getCudaEngine()),
        .mcost = nodeData->mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << nodeData->maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
            .nodeData = (void*)nodeData,
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * nodeData->batchCount * nodeData->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->req.stream = &data->stream;
    data->smlp_od = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, SMLP_GPCUSE, 1 );
    printf("%d SEI (%llu,%llu,%llu): Opened SMLP semaphore with id %d %lX\n", litmus_gettid(), node->getCost() / 1000, node->getPeriod() / 1000, node->getDeadline() / 1000, data->smlp_od, SMLP_GPCUSE );

    *processData = data;
}

void initGEO(Node* node, void** processData) {
    Log::logthreadname("GEO", node->getDAG()->getId(), node->getId(), node->isSink());
    NodeData* nodeData = static_cast<NodeData*>(node->getExtraData());

    jobData *data = new jobData {
        .model = new GEO(geo->getCudaEngine()),
        .mcost = nodeData->mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << nodeData->maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
            .nodeData = (void*)nodeData,
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * nodeData->batchCount * nodeData->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * nodeData->batchCount * nodeData->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    data->req.stream = &data->stream;
    data->smlp_od = open_smlp_sem( SMLP_OD_ID, SMLP_NAMESPACE, SMLP_GPCUSE, 1 );
    printf("%d GEO (%llu,%llu,%llu): Opened SMLP semaphore with id %d %lX\n", litmus_gettid(), node->getCost() / 1000, node->getPeriod() / 1000, node->getDeadline() / 1000, data->smlp_od, SMLP_GPCUSE );

    *processData = data;
}

void simple_jobED(Node* node, void* processData) {
    printf("ED job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * ed_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    run_batched_fft( (float2*)data->deviceInput, INPUTS_TO_USE, ed_mcost.samplesPerInput, data->stream );
    CHECK_CUDA( cudaStreamSynchronize(data->stream) );
    printf("ED job done\n");
}
void simple_jobAMC(Node* node, void* processData) {
    printf("AMC job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * amc_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("AMC job done\n");
}
void simple_jobSEI(Node* node, void* processData) {
    printf("SEI job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * sei_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("SEI job done\n");
}
void simple_jobGEO(Node* node, void* processData) {
    printf("GEO job starting\n");
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaMemcpyAsync(data->deviceInput, channelBuffer, sizeof(float2) * INPUTS_TO_USE * geo_mcost.samplesPerInput, cudaMemcpyDeviceToDevice, data->stream) );
    data->model->inference(data->deviceInput, data->deviceOutput, INPUTS_TO_USE, data->stream, true);
    printf("GEO job done\n");
}

void simple_initED(Node* node, void** processData) {
    jobData *data = new jobData {
        .model = nullptr,
        .mcost = ed_mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << ed_mcost.maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * data->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * data->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );

    *processData = data;
}
void simple_initAMC(Node* node, void** processData) {
    jobData *data = new jobData {
        .model = new AMC(amc->getCudaEngine()),
        .mcost = amc_mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << amc_mcost.maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
        },
    };
    data->req.jobData = (void*)data;
    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * data->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * data->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    *processData = data;
}
void simple_initSEI(Node* node, void** processData) {
    jobData *data = new jobData {
        .model = new SEI(sei->getCudaEngine()),
        .mcost = sei_mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << sei_mcost.maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * data->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * data->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    *processData = data;
}
void simple_initGEO(Node* node, void** processData) {
    jobData *data = new jobData {
        .model = new GEO(geo->getCudaEngine()),
        .mcost = geo_mcost,
        .smlp_od = -1,
        .req = GPURequest {
            .smctrl_tpcs_allowed = ( 1UL << geo_mcost.maxTPCcount ) - 1,
            .gpu_launch_fn = RFMLLaunchFn,
            .job_barrier = std::barrier(2),
            .lock_node = MCSNode{},
        },
    };
    data->req.jobData = (void*)data;

    CHECK_CUDA( cudaMalloc(&data->deviceInput, sizeof(float2) * INPUTS_TO_USE * data->mcost.samplesPerInput ) );
    CHECK_CUDA( cudaMalloc(&data->deviceOutput, sizeof(float) * INPUTS_TO_USE * data->mcost.floatOutsPerInput ) );
    CHECK_CUDA( cudaStreamCreate(&data->stream) );
    *processData = data;
}


void simple_cleanupED(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    if( data ) {
        delete data->model;
        if( data->smlp_od != -1 )
            close( data->smlp_od );
        delete data;
    }
    if( node && node->getExtraData() )
        delete (NodeData*)node->getExtraData();
}
void simple_cleanupAMC(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    if( data ) {
        delete data->model;
        if( data->smlp_od != -1 )
            close( data->smlp_od );
        delete data;
    }
    if( node && node->getExtraData() )
        delete (NodeData*)node->getExtraData();
}
void simple_cleanupSEI(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    if( data ) {
        delete data->model;
        if( data->smlp_od != -1 )
            close( data->smlp_od );
        delete data;
    }
    if( node && node->getExtraData() )
        delete (NodeData*)node->getExtraData();
}
void simple_cleanupGEO(Node* node, void* processData) {
    jobData *data = static_cast<jobData*>(processData);
    CHECK_CUDA( cudaFree(data->deviceInput) );
    CHECK_CUDA( cudaFree(data->deviceOutput) );
    CHECK_CUDA( cudaStreamDestroy(data->stream) );
    if( data ) {
        delete data->model;
        if( data->smlp_od != -1 )
            close( data->smlp_od );
        delete data;
    }
    if( node && node->getExtraData() )
        delete (NodeData*)node->getExtraData();
}

void GPUManagementTask(std::stop_token stopper, lt_t releaser_cost, lt_t period, std::shared_ptr<FSMLP> fsmlp) {
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    become_periodic(releaser_cost, period);
    LITMUS_CALL_TID( wait_for_ts_release() );

    struct control_page* cpage = get_ctrl_page();

    while( !stopper.stop_requested() ) {
        cpage->sched.np.flag = 1;
        fsmlp->processQueues();
        cpage->sched.np.flag = 0; // if this isn't zero before sleep_next_period, then it won't sleep
        sleep_next_period();
    }

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}