#include "RTSystem.hpp"
#include "litmushelper.hpp"
#include "Logger.hpp"

#include "taskset.hpp"
#include "taskdefs.hpp"
#include "SignalGenerator.hpp"
#include "Logger.hpp"

#include <cmath>

using std::thread;
using std::shared_ptr;

SignalGenerator* channel = nullptr;
void* channelBuffer = nullptr;
AMC* amc = nullptr;
SEI* sei = nullptr;
GEO* geo = nullptr;

void RTSystem::createSystem(int numDAGs) {
    if( numDAGs > all_nodes.size() ) {
        Log::logerror("Warning: cannot create more than %lu DAGs (%d given)\n", all_nodes.size(), numDAGs);
        exit(1);
    }

    for( int i = 0; i < numDAGs; i++ ) {
        //printf("Starting dag %d/%d\n", i, numDAGs);
        auto& dag_nodes = all_nodes[i];
        auto& dag_edges = all_edges[i];

        lt_t chPeriod = s2ns( optEntries[TEST_ITER].chBatches[i] * ed_mcost.samplesPerInput / channel_fs[i] ) * 10;
        //auto dag = new DAG(i, this, stopper.get_token());
        auto dag = make_shared<DAG>(i, this, stopper.get_token());
        dag->setPeriod( chPeriod );
        dag->setReleaserCost( us2ns(200) );
        dag->setE2EResponseTime( s2ns( optEntries[TEST_ITER].objective ) );
        //printf("Set dag period and cost: %llu us and %llu us\n", chPeriod / 1000, us2ns(200) / 1000);

        for( auto& node : dag_nodes ) {
            int nodeId = node[0];
            int processFnId = node[1];
            NodeData* extraData = new NodeData{ .fs = channel_fs[i] };

            //printf("Node %d:\n", nodeId);

            RFMLType fntype;

            if( processFnId == 3 )
                processFnId = 2; // remap FSSEI to GEO
            
            switch( processFnId ) {
            case 0: fntype = RFMLType::ED; extraData->mcost = ed_mcost; break;
            case 1: fntype = RFMLType::AMC; extraData->mcost = amc_mcost; break;
            case 2: fntype = RFMLType::SEI; extraData->mcost = sei_mcost; break;
            case 4: fntype = RFMLType::GEO; extraData->mcost = geo_mcost; break;
            default:
                fprintf(stderr, "Error: unknown processFnId %d\n", processFnId);
                Log::logerror("Error: unknown processFnId %d\n", processFnId);
                exit(1);
            };
            extraData->batchCount = optEntries[TEST_ITER].chBatches[i] * ed_mcost.samplesPerInput / extraData->mcost.samplesPerInput;
            extraData->maxTPCcount = extraData->mcost.maxTPCcount; // kinda redundant

            // printf("Creating node %d of type %d with period %llu us, init cost %llu us, and per-input cost %llu us (total cost for %d inputs: %llu us)\n",
            //     nodeId, (int)fntype, chPeriod / 1000, extraData->mcost.initCost / 1000, extraData->mcost.costPerInput / 1000, extraData->batchCount, (extraData->mcost.initCost + extraData->mcost.costPerInput * extraData->batchCount) / 1000);
            if( extraData->mcost.maxBatchSize && extraData->batchCount > extraData->mcost.maxBatchSize ) {
                Log::logerror("Error: node %d of DAG %d has batch count %d greater than max batch size %d for its RFML type, this may lead to incorrect wcets\n",
                    nodeId, i, extraData->batchCount, extraData->mcost.maxBatchSize);
                //exit(1);
                // TODO: fix the rfml sizes
                extraData->batchCount = extraData->mcost.maxBatchSize;
            }
            
            auto newNode = dag->createNode( nodeId, extraData, getNodeFns(fntype), stopper.get_token() );
            //printf("node created\n");
            newNode->setPeriod( chPeriod );
            newNode->setCost( extraData->mcost.initCost + extraData->mcost.costPerInput * extraData->batchCount );
            newNode->setDeadline( s2ns( optEntries[TEST_ITER].objective ) );
        }

        // Now add the edges
        for( auto& edge : dag_edges ) {
            int from_id = edge[0];
            int to_id = edge[1];
            dag->addEdge(from_id, to_id);
        }

        addDAG(dag);
    }

    
    //printf("Added DAGs, now calculating offsets, deadlines, and extra instances.\n");

    // Now that all of the DAGs are created, we can calculate the Omaxparent and R values for each node based on the DAG structure
    calculateOffsetsAndDeadlines();
}

void RTSystem::calculateOffsetsAndDeadlines() {
    // TODO: recalculate offsets and deadlines, but for now, use the e2e_R given by Joseph

    std::vector<shared_ptr<DAG>> dagCopies;
    for( auto& dag : dags ) {
        unsigned int neededDAGs = ( std::floor( dag->getE2EResponseTime() / dag->getPeriod() ) + 1 );
        //fprintf(stderr, "DAG with period %llu us and e2e_R %llu us needs %u instances\n", dag->getPeriod() / 1000, dag->getE2EResponseTime() / 1000, neededDAGs);

        auto invoker = std::make_shared<DAGInvoker>(dag->getPeriod(), stopper.get_token());
        invoker->addDAG(dag);

        for( auto i = 0u; i < neededDAGs; ++i ) {
            auto newDAG = dag->makeCopy();
            dagCopies.push_back(newDAG);
            invoker->addDAG(newDAG);
        }
        invokers.push_back(invoker);
    }

    // Add the copies into the system
    for( auto& dagCopy : dagCopies ) {
        addDAG(dagCopy);
    }
}

void RTSystem::start() {
    // Start every DAG
    //printf("Starting all DAG nodes..\n");
    for( auto& dag : dags ) {
        dag->startAllNodes();
    }

    //printf("Starting all DAG invokers..\n");
    // For every invoker, start it
    for( auto& invoker : invokers ) {
        invoker->start();
    }

    gpu_man_task = std::thread( GPUManagementTask, stopper.get_token(), us2ns(100), ms2ns(1), fsmlp );
    timer_thread = std::thread( &RTSystem::stopTimer, this, stopper.get_token() );
}

void RTSystem::stop() {
    stopper.request_stop();
}

void RTSystem::stopTimer(std::stop_token stop_tok) {
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    Log::logthreadname("emergencyStopper", -1, -1, false);
    become_periodic( us2ns(200), ms2ns(100) );

    auto start = std::chrono::high_resolution_clock::now();

    while( !stop_tok.stop_requested() ) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if( elapsed >= SECONDS_TO_RUN * 2 ) {
            stopper.request_stop();
            exit(1);
            break;
        }

        sleep_next_period();
    }

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}

RTSystem::RTSystem()
    : lockType(LOCK_TYPE_FOR_RUN), fsmlp(std::make_shared<FSMLP>(FSMLP_GPCUSE, 10000)) {
    // Constructor implementation
}

RTSystem::~RTSystem() {
    invokers.clear();
    dags.clear();
}

void DAGInvoker::periodicRelease() {
    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    Log::logthreadname("releaser", -1, -1, false);

    become_periodic( us2ns(200), period );

    invokeStartBarrier.arrive_and_wait();

    LITMUS_CALL_TID( wait_for_ts_release() );

    while( !stopper.stop_requested() ) {
        auto dag = dags.front();
        dags.pop();
        dag->release_nodes();
        dags.push(dag);

        sleep_next_period();
    }

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}

void DAGInvoker::start() {
    periodic_releaser = thread(&DAGInvoker::periodicRelease, this);
    invokeStartBarrier.arrive_and_wait();
}

DAGInvoker::~DAGInvoker() {
    if( periodic_releaser.joinable() ) {
        //printf("Joining releaser thread..\n");
        periodic_releaser.join();
        //printf("done.\n");
    }
}

RTSystem* RTSystem::_instance = nullptr;

RTSystem* RTSystem::Instance() {
    if (!_instance) {
        _instance = new RTSystem();
    }
    return _instance;
}