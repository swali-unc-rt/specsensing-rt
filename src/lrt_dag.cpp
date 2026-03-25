#include "lrt_dag.hpp"

#include <stdexcept>
#include <string.h>

#include "litmushelper.hpp"

using std::shared_ptr;
using std::stop_token;
using std::runtime_error;
using std::make_shared;

void Node::worker_fn() {
    auto _tid = litmus_gettid();

    if( !rtparam.T || !rtparam.C || !rtparam.D) {
        fprintf(stderr,"ERROR: Node %d: worker started with period/cost/relative deadline at %llu/%llu/%llu, will not run\n", id, rtparam.T, rtparam.C, rtparam.D);
        exit(1);
        return;
    }

    // First, initialize
    if( fns.init ) {
        fns.init(this, &processData);
    }

    LITMUS_CALL_TID( init_rt_thread() );

    become_rgtask(rtparam.C, rtparam.T, rtparam.D, dag->getReleaseGroupId() );
    LITMUS_CALL_TID( wait_for_ts_release() );

    while( !stopper.stop_requested() ) {
        // Wait for input from all input pipes
        try {
            for (auto& pipe : inputPipes) {
                if( stopper.stop_requested() )
                    break;
                pipe->read();
            }
        } catch (const std::runtime_error& e) {
            break;
        }

        if( stopper.stop_requested() )
            break;

        // Process data
        if( fns.process ) {
            fns.process(this, processData);
        }

        // Write to all output pipes
        try {
            for (auto& pipe : outputPipes) {
                if( stopper.stop_requested() )
                    break;
                pipe->write('R'); // 'R' for "ready"
            }
        } catch (const std::runtime_error& e) {
            break;
        }

        // Sleep until next period
        sleep_next_period();
    }

    if( fns.cleanup ) {
        fns.cleanup(this, processData);
    }

    inputPipes.clear();
    outputPipes.clear();

    // clean up the release group after we're done
    LITMUS_CALL_TID( litmus_releasegroup_remove() );

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}

Node::Node(int id, void* extraData, NodeFNs fns, std::shared_ptr<DAG> dag, stop_token stopper)
    : id(id), extraData(extraData), fns(fns), dag(dag), stopper(stopper), processData(nullptr) {
    memset( &rtparam, 0, sizeof(rtparam) );
}

Node::~Node() {
    if( worker.joinable() )
        worker.join();
}

void DAG::release_nodes() {
    litmus_releasegroup_release(releasegroup_id);
}

void DAG::groupReleaser_fn() {
    if( !period || !releaser_cost ) {
        fprintf(stderr,"ERROR: DAG %d: group releaser started with period/releaser cost at %llu/%llu, will not release\n", id, period, releaser_cost);
        exit(1);
        return;
    }

    auto _tid = litmus_gettid();
    LITMUS_CALL_TID( init_rt_thread() );

    become_periodic(releaser_cost, period);

    LITMUS_CALL_TID( litmus_releasegroup_cache(releasegroup_id) );

    LITMUS_CALL_TID( wait_for_ts_release() );

    while( !stopper.stop_requested() ) {
        LITMUS_CALL_TID( litmus_releasegroup_release_cached() );
        sleep_next_period();
    }

    edges.clear();
    nodes.clear(); // this will begin destruction of nodes

    LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
}

std::shared_ptr<Node> DAG::createNode(int id, void* extraData, NodeFNs fns, stop_token stopper) {
    if( nodes.find(id) != nodes.end() ) {
        throw runtime_error("Node with the same id already exists");
    }

    auto node = make_shared<Node>(id, extraData, fns, shared_from_this(), stopper);
    nodes[id] = node;
    return node;
}

void DAG::addEdge(int parentId, int childId) {
    auto fromNodeIt = nodes.find(parentId);
    auto toNodeIt = nodes.find(childId);
    if (fromNodeIt == nodes.end() || toNodeIt == nodes.end()) {
        throw runtime_error("One or both nodes not found");
    }

    auto pipe = make_shared<Pipe>();
    fromNodeIt->second->addOutputPipe(pipe);
    toNodeIt->second->addInputPipe(pipe);
    edges.emplace_back(parentId, childId);
}

static std::atomic<unsigned int> releasegroup_id_counter = 1;

DAG::DAG(int id, void* extraData, stop_token stopper)
    : id(id), extraData(extraData), stopper(stopper), period(0), releaser_cost(0) {
    auto _tid = litmus_gettid();
    releasegroup_id = releasegroup_id_counter.fetch_add(1);
    LITMUS_CALL_TID( litmus_releasegroup_create(releasegroup_id) );
}

DAG::~DAG() {
    auto _tid = litmus_gettid();
    if( groupReleaser.joinable() )
        groupReleaser.join();
}

Pipe::Pipe() {
    if (pipe(fds) == -1) {
        throw std::runtime_error("Failed to create pipe");
    }
}

Pipe::~Pipe() {
    close(fds[0]);
    close(fds[1]);
}

void Pipe::write(char c) {
    //std::lock_guard<std::mutex> lock(mtx);
    if (::write(fds[1], &c, 1) == -1) {
        throw std::runtime_error("Failed to write to pipe");
    }
}

char Pipe::read() {
    char c;
    if (::read(fds[0], &c, 1) == -1) {
        throw std::runtime_error("Failed to read from pipe");
    }
    return c;
}

void Pipe::read(int size, char* buffer) {
    //std::lock_guard<std::mutex> lock(mtx);
    ssize_t bytesRead = 0;
    while (bytesRead < size) {
        ssize_t result = ::read(fds[0], buffer + bytesRead, size - bytesRead);
        if (result == -1) {
            throw std::runtime_error("Failed to read from pipe");
        }
        bytesRead += result;
    }
}