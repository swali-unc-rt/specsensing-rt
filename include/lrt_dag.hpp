#pragma once

#include <mutex>
#include <functional>
#include <vector>
#include <memory>
#include <thread>
#include <map>
#include <tuple>
#include <barrier>

#include <litmus.h>

class Pipe {
public:
    void write(char c);
    char read();
    void read(int size, char* buffer);
    void close();

    Pipe();
    ~Pipe();
private:
    //std::mutex mtx;
    int fds[2];
};

class Node;

using InitFunction = std::function<void(Node*,void**)>;
using ProcessFunction = std::function<void(Node*,void*)>;
using CleanupFunction = std::function<void(Node*,void*)>;

struct NodeFNs {
    ProcessFunction process;
    InitFunction init;
    CleanupFunction cleanup;
};

struct NodeParams {
    lt_t C; // Cost
    lt_t T; // Period
    lt_t D; // Relative Deadline
    lt_t O; // Offset
    lt_t Omaxparent; // Max offset of parents
    lt_t R; // Analytical worst-case response time
};

class DAG;

/* Node object
 *
 * The extraData can be used to track application-specific information, like marginal costs, and is stored as a void*.
 * The initialization nodeFn can be used to initialize data for a job instance. In the initialization function, you 
 * likely want to record the litmus tid of the current thread so that when feather-trace reports response times,
 *   you can correctly correlate it back to the correct node.
 * The process nodeFn is called every job instance.
 * The cleanup nodeFn is called when the system is being stopped, and free any resources allocated in the init function.
 * 
 * Once a node is instantiated, you still need to set the node's rtparameters, that is:
 *  WCET, period, relative deadline
 * Once these values are set, the System will be able to analyze the node, and set the
 *  offset values (O and Omaxparent) and worst-case response time (R) based on the DAG structure.
 */
class Node {
    friend class DAG;
public:
    Node(int id, void* extraData, NodeFNs fns, std::shared_ptr<DAG> dag, std::stop_token stopper);
    ~Node();

    inline int getId() const { return id; }
    inline void* getExtraData() const { return extraData; }
    inline std::shared_ptr<DAG> getDAG() const { return dag; }

    inline lt_t getCost() const { return rtparam.C; }
    inline lt_t getPeriod() const { return rtparam.T; }
    inline lt_t getDeadline() const { return rtparam.D; }
    inline lt_t getOffset() const { return rtparam.O; }
    inline lt_t getMaxParentOffset() const { return rtparam.Omaxparent; }
    inline lt_t getResponseTime() const { return rtparam.R; }

    inline void setCost(lt_t C) { rtparam.C = C; }
    inline void setPeriod(lt_t T) { rtparam.T = T; }
    inline void setDeadline(lt_t D) { rtparam.D = D; }
    inline void setOffset(lt_t O) { rtparam.O = O; }
    inline void setMaxParentOffset(lt_t Omaxparent) { rtparam.Omaxparent = Omaxparent; }
    inline void setResponseTime(lt_t R) { rtparam.R = R; }

    inline bool isSink() const { return outputPipes.empty(); }

    void addInputPipe(std::shared_ptr<Pipe> pipe) {
        inputPipes.push_back(pipe);
    }

    void addOutputPipe(std::shared_ptr<Pipe> pipe) {
        outputPipes.push_back(pipe);
    }

    void startWorker() {
        worker = std::thread(&Node::worker_fn, this);
        workerStartBarrier.arrive_and_wait();
    }
private:
    int id;
    void* extraData;
    void* processData;
    NodeFNs fns;
    NodeParams rtparam;
    std::shared_ptr<DAG> dag;
    std::stop_token stopper;
    std::thread worker;
    std::barrier<> workerStartBarrier;

    std::vector<std::shared_ptr<Pipe>> inputPipes;
    std::vector<std::shared_ptr<Pipe>> outputPipes;

    void worker_fn();
};

/* DAG object
 *
 * The DAG object serves two purposes: track the structure of the DAG and the nodes, and 
 *  also manage the release of all of the nodes in a DAG by placing them in a releasegroup.
 *  All nodes are released at the same time, but precedence constraints are respected by blocking on pipes.
 * 
 * As such, you first initialize a DAG. It will get a releasegroup_id assigned to it automatically.
 * Create nodes exclusively through the DAG's createNode function with a unique id per node in this DAG.
 * 
 * Then create the edge structures. E.g.,
 *   createNode(1,...); createNode(2,...); createNode(3,...); createNode(4,...);
 *   addEdge(1,2); addEdge(1,3); addEdge(2,4); addEdge(3,4);
 * Will create a dag that looks like this:
 *                   1
 *                 /   \
 *                2     3
 *                 \   /
 *                   4
 * There can exist multiple root nodes (any nodes you added that have no predecessors).
 * 
 * Once the DAG structures have been created, the individual nodes must be parameterized
 *  with their WCET, period, and relative deadline. Now the worst-case response time and offsets will
 *  need to be calculated.
 * 
 * Once calculated, the DAG must be assigned a period so that the root node can release periodically.
 *  Ensure setReleaserCost and setPeriod are called on the DAG.
 */
class DAG : public std::enable_shared_from_this<DAG>{
public:
    DAG(int id, void* extraData, std::stop_token stopper);
    ~DAG();

    inline lt_t getReleaserCost() const { return releaser_cost; }
    inline void setReleaserCost(lt_t cost) { releaser_cost = cost; }
    inline lt_t getPeriod() const { return period; }
    inline void setPeriod(lt_t period) { this->period = period; }
    inline int getId() const { return id; }
    inline lt_t getE2EResponseTime() const { return e2e_R; }
    inline void setE2EResponseTime(lt_t R) { e2e_R = R; }


    std::shared_ptr<Node> createNode(int id, void* extraData, NodeFNs fns, std::stop_token stopper);

    void addEdge(int parentId, int childId);

    unsigned int getReleaseGroupId() const { return releasegroup_id; }

    void startReleaser() {
        groupReleaser = std::thread(&DAG::groupReleaser_fn, this);
    }

    void startAllNodes() {
        for( auto& [id, node] : nodes ) {
            node->startWorker();
        }
    }

    bool allNodesFinished() const {
        for( auto& [id, node] : nodes ) {
            if( node->worker.joinable() )
                return false;
        }
        return true;
    }

    void release_nodes();
    std::shared_ptr<DAG> makeCopy();
private:
    int id;
    unsigned int releasegroup_id;
    void *extraData;

    lt_t period;
    lt_t releaser_cost;
    lt_t e2e_R;
    
    std::map<int, std::shared_ptr<Node>> nodes;
    std::vector<std::pair<int,int>> edges; // fromid, toid
    std::stop_token stopper;

    std::thread groupReleaser;

    void groupReleaser_fn();
};