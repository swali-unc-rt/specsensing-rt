#pragma once

#include "trthelpers.hpp"
#include "rfmlp.hpp"
#include "lrt_dag.hpp"
#include "SignalGenerator.hpp"
#include "fsmlp.hpp"

#include <vector>
#include <queue>
#include <thread>

#include <litmus.h>

#define TEST_ITER 10
#define GPU_MAN_TASK_PERIOD_MS 1
#define SECONDS_TO_RUN 15
#define LOG_NAME "results/log-10-smlp.csv"
//#define GPCUSE GPC0 | GPC1 | GPC2
#define SMLP_GPCUSE GPC0 | GPC1 | GPC2 | GPC3 | GPC4 | GPC5 | GPC6
#define SMLP_NAMESPACE "./smlp_lock_od"
#define SMLP_OD_ID 1
#define LOCK_TYPE_FOR_RUN LockType::SMLP

//#define DO_NO_GPU_WORK

// make this different than the SMLP's flags
#define FSMLP_GPCUSE GPC0 | GPC1 | GPC2 | GPC3 | GPC4 | GPC5 | GPC6

enum class LockType {
    SMLP,
    FSMLP,
};

struct NodeData {
    float fs;
    marginalCost mcost;
    int batchCount;
    int maxTPCcount;
};

class DAGInvoker {
public:
    DAGInvoker(lt_t period, std::stop_token stopper) : period(period), stopper(stopper), invokeStartBarrier(2) {}
    ~DAGInvoker();

    void addDAG(std::shared_ptr<DAG> dag) {
        dags.push(dag);
    }

    void start();

    void periodicRelease();
private:
    std::queue<std::shared_ptr<DAG>> dags;
    lt_t period;
    std::stop_token stopper;
    std::barrier<> invokeStartBarrier;
    
    std::thread periodic_releaser;
};

class RTSystem {
public:
    ~RTSystem();

    void init();

    static RTSystem* _instance;
    static RTSystem* Instance();

    void createSystem(int numDAGs);
    void calculateOffsetsAndDeadlines();

    void addDAG(std::shared_ptr<DAG> dag) {
        dags.push_back(dag);
    }
    
    // DAG* getDAG(size_t index) {
    //     if( index >= dags.size() ) return nullptr;
    //     return dags[index];
    // }

    LockType getLockType() const {
        return lockType; // TODO: change this if we implement FSMLP
    }

    void setLockType(LockType type) {
        lockType = type;
    }

    void start();
    void stop();

    void cleanupDags();

    inline std::stop_token getStopToken() { return stopper.get_token(); }
    inline std::shared_ptr<FSMLP> getFSMLP() { return fsmlp; }
private:
    RTSystem();

    LockType lockType;
    std::stop_source stopper;
    std::thread gpu_man_task;
    std::thread timer_thread;

    std::vector<std::shared_ptr<DAG>> dags;
    std::vector<std::shared_ptr<DAGInvoker>> invokers;
    std::shared_ptr<FSMLP> fsmlp;

    void stopTimer(std::stop_token stop_tok);
};


extern SignalGenerator* channel;
extern void* channelBuffer;
extern AMC* amc;
extern SEI* sei;
extern GEO* geo;