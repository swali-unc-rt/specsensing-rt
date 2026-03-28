#pragma once

#include <queue>
#include <functional>
#include <barrier>
#include <atomic>
#include <list>
#include <mutex>

#include <stdint.h>
#include <thread>

#include "litmus.h"
#include "mcslock.hpp"
#include <libsmctrl.h>

#include <cuda_runtime_api.h>

static constexpr auto GPC0 = 0x0204080040UL; // 4 TPCs
static constexpr auto GPC1 = 0x0408102081UL; // 6 TPCs
static constexpr auto GPC2 = 0x0810204102UL; // 6 TPCs
static constexpr auto GPC3 = 0x1020408204UL; // 6 TPCs
static constexpr auto GPC4 = 0x2040810408UL; // 6 TPCs
static constexpr auto GPC5 = 0x4081020810UL; // 6 TPCs
static constexpr auto GPC6 = 0x8102041020UL; // 6 TPCs

class FSMLP;

#define QSIZEMAX 10000

// Thus, the mask for everything but FSMLP is:
static constexpr auto GPC_ALL_BUT_GPC6 = GPC0 | GPC1 | GPC2 | GPC3 | GPC4 | GPC5;

// This let's us iterate through a priority queue
template <class T, class S, class C>
S& PQContainer(std::priority_queue<T, S, C>& q) {
	struct HackedQueue : private std::priority_queue<T, S, C> {
		static S& Container(std::priority_queue<T, S, C>& q) {
			return q.* & HackedQueue::c;
		}
	};
	return HackedQueue::Container(q);
}

struct GPURequest {
    unsigned long long abs_deadline;
    uint64_t smctrl_tpcs_allowed;
    uint64_t smctrl_mask_assigned;
    cudaStream_t* stream;
    std::function<void(GPURequest* req)> gpu_launch_fn;
    std::barrier<> job_barrier;
    MCSNode lock_node;
    void* jobData;
    void* nodeData;
};

typedef GPURequest *pGPURequest;

bool cmpGPURequest(GPURequest* a, GPURequest* b);

class FSMLP {
public:
    FSMLP(uint64_t init_mask, int ncpus) : mask(init_mask), m(ncpus), fqsize(0), fqend(0), fqstart(0), sqend(0), sqstart(0), pq(cmpGPURequest), stopped(false) {}
    ~FSMLP() {}

    lt_t submitRequest(GPURequest* req);
    bool SMsAvailable() const { return mask != 0; }
    void processQueues();
    void stop();
private:
    uint64_t mask;
    int m;

    std::priority_queue<GPURequest*, std::vector<GPURequest*>, decltype(&cmpGPURequest)> pq;
    
    pGPURequest FQ[QSIZEMAX];
    int fqend;
    int fqstart;
    int fqsize;

    pGPURequest SQ[QSIZEMAX];
    int sqend;
    int sqstart;
    bool stopped;

    MCSLock fsmlp_lock;
};