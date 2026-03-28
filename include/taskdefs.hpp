#pragma once

#include "lrt_dag.hpp"
#include "rfmlp.hpp"
#include "fsmlp.hpp"

struct jobData {
    void* deviceInput;
    void* deviceOutput;
    cudaStream_t stream;
    RFML* model;
    marginalCost mcost;
    int smlp_od;
    GPURequest req;
};

void initED(Node* node, void** processData);
void initAMC(Node* node, void** processData);
void initSEI(Node* node, void** processData);
void initGEO(Node* node, void** processData);
void simple_initED(Node* node, void** processData);
void simple_initAMC(Node* node, void** processData);
void simple_initSEI(Node* node, void** processData);
void simple_initGEO(Node* node, void** processData);

void jobED(Node* node, void* processData);
void jobAMC(Node* node, void* processData);
void jobSEI(Node* node, void* processData);
void jobGEO(Node* node, void* processData);
void simple_jobED(Node* node, void* processData);
void simple_jobAMC(Node* node, void* processData);
void simple_jobSEI(Node* node, void* processData);
void simple_jobGEO(Node* node, void* processData);

// void cleanupED(Node* node, void* processData);
// void cleanupAMC(Node* node, void* processData);
// void cleanupSEI(Node* node, void* processData);
// void cleanupGEO(Node* node, void* processData);
void simple_cleanupED(Node* node, void* processData);
void simple_cleanupAMC(Node* node, void* processData);
void simple_cleanupSEI(Node* node, void* processData);
void simple_cleanupGEO(Node* node, void* processData);

NodeFNs getNodeFns(RFMLType fntype);
NodeFNs getNodeFnsSimple(RFMLType fntype);
void GPUManagementTask(std::stop_token stopper, lt_t releaser_cost, lt_t period, std::shared_ptr<FSMLP> fsmlp);