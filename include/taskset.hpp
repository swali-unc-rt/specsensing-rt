#pragma once

#include <vector>

extern std::vector<std::vector<std::vector<int>>> all_nodes;
extern std::vector<std::vector<std::vector<int>>> all_edges;
extern std::vector<float> channel_fs;

struct test_entry {
    int m;
    int Hcpu;
    int Hgpu;
    int numChannels;
    std::vector<int> chBatches;
    float objective;
    float computation_time;
};

extern std::vector<test_entry> optEntries;