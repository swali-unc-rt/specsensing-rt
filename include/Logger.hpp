#pragma once

#include <cstddef>

#define RESULTS_FOLDER "results/"
#define CSV_THREADNAMES "threadnames.csv"
#define CSV_ERRORS "errors.csv"

class Log {
public:
    static void logthreadname(const char* threadtypename, int dagID, int nodeID, bool isSink);
    static void logerror(const char* msg, ...);

    static void closeAllLogging();
private:
};