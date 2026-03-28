#include "Logger.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <cstdarg>

#include <thread>
#include <chrono>
#include <mutex>

#include <litmus.h>

static FILE* threadnamecsv = nullptr;
//std::mutex threadnamemutex;
void Log::logthreadname(const char* threadtypename, int dagID, int nodeID, bool isSink) {
    //std::lock_guard<std::mutex> lock(threadnamemutex);

    if( !threadnamecsv ) {
        threadnamecsv = fopen(RESULTS_FOLDER CSV_THREADNAMES, "w");
        fprintf(threadnamecsv, "threadid, ThreadType, dagid, nodeid, isSink\n");
    }
    fprintf(threadnamecsv, "%d, %s, %d, %d, %d\n", litmus_gettid(), threadtypename, dagID, nodeID, (int)isSink);
    fflush(threadnamecsv);
}

static FILE* errorcsv = nullptr;
//std::mutex errormutex;
void Log::logerror(const char* msg, ...) {
    //std::lock_guard<std::mutex> lock(errormutex);
    
    if( !errorcsv ) {
        errorcsv = fopen(RESULTS_FOLDER CSV_ERRORS, "w");
        fprintf(errorcsv, "threadid, msg\n");
    }
    va_list args;
    va_start(args, msg);
    fprintf(errorcsv, "%d, ", litmus_gettid());
    vfprintf(errorcsv, msg, args);
    va_end(args);
    fflush(errorcsv);
    // output to stdout as well
    
    // printf("%d, ", litmus_gettid());
    // va_start(args, msg);
    // vprintf(msg, args);
    // va_end(args);
    // fflush(stdout);
}

void Log::closeAllLogging() {
    if( threadnamecsv ) {
        fclose(threadnamecsv);
        threadnamecsv = nullptr;
    }
    if( errorcsv ) {
        fclose(errorcsv);
        errorcsv = nullptr;
    }
}