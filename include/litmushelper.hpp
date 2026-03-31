#pragma once
/* litmushelper.hpp - helper functions for litmusrt stress tests
 *
 * These are just some macros and functions to make becoming a litmus task
 * more readable.
 */

#include <stdio.h>
#include <stdlib.h>

#include <litmus.h>

// Calls a litmus function. If the function fails, it also outputs the error message.
#define LITMUS_CALL( exp ) do {                     \
    int _ret;                                       \
    _ret = exp;                                     \
    if (_ret != 0)                                  \
        fprintf(stdout, "%s failed: %m\n", #exp);   \
    else                                            \
        fprintf(stdout, "%s ok.\n", #exp);          \
} while (0)

// Calls a litmus function but with a thread id in the output. If the function fails, it also outputs the error message.
// #define LITMUS_CALL_TID( exp ) do {               \
//     int _ret;                                       \
//     _ret = exp;                                     \
//     if (_ret != 0)                                  \
//         fprintf(stderr, "%d: %s failed: %m\n", _tid, #exp);   \
//     else                                            \
//         fprintf(stderr, "%d: %s ok.\n", _tid, #exp );          \
// } while (0)

#define LITMUS_CALL_TID( exp ) do {               \
    int _ret;                                       \
    _ret = exp;                                     \
} while (0)


// Become a releasegroup task. rgid cannot be zero. Task will not be immediately released.
void become_rgtask(lt_t exec_cost, lt_t period, unsigned int rgid);
void become_rgtask(lt_t exec_cost, lt_t period, lt_t relative_deadline, unsigned int rgid);

// Become a periodic task.
void become_periodic(lt_t exec_cost, lt_t period);
void become_periodic(lt_t exec_cost, lt_t period, lt_t relative_deadline);
void become_periodic(lt_t offset, lt_t exec_cost, lt_t period, lt_t relative_deadline);

// Release the taskset. The taskset will be released after the delay specified, rounded up to the nearest quantum.
int release_taskset(lt_t delay, lt_t quantum);