#include "litmushelper.hpp"

void become_periodic(lt_t exec_cost, lt_t period, lt_t relative_deadline) {
    struct rt_task param;
    auto _tid = litmus_gettid();
    
    init_rt_task_param(&param);

    // User supposedly only needs to set these two parameters
    param.exec_cost = exec_cost;
    param.period = period;

    param.relative_deadline = relative_deadline;
    param.phase = 0;
    param.cpu = 0; // only relevant for non-global and clustered
    param.cls = RT_CLASS_SOFT;
    param.budget_policy = NO_ENFORCEMENT;
    param.release_policy = TASK_PERIODIC; // no need to infer sporadic releases

    // Set our parameters and begin real-time mode
    LITMUS_CALL_TID( set_rt_task_param(_tid, &param) );
    LITMUS_CALL_TID( be_migrate_to_cpu(param.cpu) );
    LITMUS_CALL_TID( task_mode(LITMUS_RT_TASK) );
}

void become_periodic(lt_t exec_cost, lt_t period) {
    become_periodic(exec_cost, period, period);
}

int release_taskset(lt_t delay, lt_t quantum) {
    auto _tid = litmus_gettid();
    lt_t release_time = litmus_clock() + delay;
    release_time = (release_time/quantum + 1) * quantum; // align to quantum
    if( release_ts(&release_time) < 0 ) {
        fprintf(stderr,"%d: release_ts failed: %m\n", _tid);
        LITMUS_CALL_TID( task_mode(BACKGROUND_TASK) );
        printf("set sched back to Linux..\n");
        system(LIBLITMUS_LIB_DIR "/setsched Linux");
        return -1;
    }

    return 0;
}

void become_rgtask(lt_t exec_cost, lt_t period, lt_t relative_deadline, unsigned int rgid) {
    struct rt_task param;
    auto _tid = litmus_gettid();
    
    init_rt_task_param(&param);

    // User supposedly only needs to set these two parameters
    param.exec_cost = exec_cost;
    param.period = period;

    param.relative_deadline = relative_deadline;
    param.phase = 0;
    param.cpu = 0; // only relevant for non-global and clustered
    param.cls = RT_CLASS_SOFT;
    param.budget_policy = NO_ENFORCEMENT;
    param.release_policy = TASK_RELEASEGROUP; // no need to infer sporadic releases
    param.releasegroup_id = rgid;

    // Set our parameters and begin real-time mode
    LITMUS_CALL_TID( set_rt_task_param(_tid, &param) );
    LITMUS_CALL_TID( be_migrate_to_cpu(param.cpu) );
    LITMUS_CALL_TID( task_mode(LITMUS_RT_TASK) );
}

void become_rgtask(lt_t exec_cost, lt_t period, unsigned int rgid) {
    become_rgtask(exec_cost, period, period, rgid );
}