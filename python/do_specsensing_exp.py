from __future__ import print_function
import subprocess as sp
import time
import signal
import random
import os
import csv
import numpy as np

LIBLITMUS = "../../liblitmus"
FEATHER_TRACE = "../../feather-trace-tools"

def main():
    CPU_TASK_COUNT = 0
    CPU_TASK_PERIOD_US = 10000
    CPU_TASK_COST_US = 2000
    taskset = []
    schedname = 'GSN-EDF'
    numpages = 500
    Duration = 60
    tracename = 'specsensingrt1'

    def cleanup_rtspin(*args):
        for task in taskset:
            task.kill()
        exit()

    signal.signal(signal.SIGINT, cleanup_rtspin)
    signal.signal(signal.SIGSEGV, cleanup_rtspin)
    signal.signal(signal.SIGTERM, cleanup_rtspin)

    print('Set sched to ' + schedname)
    ss = sp.Popen([LIBLITMUS + '/setsched ' + schedname], shell=True)
    ss.wait()
    time.sleep(2.0)

    print('starting up tracing framework...')
    trace = sp.Popen([FEATHER_TRACE + '/st-trace-schedule',tracename], stdin=sp.PIPE)
    time.sleep(5.0)

    print('starting CPU tasks..')
    for i in range(CPU_TASK_COUNT):
        task = sp.Popen([LIBLITMUS + '/rtspin', '-w', 'm', str(numpages), '-d', str(CPU_TASK_PERIOD_US/1000000), str(CPU_TASK_COST_US/1000000), str(CPU_TASK_PERIOD_US/1000000), str(Duration)], shell=True)
        taskset.append(task)
        time.sleep(0.1)
    time.sleep(2)

    print('Start main program now!')
    # mainprogram = sp.Popen(['../bin/specsensing-rt'], shell=False)
    time.sleep(10)

    # print('Releasing tasks..')
    # releaser = sp.Popen([LIBLITMUS + '/release_ts', '-q', '1000'])
    # releaser.wait()

    print('Experiment running for {} seconds..'.format(Duration))
    time.sleep(Duration)
    print('Experiment finished, cleaning up..')
    print('specified time has passed, ending tracing')
    trace.communicate(b'\n')
    trace.wait()

    print('letting system settle down')
    for task in taskset:
        task.kill()
        time.sleep(1.5)
    time.sleep(3)
    #print('killing main program (if not already)')
    #mainprogram.kill()
    time.sleep(1.5)

    print('restoring Linux scheduler')
    trace = sp.Popen([LIBLITMUS + '/setsched', 'Linux'], shell=False)
    trace.wait()

    stats = sp.Popen([FEATHER_TRACE +
												'/st-job-stats schedule_host=*_trace='+tracename+'_cpu=*.bin > response_times.csv'], shell=True)
    stats.wait()
    print('Stats finished')

    #os.remove('smlp_lock_od') # Clean up lock file


if __name__ == '__main__':
	main()