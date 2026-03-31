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
TASKGEN = "../../litmusrt-stress-tests/taskgen"
MAINPROG = "../bin/fsmlp-microbench"

def main():
    taskset = []
    schedname = 'GSN-EDF'
    tracename = 'fsmlpmb'

    # print('Set sched to ' + schedname)
    # ss = sp.Popen([LIBLITMUS + '/setsched ' + schedname], shell=True)
    # ss.wait()
    # time.sleep(2.0)

    print('starting up tracing framework...')
    trace = sp.Popen([FEATHER_TRACE + '/st-trace-schedule',tracename], stdin=sp.PIPE)
    time.sleep(5.0)

    print('Start main program now!')
    # mainprogram = sp.Popen([MAINPROG], shell=True)
    # mainprogram.wait()
    time.sleep(30)

    time.sleep(1.0)

    print('Experiment finished, cleaning up..')
    print('specified time has passed, ending tracing')
    trace.communicate(b'\n')
    trace.wait()

    print('letting system settle down')
    time.sleep(1.5)

    # print('restoring Linux scheduler')
    # trace = sp.Popen([LIBLITMUS + '/setsched', 'Linux'], shell=False)
    # trace.wait()

    stats = sp.Popen([FEATHER_TRACE +
												'/st-job-stats schedule_host=*_trace='+tracename+'_cpu=*.bin > fsmlpbench.csv'], shell=True)
    stats.wait()
    print('Stats finished')
    cleaner = sp.Popen(['rm schedule_host=*'], shell=True)
    cleaner.wait()


if __name__ == '__main__':
	main()