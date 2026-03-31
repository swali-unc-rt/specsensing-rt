from __future__ import print_function
import subprocess as sp
import time
import signal
import random
import os
import csv
#import numpy as np

LIBLITMUS = "../../liblitmus"
FEATHER_TRACE = "../../feather-trace-tools"

def main():
    tracename = 'specsensingrt1'
    stats = sp.Popen([FEATHER_TRACE +
												'/st-job-stats schedule_host=*_trace='+tracename+'_cpu=*.bin > response_times_'+tracename+'.csv'], shell=True)
    stats.wait()
    print('Stats finished')
    cleaner = sp.Popen(['rm schedule_host=*'], shell=True)
    cleaner.wait()

if __name__ == '__main__':
	main()