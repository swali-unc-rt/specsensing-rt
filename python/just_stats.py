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
    stats = sp.Popen([FEATHER_TRACE +
												'/st-job-stats schedule_host=*_trace=test1_cpu=*.bin > response_times.csv'], shell=True)
    stats.wait()
    print('Stats finished')

if __name__ == '__main__':
	main()