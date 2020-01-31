from EC_qutip_lib_sample import *
import os
from subprocess import run
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import scipy.special as special
import scipy.signal as signal
import scipy.integrate as integrate
from numpy import pi, exp, sin, cos
import concurrent.futures
from scipy.optimize import minimize_scalar
import functools
import itertools as iter
from multiprocessing import Pool
from multiprocessing import cpu_count
import time
import qutip
from tabulate import tabulate

#########################
foldername='timing'
Name='test'
#########################
parallel=True
NCores=cpu_count()
#########################
Dmin=0.3
Dmax=0.4
Datapoints=2
NP=1
Resolution=1111
Num_max=10
qmax=Num_max*np.sqrt(np.pi)
cutoff=200
M_rounds_Min=10
M_rounds_Max=10
NFock=100
#########################
Forward_samples=100
#########################
		
if __name__ == '__main__':
	Params=Parameters(Dmin, Dmax, Datapoints, M_rounds_Min, M_rounds_Max, NP, Num_max, Resolution, cutoff, NFock, parallel, NCores, Forward_samples)
	#print number of parameters
	print('Number of parameters:', len(Params.Domain_t_MDomain))
	
	if os.path.isdir('%s'%(foldername)):
		run(["rm", "-r", '%s'%(foldername)])
	os.mkdir('%s'%(foldername))
	
	QT=random.choice(Params.QX,size=M_rounds_Max)
	PT=random.choice(Params.QX,size=M_rounds_Max)
	
	startMLD=time.time()
	XMLD=MLD_ini0(QT,PT,Dmin,M_rounds_Max,Params)
	endMLD=time.time()-startMLD
	
	startForward=time.time()
	XForward=minimum_energy_decoding_gkp_sample0(QT, Dmin, np.sqrt(np.pi), Params)
	endForward=time.time()-startForward
	
	file=open('%s/%s_%i.txt'%(foldername,Name,Forward_samples),'w')
	file.write("Time \n MLD: %.3f \n Forward %.5f"%())
	file.close()