from EC_qutip_lib import *
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

#########################
foldername='test_new'
Name='test'
#########################
parallel=True
NCores=cpu_count()
#########################
Dmin=0.3
Dmax=0.45
Datapoints=4
NP=1
Resolution=1111
Num_max=10
qmax=Num_max*np.sqrt(np.pi)
cutoff=200
M_rounds_Min=1
M_rounds_Max=10
NFock=100
#########################

		
if __name__ == '__main__':
	Params=Parameters(Dmin, Dmax, Datapoints, M_rounds_Min, M_rounds_Max, NP, Num_max, Resolution, cutoff, NFock, parallel, NCores)
	#print number of parameters
	print('Number of parameters:', len(Params.Domain_t_MDomain))
	
	if os.path.isdir('%s'%(foldername)):
		run(["rm", "-r", '%s'%(foldername)])
	os.mkdir('%s'%(foldername))
	
	if parallel:
		print('Cores initialized:', NCores)
	else:
		print('Sequential mode')

	start=time.time()
	
	PL=ExecMC_MUDec(Params)
	elapsed=time.time()
	
	elapsed=elapsed-start
	print('Time:', elapsed)
	PLMean=PL[0]
	PLsig=PL[1]

	PLMean,PLsig=list(map(list, zip(*PLMean))),list(map(list, zip(*PLsig)))

	labels=['0','Forward','Forwardcoh','MLD']

	file=open('%s/PLog_%s_Res%i_NP%i_M%ito%i.txt'%(foldername,Name,Resolution,NP,M_rounds_Min,M_rounds_Max),'w')
	file.write(str(labels))
	file.write('\n')
	file.write('DM='+str(list(Params.Domain_t_MDomain)))
	file.write('\n')
	file.write('Mean Vals: \n')
	for i in range(len(labels)):
		file.write(str(list(PLMean[i]))+'\n')
	file.write('Standard: \n')
	for i in range(len(labels)):
		file.write(str(list(PLsig[i]))+'\n')
	file.write('\n Execution time: %f s'%(elapsed))
	file.close()
	
	
	for i in range(len(labels)):
		for j in range(len(Params.DDomain)):
			file=open('%s/%s_D%i.txt'%(foldername, labels[i],round(1000*Params.DDomain[j])),'w')
			file.write('n_steps\t logical_error_rate\t std \n')
			file.close()
	for i in range(len(labels)):
		for j in range(len(Params.Domain_t_MDomain)):
			delta=Params.Domain_t_MDomain[j][0]
			file=open('%s/%s_D%i.txt'%(foldername, labels[i],round(1000*delta)),'a')
			file.write('%.3f \t %f \t %.3f  \n'%(Params.Domain_t_MDomain[j][1],PLMean[i][j],PLsig[i][j]))
			file.close()