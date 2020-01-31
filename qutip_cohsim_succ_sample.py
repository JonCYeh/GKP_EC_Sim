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
foldername='test_succ500'
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
	
	if parallel:
		print('Cores initialized:', NCores)
	else:
		print('Sequential mode')

	start=time.time()
	
	PL=ExecMC_MUDec_succ(Params)
	elapsed=time.time()
	
	elapsed=elapsed-start
	print('Time:', elapsed)
	PLMean=PL[0]
	PLsig=PL[1]

	labels=['0','Forwardcoh','Forwardsample','MLD']

	file=open('%s/PLog_%s_Res%i_NP%i_M%ito%i.txt'%(foldername,Name,Resolution,NP,M_rounds_Min,M_rounds_Max),'w')
	file.write('DM='+str(list(Params.Domain_t_MDomain)))
	file.write('\n')
	file.write('Mean Vals: \n')
	file.write(str(list(PLMean))+'\n')
	file.write('Standard: \n')
	file.write(str(list(PLsig))+'\n')
	file.write('\n Execution time: %f s'%(elapsed))
	file.close()
	
	header=["M", "P_0", "s_P_0", "P_Forward", "s_P_Forward", "P_Forward_sample", "s_P_Forward_sample","P_MLD", "s_P_MLD"]
	for i in range(len(Params.Domain_t_MDomain)):
		temp=[[j+1,PLMean[i][j][0],PLsig[i][j][0],PLMean[i][j][1],PLsig[i][j][1],PLMean[i][j][2],PLsig[i][j][2],PLMean[i][j][3],PLsig[i][j][3] ] for j in range(len(PLMean[i]))]
		file=open('%s/%s_D%i.txt'%(foldername, Name,round(1000*Params.Domain_t_MDomain[i][0])),'w')
		file.write(tabulate(temp,header,tablefmt="plain"))
		file.close()