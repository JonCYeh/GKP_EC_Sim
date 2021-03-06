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
foldername='test_photons_new'
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
M_rounds_Min=5
M_rounds_Max=5
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
	
	Photons=Exec_Photons(Params)#lenDN x 2*Mrounds+1
	elapsed=time.time()
	
	elapsed=elapsed-start
	print('Time:', elapsed)
	Photons_mean=Photons[0]
	Photons_sig=Photons[1]
	
	Deltaq_mean=Photons[2]
	Deltaq_sig=Photons[3]
	
	Deltap_mean=Photons[4]
	Deltap_sig=Photons[5]



	file=open('%s/Photons_%s_Res%i_NP%i_M%i.txt'%(foldername,Name,Resolution,NP,M_rounds_Max),'w')
	file.write("Photons")
	file.write('\n')
	file.write('DM='+str(list(Params.Domain_t_MDomain)))
	file.write('\n')
	file.write('Mean Vals: \n')
	file.write(str(list(Photons_mean))+'\n')
	file.write('Standard: \n')
	file.write(str(list(Photons_sig))+'\n')
	file.write('\n Execution time: %f s'%(elapsed))
	file.close()
	
			
		
	header=["M", "Photons_mean", "Photons_sig", "Delta_q_mean", "Delta_q_sig", "Delta_p_mean", "Delta_p_sig"]	
	for i in range(len(Params.Domain_t_MDomain)):
		temp=[[j/2, Photons_mean[i][j], Photons_sig[i][j], Deltaq_mean[i][j], Deltaq_sig[i][j], Deltap_mean[i][j], Deltap_sig[i][j]] for j in range(len(Photons_mean[i]))]
		file=open('%s/%s_D%i.txt'%(foldername, "Photons",round(1000*Params.Domain_t_MDomain[i][0])),'w')
		file.write(tabulate(temp,header,tablefmt="plain"))
		file.close()