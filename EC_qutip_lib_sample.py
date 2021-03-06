"""
Author: Jonathan Conrad
A library of functions to simulate GKP Steane-EC with finitely squeezed ancillas as investigated in my thesis
"Tailoring GKP Error Correction to Real Noise".
"""
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
from math import factorial
from scipy.special import eval_hermite

class Parameters():
	"""
	container for all global objects
	"""
	
	def __init__(self, Dmin, Dmax, Datapoints, M_rounds_Min, M_rounds_Max, NP, Num_max, Resolution, cutoff, NFock, parallel, NCores, Forward_samples):
		
		DDomain=np.linspace(Dmin,Dmax,Datapoints)
		MDomain=np.arange(M_rounds_Min,M_rounds_Max+1,1)
		qmax=Num_max*np.sqrt(np.pi)
		
		self.NP=NP
		self.DDomain=DDomain
		self.MDomain=MDomain
		self.Domain_t_MDomain=list(iter.product(DDomain,MDomain))
		self.qmax=qmax
		self.Num_max=Num_max
		self.Resolution=Resolution
		self.cutoff=cutoff
		self.NFock=NFock
		self.parallel=parallel
		self.NCores=NCores
		self.Forward_samples=Forward_samples
		
		Stabilizer_q=qutip.displace(NFock,1j*np.sqrt(2*np.pi))
		Stabilizer_p=qutip.displace(NFock,np.sqrt(2*np.pi))
		Z=qutip.displace(NFock,1j*np.sqrt(np.pi/2))
		X=qutip.displace(NFock,np.sqrt(np.pi/2))
		self.stab_q=Stabilizer_q
		self.stab_p=Stabilizer_p
		self.X=X
		self.Z=Z
		
		QX, b0qmat2, b0pmat2, g0, g1, g_range, b_range, q_n_mat, p_n_mat, g0q=init_global_objects(self)
		
		self.QX=QX
		self.b0qmat2=b0qmat2
		self.b0pmat2=b0pmat2
		self.g0=g0
		self.g1=g1
		self.g_range=g_range
		self.b_range=b_range
		self.q_n_mat=q_n_mat
		self.p_n_mat=p_n_mat
		self.g0q=g0q


def init_global_objects(par):
	"""
	initialize global objects that are not parameters
	"""
	g_range=np.arange(-par.cutoff, par.cutoff,1).reshape((1,-1))
	b_range=np.arange(-2*par.cutoff,2*par.cutoff,1).reshape((1,-1))
	QX = np.linspace(-par.qmax,par.qmax,par.Resolution)

	qtile=np.tile(QX,(len(QX),1)) ###the array qx stacked len(qx)times
	qmat=qtile-np.transpose(qtile)

	ptile=np.tile(QX,(len(QX),1)) ###the array qx stacked len(qx)times
	pmat=ptile+np.transpose(ptile) #the 0-axis represents q1, the 1-axis qt


	b0qmat2={}
	b0pmat2={}
	g0={}
	g1={}
	g0q={}
	for D in par.DDomain:
		b0qmat2[D]=b_0(qmat,D, b_range)**2
		b0pmat2[D]=b_0(pmat,D, b_range)**2
		g0[D]=g_0n(D, par)
		g1[D]=(par.X*g_0n(D, par)).unit()
		
		temp_g=g_0(QX ,D, g_range)
		g0q[D]=temp_g/np.sqrt(np.sum(np.abs(temp_g)**2))
		
	q_n_mat=q_fockn_mat(QX,par.NFock)	
	p_n_mat=p_fockn_mat(QX,par.NFock)	
		
	return QX, b0qmat2, b0pmat2, g0, g1, g_range, b_range, q_n_mat, p_n_mat, g0q

def q_fockn(q,n):
	"""
	returns <q|n>.
	"""

	prefactor=1/np.sqrt(np.sqrt(np.pi)*factorial(n) *2**n)*np.exp(-q**2/2)
	return prefactor*eval_hermite(n,q)

def q_fockn_mat(qx, NFock):
	"""
	matrix m_ij=<q_i|j> .
	"""

	temp=[[q_fockn(q,n) for n in range(NFock)] for q in qx]
	return np.asarray(temp)

def p_fockn_mat(qx, NFock):
	"""
	matrix m_ij=<p_i|j> .
	"""
	phasemat=[[np.exp(-1j*p*q) for q in qx] for p in qx]
	phasemat=np.asarray(phasemat)

	return np.dot(phasemat,q_fockn_mat(qx, NFock))
	
def g_0(q ,D, g_range):
	"""
	\gamma wavefunction
	"""
	qshape=q.shape
	Norm=np.sqrt(2)/np.pi**(1/4)
	temp=np.exp(-2*D**2*g_range**2*np.pi-(q.reshape((-1,1))-2*g_range*np.sqrt(np.pi))**2/(2*D**2))
	return (Norm*np.sum(temp, axis=-1)).reshape(qshape)


def b_0(q,D, b_range):
	"""
	\beta wavefunction
	"""
	qshape=q.shape
	Norm=1/np.pi**(1/4)
	temp=np.exp(-0.5*D**2*b_range**2*np.pi-(q.reshape(-1,1)-b_range*np.sqrt(np.pi))**2/(2*D**2))
	return (Norm*np.sum(temp, axis=-1)).reshape(qshape)

def LogBin(q):
	"""
	square wave-function. 1 if q \in [1/2 \sqrt(\pi),3/2 \sqrt(\pi)/2] + \mathbb{Z} \sqrt(\pi)
	"""
	return 0.5*(1+signal.square(np.sqrt(np.pi)*(q-0.5*np.sqrt(np.pi)),duty=0.5))

def Decode(state,qx,basis, par):
	"""
	implements exact ML decoding given final state and outputs logical probabilities.
	"""

	state=state.unit()
	if basis=='z':
		P=Pqt(state,qx, par)
	elif basis=='x':
		P=Ppt(state,qx, par)

	Prob0=integrate.simps(P*(1-LogBin(qx)),x=qx) #LogBin=1 on 1-subspace
	Prob1=integrate.simps(P*LogBin(qx),x=qx)
	Norm=Prob0+Prob1
	Prob0, Prob1=Prob0/Norm, Prob1/Norm

	return Prob1>Prob0, {0:Prob0, 1:Prob1}

def g_0n(D, par):
	"""
	returns a D-squeezed 0-ancilla
	"""

	r=np.log(1/D)
	psi0=qutip.squeeze(par.NFock, r) * qutip.basis(par.NFock,0)
	psi = qutip.Qobj()
	for n in np.arange(-par.Num_max, par.Num_max+1):
		psi+=np.exp(-2*np.pi*D**2 * n**2) * qutip.displace(par.NFock, n*np.sqrt(2*np.pi)) * psi0
	return psi.unit()

	
def b_0n(D, par): ##returns a D-squeezed + ancilla
	"""
	returns a D-squeezed +-ancilla
	"""	
	r=np.log(1/D)
	psi0=qutip.squeeze(par.NFock, r) * qutip.basis(par.NFock,0)
	psi = qutip.Qobj()
	for n in np.arange(-par.Num_max, par.Num_max+1):
		psi+=np.exp(-np.pi*D**2/2 * n**2) * qutip.displace(par.NFock, n*np.sqrt(np.pi/2)) * psi0	
	return psi.unit()
		
def stabilizer_projq(qt,D, par):
	"""
	returns stabilizer projector for q-measurement using Villain approximation eq. (3.39)
	"""
	I=qutip.qeye(par.NFock)
	pos=qutip.position(par.NFock)

	op=pos-I*(qt)
	op2=D**2/2*op**2
	cosop=(2*np.sqrt(np.pi)*op).cosm()/(4*np.pi*D**2)
	expop=(-op2+cosop).expm()

	return expop

def stabilizer_projp(pt,D,par):
	"""
	returns stabilizer projector for p-measurement using Villain approximation eq. (3.39)
	"""
	I=qutip.qeye(par.NFock)
	mom=qutip.momentum(par.NFock)

	op=mom+I*(pt)
	op2=D**2/2*op**2
	cosop=(2*np.sqrt(np.pi)*op).cosm()/(4*np.pi*D**2)
	expop=(-op2+cosop).expm()

	return expop


def Pqt(state,qx, par): 
	"""
	probability to measure q given a state
	"""
	state_np=state.full()[:,0]
	psi_q=np.dot(par.q_n_mat,state_np)
	Pq=np.abs(psi_q)**2

	return Pq/np.sum(Pq)

def Ppt(state,qx, par):
	"""
	probability to measure p given a state
	"""
	state_np=state.full()[:,0]
	psi_p=np.dot(par.p_n_mat,state_np)
	Pp=np.abs(psi_p)**2

	return Pp/np.sum(Pp)

def stabilizer_probq(state,qx,D, par):
	"""
	return probability distribution over to measure qt given a. according to eq. (3.32)
	"""
	Pq=Pqt(state,qx, par)
	b0_q2=par.b0qmat2[D]

	abqmat=np.dot(np.diag(Pq),b0_q2) # b_0(qmat) evaluates b_0(q1-qt) for all pairs q1, qt
	#abqmat equals the matrix over i,j {a(q_i) b_0((q1)_i-(qt)_j) 

	temp=np.abs(integrate.simps(abqmat,x=qx,axis=0)) #integration over 0 axis gives unnormalized stabilizer probabilities.
	return temp/np.sum(temp)

def stabilizer_probp(state,qx,D, par):
	"""
	return probability distribution over to measure pt given a. according to eq. (3.30)
	"""
	Pp=Ppt(state,qx, par)
	b0_p2=par.b0pmat2[D]

	abpmat=np.dot(np.diag(Pp), b0_p2)
	#abqmat equals the matrix over i,j {a((q1)_i) b_0((q1)_i-(qt)_j) 

	temp=np.abs(integrate.simps(abpmat,x=qx,axis=0)) #integration over 0 axis gives unnormalized stabilizer probabilities.
	return temp/np.sum(temp)


	
def G(state,D, par):
	"""
	Execute one round of EC
	"""
	qx=par.QX

	pprobs=stabilizer_probp(state,qx,D, par)
	PT=random.choice(qx, p=pprobs)
	meas_projector_p=stabilizer_projp(PT,D, par)
	state=meas_projector_p*state
	state=state.unit()

	qprobs=stabilizer_probq(state,qx,D, par)
	QT=random.choice(qx,p=qprobs)
	meas_projector_q=stabilizer_projq(QT,D, par)
	state=meas_projector_q*state
	state=state.unit()
	
	return state,PT,QT
	
def GP(state,D, par):
	"""
	Execute one round of stabilizer-p mesurement
	"""
	qx=par.QX

	pprobs=stabilizer_probp(state,qx,D, par)
	PT=random.choice(qx, p=pprobs)
	meas_projector_p=stabilizer_projp(PT,D, par)
	state=meas_projector_p*state
	state=state.unit()
	
	return state,PT
	
	
def GQ(state,D, par):
	"""
	Execute one round of stabilizer-q measurement
	"""
	qx=par.QX

	qprobs=stabilizer_probq(state,qx,D, par)
	QT=random.choice(qx,p=qprobs)
	meas_projector_q=stabilizer_projq(QT,D, par)
	state=meas_projector_q*state
	state=state.unit()
	
	return state,QT
	
def G_deterministic(state,D,PT,QT, par):
	"""
	Execute one round of EC with given PT,QT
	"""
	qx=par.QX
	
	meas_projector_p=stabilizer_projp(PT,D, par)
	state=meas_projector_p*state
	state=state.unit()

	meas_projector_q=stabilizer_projq(QT,D, par)
	state=meas_projector_q*state
	state=state.unit()
	
	return state

def unbiased_decoder(QT, PT, D, Mrounds, par):
	
	b0=g_0n(D, par)
	b1=g_1n(D, par)
	
	for i in range(len(QT)):
		b0=G_deterministic(b0,D,PT[i],QT[i], par)
		b1=G_deterministic(b1,D,PT[i],QT[i], par)
	
	XML_b0, Pdict_b0=Decode(b0,par.QX,'z', par)
	XML_b1, Pdict_b1=Decode(b1,par.QX,'z', par)
	
	gap=np.zeros(2)
	gap[0]=np.abs(Pdict_b0[0]-Pdict_b0[1])
	gap[1]=np.abs(Pdict_b1[0]-Pdict_b1[1])
	
	b_best=int(gap[1]>gap[0])
	
	
	return
	
def MLD_ini0(QT, PT, D, Mrounds, par):
	"""
	MLD decoder using an initial 0 state. If outcome is logical 1, predict flip.
	"""
	b0=par.g0[D]
	
	for i in range(len(QT)):
		b0=G_deterministic(b0,D,PT[i],QT[i], par)
	
	XML_b0, Pdict_b0=Decode(b0,par.QX,'z', par)
	
	return int(XML_b0)

def MLD_ini1(QT, PT, D, Mrounds, par):
	"""
	MLD decoder using an initial 1 state. If outcome is logical 0, predict flip.
	"""
	b1=par.g1[D]
	
	for i in range(len(QT)):
		b1=G_deterministic(b1,D,PT[i],QT[i], par)
	
	XML_b1, Pdict_b1=Decode(b0,par.QX,'z', par)
	
	return int(not XML_b1)

def LogBin(q):
    return 0.5*(1+signal.square(np.sqrt(np.pi)*(q-0.5*np.sqrt(np.pi)),duty=0.5))

def Decision_0():
    return 0



def energy(phi_1, phi_0, q_1, sigma, sigma_meas, alpha):
    """energy(phi_1, phi_0, q_1, sigma, sigma_meas, alpha):
    """
    return (phi_1 - phi_0)**2/(2*sigma**2) - cos(q_1 - 2*alpha*phi_1)/(4*alpha**2*sigma_meas**2)



def minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
    """minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
    """
    meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)

    n_steps = len(meas_outcomes) - 1

    phi = 0.0
    bestphi = 0.0
    for j in range(n_steps):
        old_phi = bestphi
        res_opt = minimize_scalar(energy, args=(old_phi, meas_outcomes[j], sigma, sigma_meas, alpha), bracket=(old_phi-3*pi/alpha, old_phi+3*pi/alpha))
        phi = res_opt.x
        bestphi = phi

    k = np.floor((alpha*bestphi + pi/2)/pi)
    assert meas_outcomes[n_steps] + 2*(k-1)*pi < 2*alpha*bestphi and meas_outcomes[n_steps] + 2*(k+1)*pi > 2*alpha*bestphi

    if meas_outcomes[n_steps] + 2*k*pi < 2*alpha*bestphi:
        if 2*alpha*bestphi - meas_outcomes[n_steps] - 2*k*pi < meas_outcomes[n_steps] + 2*(k+1)*pi - 2*alpha*bestphi:
            final_phi = meas_outcomes[n_steps]/(2*alpha) + k*pi/alpha
        else:
            final_phi = meas_outcomes[n_steps]/(2*alpha) + (k+1)*pi/alpha
    else:
        if 2*alpha*bestphi - meas_outcomes[n_steps] - 2*(k-1)*pi < meas_outcomes[n_steps] + 2*k*pi - 2*alpha*bestphi:
            final_phi = meas_outcomes[n_steps]/(2*alpha) + (k-1)*pi/alpha
        else:
            final_phi = meas_outcomes[n_steps]/(2*alpha) + k*pi/alpha
    return LogBin(final_phi)
    
####

def energy_2(q_1, q_0, q_meas_1, D, alpha):
    """energy(phi_1, phi_0, q_1, D, K, alpha):
    """
    return (q_1 - q_0)**2*D**2/2+(q_meas_1 - q_1)**2*D**2/2 - cos(2*alpha*(q_meas_1 - q_1))/(4*alpha**2*D**2)-cos(alpha*(q_1-q_0))/(alpha**2*D**2)

def energy_disp(q_1, q_0, q_meas_1, D, alpha):
    """energy(phi_1, phi_0, q_1, D, K, alpha):
    """
    return (q_1 - q_0+q_meas_1)**2*D**2/2+(q_1)**2*D**2/2 - cos(2*alpha*q_1)/(4*alpha**2*D**2)-cos(alpha*(q_1-q_0+q_meas_1))/(alpha**2*D**2)


def minimum_energy_decoding_gkp_2(meas_outcomes, D, alpha):
	"""minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
	"""
	#meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)
	n_steps = len(meas_outcomes)

	phi = 0.0
	bestphi = 0.0
	for j in range(n_steps):
		old_phi = bestphi
		res_opt = minimize_scalar(energy_2, args=(old_phi, meas_outcomes[j], D, alpha))
		phi = res_opt.x
		bestphi = phi

		
	return LogBin(bestphi)
	
def minimum_energy_decoding_gkp_disp(meas_outcomes, D, alpha):
	"""minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
	"""
	#meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)
	n_steps = len(meas_outcomes)

	phi = 0.0
	bestphi = 0.0
	for j in range(n_steps):
		old_phi = bestphi
		
		res_opt = minimize_scalar(energy_disp, args=(old_phi, meas_outcomes[j], D, alpha))
			
		phi = res_opt.x
		bestphi = phi

		
	return LogBin(bestphi)

def minimum_energy_decoding_gkp_sample0(meas_outcomes, D, alpha, par):
	"""minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
	"""
	#meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)
	n_steps = len(meas_outcomes)
	
	rho_in=np.abs(par.g0q[D])**2
	
	Num_samples=par.Forward_samples
	sample_decision=[]
	
	for n in range(Num_samples):
		phi_ini = random.choice(par.QX,p=rho_in)
		bestphi = phi_ini
		for j in range(n_steps):
			old_phi = bestphi
			res_opt = minimize_scalar(energy_2, args=(old_phi, meas_outcomes[j], D, alpha))
			phi = res_opt.x
			bestphi = phi
			
		sample_decision+=[LogBin(bestphi)]
	
	sample_decision=np.asarray(sample_decision)
	X=int(np.mean(sample_decision)>0.5)
	
	return X
	
def minimum_energy_decoding_gkp_sample0_disp(meas_outcomes, D, alpha, par):
	"""minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
	"""
	#meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)
	n_steps = len(meas_outcomes)
	
	rho_in=np.abs(par.g0q[D])**2
	
	Num_samples=par.Forward_samples
	sample_decision=[]
	
	for n in range(Num_samples):
		phi_ini = random.choice(par.QX,p=rho_in)
		bestphi = phi_ini
		for j in range(n_steps):
			old_phi = bestphi
			res_opt = minimize_scalar(energy_disp, args=(old_phi, meas_outcomes[j], D, alpha))
				
			phi = res_opt.x
			bestphi = phi
			
		sample_decision+=[LogBin(bestphi)]
	
	sample_decision=np.asarray(sample_decision)
	X=int(np.mean(sample_decision)>0.5)
	
	return X
	
def minimum_energy_decoding_gkp_d(meas_outcomes, D, alpha):
	"""minimum_energy_decoding_gkp(meas_outcomes, sigma, sigma_meas, alpha):
	"""
	#meas_outcomes=cmod(2*np.sqrt(np.pi)*meas_outcomes,2*np.pi)
	Diff=[]
	n_steps = len(meas_outcomes)

	phi = 0.0
	bestphi = 0.0
	for j in range(n_steps):
		old_phi = bestphi
		res_opt = minimize_scalar(energy_2, args=(old_phi, meas_outcomes[j], D, alpha))
		phi = res_opt.x
		bestphi = phi
		
		#test energy difference between optimized and no-shift energies
		diff_to_zero=energy_2(bestphi,old_phi, meas_outcomes[j], D, alpha)-energy_2(old_phi,old_phi, meas_outcomes[j], D, alpha)
		Diff+=[diff_to_zero]
		
	return LogBin(bestphi), Diff

def MCRound_MUDec(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
	
	XML, Pdict=Decode(State,par.QX,'z', par)


	X_0=Decision_0()
	X_Forward=minimum_energy_decoding_gkp_2(QT, D, np.sqrt(np.pi))
	X_Forward_sample=minimum_energy_decoding_gkp_sample0(QT, D, np.sqrt(np.pi), par)
	
	Fail_0=1-Pdict[X_0]
	Fail_Forward=1-Pdict[X_Forward]
	Fail_Forward_sample=1-Pdict[X_Forward_sample]
	Fail_MLD=1-Pdict[XML]

	return tuple([Fail_0, Fail_Forward, Fail_Forward_sample,Fail_MLD])
	


def ExecMC_MUDec(par): 
	"""
	receive list of tuples (D,Nrounds) and return PFail, Sigma_PFail for Decoders
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,4)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*lenDN,3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Failures=pool.map(MCRound_MUDec,flat_stack)
	else:
		Failures=[MCRound_MUDec(dn) for dn in flat_stack]
	
	Failures=np.array(list(Failures))
	Failures=Failures.reshape(stack_shape_out)

	PFail=np.mean(Failures,axis=0)
	Psig=np.std(Failures,axis=0)

	return [list(PFail),list(Psig)]
	
def MCRound_MUDec_succ(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	
	Fails=np.zeros((NRounds,4))
	
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
		
		XML, Pdict=Decode(State,par.QX,'z', par)
		
		QT_hist=QT[:i+1]
		PT_hist=PT[:i+1]
		

		X_0=Decision_0()
		X_Forward=minimum_energy_decoding_gkp_2(QT_hist, D, np.sqrt(np.pi))
		X_Forward_sample=minimum_energy_decoding_gkp_sample0(QT_hist, D, np.sqrt(np.pi), par)
		
		Fail_0=1-Pdict[X_0]
		Fail_Forward=1-Pdict[X_Forward]
		Fail_Forward_sample=1-Pdict[X_Forward_sample]
		Fail_MLD=1-Pdict[XML]
		
		Fails[i]=np.asarray([Fail_0, Fail_Forward, Fail_Forward_sample,Fail_MLD])

	return tuple(Fails.flatten())
	
def MCRound_MUDec_disp(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	
	Fails=np.zeros((NRounds,4))
	
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
		
		alpha_disp=(-QT[i]+1j*PT[i])/np.sqrt(2)
		
		State=qutip.displace(par.NFock,alpha_disp)*State
		State=State.unit()
		
		XML, Pdict=Decode(State,par.QX,'z', par)
		
		QT_hist=QT[:i+1]
		PT_hist=PT[:i+1]
		

		X_sum=LogBin(np.sum(QT))
		X_Forward_disp=minimum_energy_decoding_gkp_disp(QT_hist, D, np.sqrt(np.pi))
		X_Forward_sample_disp=minimum_energy_decoding_gkp_sample0_disp(QT_hist, D, np.sqrt(np.pi), par)
		
		Fail_sum=1-Pdict[X_sum]
		Fail_Forward_disp=1-Pdict[X_Forward_disp]
		Fail_Forward_sample_disp=1-Pdict[X_Forward_sample_disp]
		Fail_MLD=1-Pdict[XML]
		
		Fails[i]=np.asarray([Fail_sum, Fail_Forward_disp, Fail_Forward_sample_disp,Fail_MLD])

	return tuple(Fails.flatten())
	
def MCRound_MUDec_memoryless(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	
	Fails=np.zeros((NRounds,2))
	
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
		
		epsilon_q=cmod(QT[i],np.sqrt(np.pi))
		epsilon_p=cmod(PT[i],np.sqrt(np.pi))
		
		Sq=np.round((QT[i]-epsilon_q)/(2*np.sqrt(np.pi)))
		Sp=np.round((PT[i]-epsilon_p)/(2*np.sqrt(np.pi)))
		
		disp_q=Sq*2*np.sqrt(np.pi)+epsilon_q
		disp_p=Sp*2*np.sqrt(np.pi)+epsilon_p
		
		alpha_disp=(-disp_q+1j*disp_p)/np.sqrt(2)
		
		State=qutip.displace(par.NFock,alpha_disp)*State
		State=State.unit()
		
		XML, Pdict=Decode(State,par.QX,'z', par)
		
		# QT_hist=QT[:i+1]
		# PT_hist=PT[:i+1]
		

		X_0=Decision_0()
		# X_Forward=minimum_energy_decoding_gkp_2(QT_hist, D, np.sqrt(np.pi))
		# X_Forward_sample=minimum_energy_decoding_gkp_sample0(QT_hist, D, np.sqrt(np.pi), par)
		
		Fail_0=1-Pdict[X_0]
		# Fail_Forward=1-Pdict[X_Forward]
		# Fail_Forward_sample=1-Pdict[X_Forward_sample]
		Fail_MLD=1-Pdict[XML]
		
		Fails[i]=np.asarray([Fail_0, Fail_MLD])

	return tuple(Fails.flatten())
	
	

def ExecMC_MUDec_succ(par): 
	"""
	receive list of tuples (D,Nrounds) and return PFail, Sigma_PFail for Decoders
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,D_N_Array[0][1],4)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*lenDN,3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Failures=pool.map(MCRound_MUDec_succ,flat_stack)
	else:
		Failures=[MCRound_MUDec(dn) for dn in flat_stack]
	
	Failures=np.array(list(Failures))
	Failures=Failures.reshape(stack_shape_out)

	PFail=np.mean(Failures,axis=0)
	Psig=np.std(Failures,axis=0)

	return [list(PFail),list(Psig)]

def ExecMC_MUDec_disp(par): 
	"""
	receive list of tuples (D,Nrounds) and return PFail, Sigma_PFail for Decoders
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,D_N_Array[0][1],4)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*lenDN,3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Failures=pool.map(MCRound_MUDec_disp,flat_stack)
	else:
		Failures=[MCRound_MUDec_disp(dn) for dn in flat_stack]
	
	Failures=np.array(list(Failures))
	Failures=Failures.reshape(stack_shape_out)

	PFail=np.mean(Failures,axis=0)
	Psig=np.std(Failures,axis=0)

	return [list(PFail),list(Psig)]

def ExecMC_MUDec_memoryless(par): 
	"""
	receive list of tuples (D,Nrounds) and return PFail, Sigma_PFail for Decoders
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,D_N_Array[0][1],2)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*lenDN,3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Failures=pool.map(MCRound_MUDec_memoryless,flat_stack)
	else:
		Failures=[MCRound_MUDec_memoryless(dn) for dn in flat_stack]
	
	Failures=np.array(list(Failures))
	Failures=Failures.reshape(stack_shape_out)

	PFail=np.mean(Failures,axis=0)
	Psig=np.std(Failures,axis=0)

	return [list(PFail),list(Psig)]

def MCRound_MUDec_1in(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g1[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
	
	XML, Pdict=Decode(State,par.QX,'z', par)
			
	X_0=0
	X_Forward=minimum_energy_decoding_gkp(QT, D/np.sqrt(2), D/np.sqrt(2), np.sqrt(np.pi))
	X_Forward_2=minimum_energy_decoding_gkp_2(QT, D, np.sqrt(np.pi))

	Fail_0=Pdict[X_0]
	Fail_Forward=Pdict[X_Forward]
	Fail_Forward_2=Pdict[X_Forward_2]
	Fail_MLD=1-Pdict[XML]

	return tuple([Fail_0, Fail_Forward, Fail_Forward_2,Fail_MLD])

def MCRound_MUDec_x0(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g1[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	for i in range(NRounds):
		State,PT[i],QT[i]=G_deterministic(State,D, par)
	
	XML, Pdict=Decode(State,par.QX,'z', par)
	
	XML1=MLD_ini1(QT, PT, D, Mrounds, par)

	Fail_MLD=1-Pdict[XML]
	Fail_MLD1=1-Pdict[XML1]

	return tuple([Fail_MLD, Fail_MLD1])
	
def MCRound_MUDec_x1(DN_par):
	"""
	For Delta, M get error probabilities
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)

	random.seed()
	State=par.g1[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	for i in range(NRounds):
		State,PT[i],QT[i]=G_deterministic(State,D, par)

	XML, Pdict=Decode(State,par.QX,'z', par)
	XML0=MLD_ini0(QT, PT, D, Mrounds, par)		

	Fail_MLD=Pdict[XML]
	Fail_MLD0=Pdict[XML0]

	return tuple([Fail_MLD, Fail_MLD0])


def ExecMC_MUDec_1in(par): 
	"""
	receive list of tuples (D,Nrounds) and return PFail, Sigma_PFail for Decoders
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,4)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*lenDN,3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Failures=pool.map(MCRound_MUDec_1in,flat_stack)
	else:
		Failures=[MCRound_MUDec(dn) for dn in flat_stack]
	
	Failures=np.array(list(Failures))
	Failures=Failures.reshape(stack_shape_out)

	PFail=np.mean(Failures,axis=0)
	Psig=np.std(Failures,axis=0)

	return [list(PFail),list(Psig)]	
	
def cmod(x,a):
	"""
	centered "x modulo a" function.
	"""
	return np.mod(x+a/2,a)-a/2
	
def holevophase(state, quadrature, par):
	"""
	returns estimated peakwidth Delta and circular stabilizer mean phi
	as in eq. (1), Terhal B. M., Weigand D. J.,	arXiv:1909.10075
	"""
	if quadrature=='q':
		exp_q=qutip.expect(par.stab_q, state)
		
		Delta, phi= np.sqrt(np.log(1/np.abs(exp_q))/np.pi), np.angle(exp_q)/(2*np.sqrt(np.pi))

	elif quadrature=='p':
		exp_p=qutip.expect(par.stab_p, state)
		Delta, phi=np.sqrt(np.log(1/np.abs(exp_p))/np.pi), np.angle(exp_p)/(2*np.sqrt(np.pi))
	
	return Delta, phi
	
def Track_Photons(DN_par):
	"""
	For Delta, M get photonnumbers
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	Num=np.zeros(2*NRounds+1)
	Deltaq=np.zeros(2*NRounds+1)
	Deltap=np.zeros(2*NRounds+1)
	
	numberop=qutip.num(par.NFock)
	Num[0]=qutip.expect(numberop, State)
	Deltaq[0],_=holevophase(State, 'q', par)
	Deltap[0],_=holevophase(State, 'p', par)
	
	
	for i in range(2*NRounds):
		if i%2==0:
			State,PT[int(i/2)]=GP(State,D, par)
			
		
		elif i%2==1:
			State,QT[int((i-1)/2)]=GQ(State,D, par)
			
		Num[i+1]=qutip.expect(numberop, State)
		Deltaq[i+1],_=holevophase(State, 'q', par)
		Deltap[i+1],_=holevophase(State, 'p', par)

	return tuple(Num.tolist()+Deltaq.tolist()+Deltap.tolist())

def Exec_Photons(par): 
	"""
	receive list of tuples (D,Nrounds) and return <n>(M), Sigma_<n>(M)
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	Blocksize=2*D_N_Array[0][1]+1
	
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,3*Blocksize)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*len(D_N_Array),3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Data=pool.map(Track_Photons,flat_stack)
	else:
		Data=[Track_Photons(dn) for dn in flat_stack]
	
	Data=np.array(list(Data))
	Data=Data.reshape(stack_shape_out)
	
	AVG_Data=np.mean(Data,axis=0)
	SIG_Data=np.std(Data,axis=0)
	
	AVG_Photon_nums, AVG_Deltaqs, AVG_Deltaps=AVG_Data[:,:Blocksize], AVG_Data[:,Blocksize:2*Blocksize], AVG_Data[:,2*Blocksize:3*Blocksize]
	SIG_Photon_nums, SIG_Deltaqs, SIG_Deltaps=SIG_Data[:,:Blocksize], SIG_Data[:,Blocksize:2*Blocksize], SIG_Data[:,2*Blocksize:3*Blocksize]


	return [list(AVG_Photon_nums),list(SIG_Photon_nums),list(AVG_Deltaqs),list(SIG_Deltaqs),list(AVG_Deltaps),list(SIG_Deltaps)]
	
	
def Track_Photons_Disp(DN_par):
	"""
	For Delta, M get photonnumbers
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	
	Num=np.zeros(NRounds+1)
	Deltaq=np.zeros(NRounds+1)
	Deltap=np.zeros(NRounds+1)
	Deltaq[0],_=holevophase(State, 'q', par)
	Deltap[0],_=holevophase(State, 'p', par)
	
	numberop=qutip.num(par.NFock)
	Num[0]=qutip.expect(numberop, State)
	
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
		
		alpha_disp=(-QT[i]+1j*PT[i])/np.sqrt(2)
		
		State=qutip.displace(par.NFock,alpha_disp)*State
		State=State.unit()
		
		Num[i+1]=qutip.expect(numberop, State)
		Deltaq[i+1],_=holevophase(State, 'q', par)
		Deltap[i+1],_=holevophase(State, 'p', par)
	
	return tuple(Num.tolist()+Deltaq.tolist()+Deltap.tolist())
	

def Exec_Photons_Disp(par): 
	"""
	receive list of tuples (D,Nrounds) and return <n>(M), Sigma_<n>(M) with correctional displacement.
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	Blocksize=D_N_Array[0][1]+1
	
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,3*Blocksize)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*len(D_N_Array),3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Data=pool.map(Track_Photons_Disp,flat_stack)
	else:
		Data=[Track_Photons_Disp(dn) for dn in flat_stack]
	
	Data=np.array(list(Data))
	Data=Data.reshape(stack_shape_out)
	
	AVG_Data=np.mean(Data,axis=0)
	SIG_Data=np.std(Data,axis=0)
	
	AVG_Photon_nums, AVG_Deltaqs, AVG_Deltaps=AVG_Data[:,:Blocksize], AVG_Data[:,Blocksize:2*Blocksize], AVG_Data[:,2*Blocksize:3*Blocksize]
	SIG_Photon_nums, SIG_Deltaqs, SIG_Deltaps=SIG_Data[:,:Blocksize], SIG_Data[:,Blocksize:2*Blocksize], SIG_Data[:,2*Blocksize:3*Blocksize]


	return [list(AVG_Photon_nums),list(SIG_Photon_nums),list(AVG_Deltaqs),list(SIG_Deltaqs),list(AVG_Deltaps),list(SIG_Deltaps)]
	
	
def Track_Photons_memoryless(DN_par):
	"""
	For Delta, M get photonnumbers
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	
	Num=np.zeros(NRounds+1)
	Deltaq=np.zeros(NRounds+1)
	Deltap=np.zeros(NRounds+1)
	Deltaq[0],_=holevophase(State, 'q', par)
	Deltap[0],_=holevophase(State, 'p', par)
	
	numberop=qutip.num(par.NFock)
	Num[0]=qutip.expect(numberop, State)
	
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
		
		epsilon_q=cmod(QT[i],np.sqrt(np.pi))
		epsilon_p=cmod(PT[i],np.sqrt(np.pi))
		
		Sq=np.round((QT[i]-epsilon_q)/(2*np.sqrt(np.pi)))
		Sp=np.round((PT[i]-epsilon_p)/(2*np.sqrt(np.pi)))
		
		disp_q=Sq*2*np.sqrt(np.pi)+epsilon_q
		disp_p=Sp*2*np.sqrt(np.pi)+epsilon_p
		
		alpha_disp=(-disp_q+1j*disp_p)/np.sqrt(2)
		
		State=qutip.displace(par.NFock,alpha_disp)*State
		State=State.unit()
		
		Num[i+1]=qutip.expect(numberop, State)
		Deltaq[i+1],_=holevophase(State, 'q', par)
		Deltap[i+1],_=holevophase(State, 'p', par)
	
	return tuple(Num.tolist()+Deltaq.tolist()+Deltap.tolist())
	

def Exec_Photons_memoryless(par): 
	"""
	receive list of tuples (D,Nrounds) and return <n>(M), Sigma_<n>(M) with correctional displacement.
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	Blocksize=D_N_Array[0][1]+1
	
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,3*Blocksize)
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*len(D_N_Array),3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Data=pool.map(Track_Photons_memoryless,flat_stack)
	else:
		Data=[Track_Photons_memoryless(dn) for dn in flat_stack]
	
	Data=np.array(list(Data))
	Data=Data.reshape(stack_shape_out)
	
	AVG_Data=np.mean(Data,axis=0)
	SIG_Data=np.std(Data,axis=0)
	
	AVG_Photon_nums, AVG_Deltaqs, AVG_Deltaps=AVG_Data[:,:Blocksize], AVG_Data[:,Blocksize:2*Blocksize], AVG_Data[:,2*Blocksize:3*Blocksize]
	SIG_Photon_nums, SIG_Deltaqs, SIG_Deltaps=SIG_Data[:,:Blocksize], SIG_Data[:,Blocksize:2*Blocksize], SIG_Data[:,2*Blocksize:3*Blocksize]


	return [list(AVG_Photon_nums),list(SIG_Photon_nums),list(AVG_Deltaqs),list(SIG_Deltaqs),list(AVG_Deltaps),list(SIG_Deltaps)]	
	
def MCRound_Diff(DN_par):
	"""
	For Delta, M get energy differences to zero
	"""
	D,NRounds, par=DN_par
	NRounds=int(NRounds)
	
	random.seed()
	State=par.g0[D]
	QT=np.zeros(NRounds)
	PT=np.zeros(NRounds)
	for i in range(NRounds):
		State,PT[i],QT[i]=G(State,D, par)
	
		
	X_Forward_2, Diff=minimum_energy_decoding_gkp_d(QT, D, np.sqrt(np.pi))

	return tuple(Diff)
	
def Exec_Diff(par): 
	"""
	receive list of tuples (D,Nrounds) and return <Differences> between no-shift- and optimized energy
	"""
	
	D_N_Array=par.Domain_t_MDomain
	lenDN=len(D_N_Array)
	
	stack_shape_in=(par.NP,lenDN,3)
	stack_shape_out=(par.NP,lenDN,D_N_Array[0][1])
	
	D_N_par=[tuple((d[0],d[1], par)) for d in D_N_Array]
	
	Stacked_Domain=np.tile(np.array(D_N_par,dtype=tuple),(par.NP,1)).reshape(stack_shape_in) #### (NP x Parameters)

	flat_stack=Stacked_Domain.reshape((par.NP*len(D_N_Array),3)) ###flattened
	
	if par.parallel:
		pool=Pool(par.NCores)
		Diffs=pool.map(MCRound_Diff,flat_stack)
	else:
		Diffs=[MCRound_Diff(dn) for dn in flat_stack]
	
	Diffs=np.array(list(Diffs))
	print(Diffs)
	Diffs=Diffs.reshape(stack_shape_out)

	AVG_Diffs=np.mean(Diffs,axis=0)
	SIG_Diffs=np.std(Diffs,axis=0)

	return [list(AVG_Diffs),list(SIG_Diffs)]