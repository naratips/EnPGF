# EuroMail_EnPGF.py
#
# Ensemble Poisson Gamma filter:
#
# Code used to produce the European Email data results presented in the paper
# 
# Identification of an influence network by an ensemble-based filtering
# for Hawkes process driven by count data (2022)
#
# by N. Santitissadeekorn, S. Delahaies (s.delahaies@gmail.com) and D. J. B. Lloyd 
#
# Department of Mathematics, University of Surrey  
# 
 
from scipy.io import loadmat
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numba import jit,njit,int32,float64,vectorize,guvectorize
import time
import multiprocessing as mp



@jit(float64(int32,float64[:]),nopython=True,target_backend='cpu')
def mean(Nsamp,lb):
    lbm=np.float64(0.)
    for i in range(Nsamp):
        lbm += lb[i]/Nsamp
    return lbm

@njit(nb.types.Tuple((nb.float64, nb.float64,nb.float64))(int32,int32,int32,float64[:],float64[:]))
def Analysis_numba(N,Nsamp,dN,yn,lb):
    lbm=np.float64(0.)
    Vf=np.float64(0.)
    for i in range(Nsamp):
        lbm += lb[i]/Nsamp
    for i in range(Nsamp):
        Vf += (lb[i]-lbm)*(lb[i]-lbm)/(Nsamp-1)
    lbm_a=lbm+(dN-lbm)*Vf/(lbm+Vf)
    return lbm_a,lbm,Vf

@vectorize([float64(float64,float64,float64,float64,float64,float64,float64)])
def Analysis_ens_2(dN,lbm_a,lbm,Vf,ynm,yn,lb):
    lb_a=lbm_a+lbm_a*(lb-lbm)/lbm
    if (dN>0.):
        Vfr=Vf/(lbm*lbm)
        lb_a += lbm_a*Vfr/(Vfr+1./dN)*((yn-ynm)/ynm-(lb-lbm)/lbm)
    return lb_a    

@vectorize([float64(float64,float64,float64)])
def Analysis_kk(lb_a,Vf,lb):
    kk=(lb_a-lb)/Vf
    return kk

@jit(float64(int32,float64[:],float64,float64[:],float64),nopython=True,target_backend='cpu')
def parVf_mod(Nsamp,lb,lbm,p,pm):
    Vf=np.float64(0.)
    for i in range(Nsamp):
        Vf +=(lb[i]-lbm) *(p[i]-pm)/(Nsamp-1)
    return Vf

@vectorize([float64(float64,float64,float64)])
def p_up(Vf,kk,p):
    p_a=p+Vf*kk
    return p_a

@jit(float64[:,:](int32,int32,float64[:],float64[:],float64,float64[:,:]),nopython=True,target_backend='cpu')
def p_up_vec_jit(N,Nsamp,kk,lb,lbm,a_in):
    #a_out=np.zeros((N),dtype=np.float64)
    for i in range(N):
        a_m=mean(Nsamp,a_in[i,:])
        a_Vf=parVf_mod(Nsamp,lb,lbm,a_in[i,:],a_m)
        a_in[i,:]=p_up(a_Vf,kk,a_in[i,:])
    return a_in

@jit(nopython=True)
def AdN(N,Nsamp,dN,a_in):
    AdN_out=np.zeros((Nsamp),dtype=np.float64)
    for j in range(Nsamp):
        AdN_tmp=np.float64(0.) 
        for i in range(N):
            AdN_tmp+=np.exp(a_in[i,j])*dN[i]    
        AdN_out[j]=AdN_tmp
    return AdN_out

@vectorize([float64(float64,float64,float64,float64,float64)])
def Forecast(dt,lbdt,m_log,b_log,AdN):
    lb_out=dt*(np.exp(m_log)+(lbdt/dt-np.exp(m_log))*(1.-dt*np.exp(b_log))+AdN)
    if (lb_out<=0.):
        lb_out=0.0000001
    return lb_out

######################################
def Kernel_EnPGF_numba(l,N,Nsamp,Nrun,dt,Nstep,dNall):
    print(l)
    tmp = np.random.gamma(1.5,1,[Nsamp])
    m_log = np.log(tmp.copy()).astype(np.float64)
    lbdt = dt*tmp.copy().astype(np.float64)
    b_log = np.log(np.random.normal(3,0.5,[Nsamp])).astype(np.float64)
    a_log=np.log(0.2*np.random.gamma(1,0.1,[N,Nsamp])).astype(np.float64)
    dN_rnd_all=np.random.gamma(1,1,[Nsamp,Nstep]).astype(np.float64)

    t0=time.time()

    for k_run in range(Nrun):
        for k in range(Nstep):
            # ANALYSIS
            lbm_a,lbm,Vf=Analysis_numba(N,Nsamp,dNall[k,l],dN_rnd_all[:,k],lbdt)
            ynm=mean(Nsamp,dN_rnd_all[:,k])
            lb_a=Analysis_ens_2(dNall[k,l],lbm_a,lbm,Vf,ynm,dN_rnd_all[:,k],lbdt)
            kk=Analysis_kk(lb_a,Vf,lbdt)
            
            # UPDATE
            m_m=mean(Nsamp,m_log)
            m_Vf=parVf_mod(Nsamp,lbdt,lbm,m_log,m_m)
            m_log=p_up(m_Vf,kk,m_log)

            b_m=mean(Nsamp,b_log)
            b_Vf=parVf_mod(Nsamp,lbdt,lbm,b_log,b_m)
            b_log=p_up(b_Vf,kk,b_log)
            
            a_log=p_up_vec_jit(N,Nsamp,kk,lbdt,lbm,a_log)

            # FORECAST
            AdN_out=AdN(N,Nsamp,dNall[k,:],a_log)
            lbdt=Forecast(dt,lb_a,m_log,b_log,AdN_out)
        t1=time.time()
        print(l,t1-t0)
        
    return l,t1-t0,lbdt,m_log,b_log,a_log
######################################

###
### Initialization 
###

np.random.seed(19006)
dat=loadmat('dN1_1dayunit_weekday.mat')
dNall=dat['dN1'].astype(np.int64)
N=dNall.shape[0]
Nstep=dNall.shape[1]

Nsamp=100
run=1
dt=0.1
Nrun=2

dNall=dNall.transpose()
Npool=mp.cpu_count()
print('number of cpu',Npool)
pool = mp.Pool(Npool)

t_start=time.time()

#
#  numba Kernel
#

results = pool.starmap(Kernel_EnPGF_numba, [(l,N,Nsamp,Nrun,dt,Nstep,dNall) for l in range(N)])       
pool.close()

t_end=time.time()
   
#
#  process and write outputs
#    

t_ellapsed=np.empty([N])
mu=np.empty([N])
beta=np.empty([N])
alpha=np.empty([N,N])
 
for l in range(N):
    t_ellapsed[results[l][0]]=results[l][1]
    mu[results[l][0]]=np.mean(results[l][2])
    beta[results[l][0]]=np.mean(results[l][3])
    alpha[results[l][0],:]=np.mean(results[l][-1],1)
print(t_ellapsed)

np.savetxt('EuroMail_mu_Nsamp{}_run{}_numba.csv'.format(Nsamp,Nrun),mu)
np.savetxt('EuroMail_beta_Nsamp{}_run{}_numba.csv'.format(Nsamp,Nrun),beta)
np.savetxt('EuroMail_alpha_Nsamp{}_run{}_numba.csv'.format(Nsamp,Nrun),alpha,delimiter=',')

print(t_end-t_start)

# 
# display excitation matrix
#
plt.matshow(np.exp(alpha))
plt.show()

