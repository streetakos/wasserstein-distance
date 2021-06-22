from scipy.optimize import linprog
from time import time
import numpy as np
from scipy.linalg import sqrtm


'''
As ground distances between 2 kernels we use Frechet measure
'''
def frechet(m1, s1, m2, s2):
    '''
    Computes Frechet distance
    '''
    fid = np.sqrt( (m1-m2) @ (m1-m2) + np.trace( s1 + s2 - 2*sqrtm(s1@s2)) )
    return fid


def emdDistance(Pm, Ps, PPj, Qm, Qs, QPj): 
    '''
    Computes the earth mover distance between 2 gaussian mixture models.
    Pm: first model's means
    Ps: first model's covariances
    PPj:first model's weights
    Qm, Qs, QPj for the second model respectivelly
    '''
    
    
    Wp = PPj
    Wq = QPj
    
    start = time()
    m  = len(PPj) 
    n  = len(QPj)  
    dist   = np.zeros((m,n))

    # Calculate Frechet distance
    for i in range(m):
        for j in range(n):
            dist[i,j] = frechet(Pm[i],Ps[i], Qm[j],Qs[j])  
            
    sum_Wp = np.sum(PPj)   
    sum_Wq = np.sum(QPj) 
    movableEarth = np.min([sum_Wp, sum_Wq])
    dist = dist.flatten()

    ########### Create Constrains ###########
    
    A_ub = np.zeros((m+n,m*n))
    b_ub = np.zeros(m+ n)
    for i in range(m):
        for j in range(n):
            A_ub[i, i*n +j] = 1
            A_ub[j+m, (j) + i*n] = 1
    for i in range(m):
        b_ub[i] = Wp[i]
    for i in range(m,m+ n):
        b_ub[i] = Wq[i-m]
    A_eq = np.ones((1,m*n))
    b_eq = np.array([movableEarth])
    bounds = [(0,None) for i in range(m*n)]
    
    
    ########### Optimize ###########
    res = linprog(dist,  A_ub=A_ub, b_ub=b_ub, bounds=bounds, A_eq=A_eq, b_eq=b_eq)

    # Compute EMD
    optimal_flow = res.x
    emd = np.sum(dist*optimal_flow)/np.sum(optimal_flow)
    
    return emd
