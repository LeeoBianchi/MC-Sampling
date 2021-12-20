import time as t
import numpy as np


#Gaussian with mu=0, sigma = 1
def phi(x):
    return 1/np.sqrt(2*np.pi) * np.exp(- x**2/(2))

#proposal distribution
def g(x):
    return np.where((x>-0.5)&(x<0.5), 1/3, np.exp(-np.abs(x) + 0.5)/3)

def h(x):
    return np.where(x>0, x**2, 0)

#Inverse of the cumulative of G
def G_inv(U):
    i = np.random.choice([1,2,3])
    if i == 1:
        x_p = U - 0.5
    elif i == 2:
        x_p = np.log(1 - U) - 0.5
    elif i == 3:
        x_p = - np.log(1 - U) + 0.5
    return x_p

def reject_sample (N_it, pri = True):
    t1 = t.time()
    alpha = np.sqrt(2 *np.pi)/3
    Xs = [] #target sample
    j = 0
    while j < N_it:
        acc = False
        while not acc:
            U = np.random.uniform(size = 2)
            x_p = G_inv(U[0]) #generate proposal
            
            #acceptance condition
            if U[1] < alpha * phi(x_p) / g(x_p):
                Xs.append(x_p)
                acc = True
                j += 1
    t2 = t.time()
    if pri:
        print('time taken = %.4f'%(t2-t1)+' s')
    return np.array(Xs)

def SIR(n, m, pri = True):
    t1 = t.time()
    Ys = []
    for i in range(m):
        #we generate the Ys ~ g(Y)
        Ys.append(G_inv(np.random.uniform()))
    Ys = np.array(Ys)
    w_star_s = phi(Ys)/g(Ys)
    D = np.sum(w_star_s) #normalization
    ws = w_star_s/D
    Xs = np.random.choice(Ys, size = n, p = ws) #resampling
    t2 = t.time()
    if pri:
        print('time taken = %.4f'%(t2-t1)+' s')
    return Xs