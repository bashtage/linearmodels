from linearmodels.panel.lsmr.lsmr import LSMR
import numpy as np
from scipy.sparse.csc import csc_matrix
import time
y = np.random.randn(10000)
x = np.random.randn(10000, 10)
LSMR(x, y, precondition=True)


x = [[1.0, 0.0, -1.0],
     [0.0, 2.0, 0.0],
     [2.0, 0.0, 2.0],
     [5.0, 3.0, -2.0],
     [0.0, 0.0, 6.0]]
x = np.array(x)
xc = csc_matrix(x)
y = np.ones(5)
itn = [100]
print(itn)
l = LSMR(x, y, precondition=True)
print(l.info)
itn.append(l.info['itn'])

l = LSMR(xc, y, precondition=True)
print(l.info)
itn.append(l.info['itn'])
sqrt_norm2 = np.sqrt((xc.multiply(xc)).sum(0)).A.squeeze()
xn = xc.copy()
for i in range(xn.shape[1]):
    xn[:,i] /= sqrt_norm2[i]
l = LSMR(xn, y, precondition=True)
print(l.info)
itn.append(l.info['itn'])
print(min(itn))

if min(itn) < 3:
    raise NotImplementedError
#from scipy.sparse.linalg import lsmr as splsmr

#print(splsmr(x, y)[0])

a = np.random.randn(10,5)
b = np.random.randn(5,1)
c = np.random.randn(10,1)
a = np.asarray(a,order='F')
b = np.asarray(b,order='F')
c = np.asarray(c,order='F')

from scipy.linalg.blas import dgemm
c2 = c +  a@b
c3 = c.copy()
dgemm(1.,a,b,1,c3,0,0,1)
#print(c2-c3)
c4 = c.copy()

#print(c4)

"""
    c = dgemm(alpha,a,b,[beta,c,trans_a,trans_b,overwrite_c])
    
    Wrapper for ``dgemm``.
    
    Parameters
    ----------
    alpha : input float
    a : input rank-2 array('d') with bounds (lda,ka)
    b : input rank-2 array('d') with bounds (ldb,kb)
    
    Other Parameters
    ----------------
    beta : input float, optional
        Default: 0.0
    c : input rank-2 array('d') with bounds (m,n)
    overwrite_c : input int, optional
        Default: 0
    trans_a : input int, optional
        Default: 0
    trans_b : input int, optional
        Default: 0
    
"""