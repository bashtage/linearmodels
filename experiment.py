from linearmodels.panel.lsmr.lsmr import experiment, LSMR
import numpy as np
from scipy.sparse.csc import csc_matrix

y = np.random.randn(1000)
x = np.random.randn(1000, 5)
LSMR(y, x, precondition=False)

experiment()
experiment()

x = [[1.0, 0.0, -1.0],
     [0.0, 2.0, 0.0],
     [2.0, 0.0, 2.0],
     [5.0, 3.0, -2.0],
     [0.0, 0.0, 6.0]]
x = np.array(x)
xc = csc_matrix(x)
y = np.ones(5)
l = LSMR(y, x, precondition=False)
print(l.beta)

l = LSMR(y, xc, precondition=False)
print(l.beta)
sqrt_norm2 = np.sqrt((xc.multiply(xc)).sum(0)).A.squeeze()
xn = xc.copy()
for i in range(xn.shape[1]):
     xn[:,i] /= sqrt_norm2[i]
l = LSMR(y, xn, precondition=False)
print(l.beta)
print(l.beta / sqrt_norm2)



from scipy.sparse.linalg import lsmr as splsmr

print(splsmr(x, y)[0])