import timeit
import numpy as np
from linearmodels.panel.lsmr.lsmr import LSMR
from scipy.sparse.linalg import lsmr
rs = np.random.RandomState(1234)
m = 2000000
c1 = rs.randint(0,m//3,m)
c2 = rs.randint(0,m//20,m)
y = c1 / 10000 +c2 / 1000 + rs.randn(m)
from linearmodels.panel.utility import dummy_matrix
from numpy import sqrt, finfo
eps = finfo(np.double).eps
d = dummy_matrix(np.column_stack([c1,c2]))
#pb = lsmr(d,y,atol=sqrt(eps),btol=sqrt(eps),conlim=1/(10*sqrt(eps)), show=True)[0]
#timer_sp = timeit.Timer('lsmr(d,y,atol=sqrt(eps),btol=sqrt(eps),conlim=1/(10*sqrt(eps)), show=False)[0]',globals=globals())
#times_sp = timer_sp.repeat(5,1)

l2 = LSMR(d,y,precondition=True, disp=True, local_vecs=50)
b2 = l2.beta

l2 = LSMR(d,y,precondition=True, disp=False, local_vecs=0)
b3 = l2.beta
#print('start timer')
#timer = timeit.Timer('LSMR(d,y,precondition=True, disp=False).beta',globals=globals())
#times = timer.repeat(5,1)
#print(min(times))
