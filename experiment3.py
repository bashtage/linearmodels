#import timeit
import numpy as np
from linearmodels.panel.lsmr.lsmr import LSMR
#from scipy.sparse.linalg import lsmr
from linearmodels.panel.utility import dummy_matrix
from numpy import sqrt, finfo
import pstats, cProfile


rs = np.random.RandomState(1234)
m = 2000000
c1 = rs.randint(0,m//3,m)
c2 = rs.randint(0,m//20,m)
y = c1 / 10000 +c2 / 1000 + rs.randn(m)
eps = finfo(np.double).eps
d = dummy_matrix(np.column_stack([c1,c2]))
#b = lsmr(d,y,atol=sqrt(eps),btol=sqrt(eps),conlim=1/(10*sqrt(eps)), show=True)[0]
#timer_sp = timeit.Timer('lsmr(d,y,atol=sqrt(eps),btol=sqrt(eps),conlim=1/(10*sqrt(eps)), show=False)[0]',globals=globals())
#times_sp = timer_sp.repeat(5,1)

# b2 = LSMR(d,y,precondition=True, disp=True).beta
#print('start timer')
#timer = timeit.Timer('LSMR(d,y,precondition=True, disp=False).beta',globals=globals())
#times = timer.repeat(5,1)
#print(min(times))

cProfile.runctx("LSMR(d,y,precondition=True, disp=False).beta", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
