# import timeit
import numpy as np
from linearmodels.panel.lsmr.lsmr import LSMR
from scipy.sparse.linalg import lsmr
from scipy.sparse import csr_matrix
from linearmodels.panel.utility import dummy_matrix
from numpy import sqrt, finfo
import pstats, cProfile
from timer_cm import Timer

rs = np.random.RandomState(1234)
m = 2000000
c1 = rs.randint(0, m // 3, m)
c2 = rs.randint(0, m // 20, m)
y = c1 / 10000 + c2 / 1000 + rs.randn(m)
eps = finfo(np.double).eps
d = dummy_matrix(np.column_stack([c1, c2]))  # type: scipy.sparse.csc.csc_matrix

b = lsmr(d, y, atol=sqrt(eps), btol=sqrt(eps), conlim=1 / (10 * sqrt(eps)), show=True)[0]
#with Timer('scipy'):
#    b = lsmr(d, y, atol=sqrt(eps), btol=sqrt(eps), conlim=1 / (10 * sqrt(eps)), show=True)[0]

#with Timer('conversion'):
#    dr = csr_matrix(d)

#with Timer('scipy-csr'):
#    b = lsmr(d, y, atol=sqrt(eps), btol=sqrt(eps), conlim=1 / (10 * sqrt(eps)), show=True)[0]



# sizes = """
# int n, m, n_ptr, n_row, n_data, n_rhs;
# n = {n};
# m = {m};
# n_ptr = {n_ptr};
# n_row = {n_row};
# n_data = {n_data};
# n_rhs = {n_rhs};
# """
# for f, v, fmt in (('data', d.data, '{:0.8f}'),
#                   ('ptr', d.indptr, '{:d}'),
#                   ('row', d.indices, '{:d}'),
#                   ('rhs', y, '{:0.8f}')):
#     with open(r'c:\temp\lsmr-{0}.txt'.format(f), 'w') as df:
#         df.write(','.join(map(lambda x: fmt.format(x), v)))
#
# with open(r'c:\temp\lsmr_data_size.h', 'w') as h:
#     h.write(sizes.format(n=d.shape[1], m=d.shape[0],
#                          n_row=d.indices.shape[0],
#                          n_data=d.data.shape[0],
#                          n_rhs=m,
#                          n_ptr=d.indptr.shape[0]))
#
# out = """
# const int m = {m}, n = {n};
# int ptr[]    = {{{ptr}}};
# int row[]    = {{{row}}};
# double val[] = {{{data}}};
# /* RHS */
# double b[] = {{{b}}};
# """
# with open(r'c:\temp\lsmr_data.h', 'w') as h:
#     h.write(out.format(data=', '.join(map(str, d.data)),
#                        row=', '.join(map(str, d.indices)),
#                        ptr=', '.join(map(str, d.indptr)),
#                        b=', '.join(map(lambda v: '{:0.8f}'.format(v), y)),
#                        m=d.shape[0], n=d.shape[1]))

# timer_sp = timeit.Timer('lsmr(d,y,atol=sqrt(eps),btol=sqrt(eps),conlim=1/(10*sqrt(eps)), show=False)[0]',globals=globals())
# times_sp = timer_sp.repeat(5,1)

# b2 = LSMR(d,y,precondition=True, disp=True).beta
# print('start timer')
# timer = timeit.Timer('LSMR(d,y,precondition=True, disp=False).beta',globals=globals())
# times = timer.repeat(5,1)
# print(min(times))

# cProfile.runctx("LSMR(d,y,precondition=True, disp=False).beta", globals(), locals(), "Profile.prof")

# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()
