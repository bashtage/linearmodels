# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Copyright (c) 2010, 2012 David Fong, Michael Saunders, Stanford University
Copyright (c) 2014-2016 Science and Technology Facilites Council (STFC)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                               LSMR
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

LSMR   solves Ax = b or min ||Ax - b|| with or without damping,
using the iterative algorithm of David Fong and Michael Saunders:
    http://www.stanford.edu/group/SOL/software/lsmr.html

The original LSMR code can be found at
http://web.stanford.edu/group/SOL/software/lsmr/
It is maintained by
    David Fong       <clfong@stanford.edu>
    Michael Saunders <saunders@stanford.edu>
    Systems Optimization Laboratory (SOL)
    Stanford University
    Stanford University
    Stanford, CA 94305-4026, USA

This version uses reverse communication so that control is passed back
to the user to perform products with A or A' and to perform preconditioning.
In addition, the user may choose to use their own stopping rule(s)
or to apply the stopping criteria of Fong and Saunders.
A number of options are available (see the derived type lsmr_type_options).

   Dec 2015: SPRAL version developed to allow user more control
             over preconditioning. Also comprehensive test deck written.
23 Nov 2015: revised reverse communication interface written by
             Nick Gould and Jennifer Scott.
             lsmrDataModule.f90 no longer used.
             Automatic arrays no longer used.
             Option for user to test convergence (or to use Fong and
             Saunders test).
             Uses dnrm2 and dscal (for closer compatibility with
             Fong and Saunders code ... easier to compare results).
             Sent to Saunders; made available on above SOL webpage.
20 May 2014: initial reverse-communication version written by Nick Gould
02 May 2014: With damp>0, flag=2 was incorrectly set to flag=3
             (so incorrect stopping message was printed).  Fixed.
28 Jan 2014: In lsmrDataModule.f90:
             ip added for integer(ip) declarations.
             dnrm2 and dscal coded directly
             (no longer use lsmrblasInterface.f90 or lsmrblas.f90).
07 Sep 2010: Local reorthogonalization now works (localSize > 0).
17 Jul 2010: F90 LSMR derived from F90 LSQR and lsqr.m.
             Aprod1, Aprod2 implemented via f90 interface.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Precision
    ---------
    The number of iterations required by LSMR will decrease
    if the computation is performed in higher precision.
    At least 15-digit arithmetic should normally be used.
    "real(wp)" declarations should normally be 8-byte words.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Modifications Copyright (c) 2019 Kevin Sheppard

"""

LSMR_DOCSTRING = """
LSMR finds a solution x to the following problems:

1. Unsymmetric equations:    Solve  A*x = b

2. Linear least squares:     Solve  A*x = b
                             in the least-squares sense

3. Damped least squares:     Solve  (   A    )*x = ( b )
                                    ( damp*I )     ( 0 )
                             in the least-squares sense

where A is a matrix with m rows and n columns, b is an m-vector,
and damp is a scalar.  (All quantities are real.)
The matrix A is treated as a linear operator.  It is accessed
by reverse communication, that is requests for matrix-vector 
products are passed back to the user.

If preconditioning used, solves
1. Unsymmetric equations:    Solve  AP*y = b

2. Linear least squares:     Solve  AP*y = b
                             in the least-squares sense

3. Damped least squares:     Solve  (   A    )*P*y = ( b )
                                    ( damp*I )       ( 0 )
                             in the least-squares sense
Code computes y and user must recover x = Py

LSMR uses an iterative method to approximate the solution.
The number of iterations required to reach a certain accuracy
depends strongly on the scaling of the problem.  Poor scaling of
the rows or columns of A should therefore be avoided where
possible.

For example, in problem 1 the solution is unaltered by
row-scaling.  If a row of A is very small or large compared to
the other rows of A, the corresponding row of ( A  b ) should be
scaled up or down.

In problems 1 and 2, the solution x is easily recovered
following column-scaling.  Unless better information is known,
the nonzero columns of A should be scaled so that they all have
the same Euclidean norm (e.g., 1.0).

In problem 3, there is no freedom to re-scale if damp is
nonzero.  However, the value of damp should be assigned only
after attention has been paid to the scaling of A.

The parameter damp is intended to help regularize
ill-conditioned systems, by preventing the true solution from
being very large.  Another aid to regularization is provided by
the parameter condAP, which may be used to terminate iterations
before the computed solution becomes very large.

Note that x is not an input parameter.
If some initial estimate x0 is known and if damp = 0,
one could proceed as follows:

1. Compute a residual vector     r0 = b - A*x0.
2. Use LSMR to solve the system  A*dx = r0.
3. Add the correction dx to obtain a final solution x = x0 + dx.

This requires that x0 be available before the first call 
to LSMR and after the final call. 
To judge the benefits, suppose LSMR takes k1 iterations
to solve A*x = b and k2 iterations to solve A*dx = r0.
If x0 is "good", norm(r0) will be smaller than norm(b).
If the same stopping tolerances atol and btol are used for each
system, k1 and k2 will be similar, but the final solution x0 + dx
should be more accurate.  The only way to reduce the total work
is to use a larger stopping tolerance for the second system.
If some value btol is suitable for A*x = b, the larger value
btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.

Preconditioning is another way to reduce the number of iterations.
If it is possible to solve a related system M*x = b efficiently,
where M approximates A in some helpful way
(e.g. M - A has low rank or its elements are small relative to
those of A), LSMR may converge more rapidly on the system
      A*M(inverse)*y = b, ie AP*y = b with P = M(inverse)
after which x can be recovered by solving M*x = y (ie x = Py).
Observe that, if options%ctest = 3 is used (Fong and Saunders stopping)
then the convergence will be in the preconditioner norm.
If options%ctest /= 3, the user may choose the norm that is
used for the stopping criteria.

Currently, the user must combine applying the preconditioner
with performing matrix-vector products
eg suppose we have an incomplete factorization LL^T for A'A and we
are going to use this as a preconditioner.
Let the LS problem be  min||Ay - b||.
When LSMR requires products A*v, the user should compute AL^{-T}*v.
And when A'*v is required, the user should compute L^{-1}A'*v.
On completion, the user must recover the required solution y
by solving y = L^T*x, where x is the solution returned by LSMR.
In a future release, we may include separate returns for preconditioning
operations by extending the range of action values.
This will allow LSMR to return y to the user.

NOTE: If A is symmetric, LSMR should not be used
Alternatives are the symmetric conjugate-gradient method (CG)
and/or SYMMLQ.
SYMMLQ is an implementation of symmetric CG that applies to
any symmetric A and will converge more rapidly than LSMR.
If A is positive definite, there are other implementations of
symmetric CG that require slightly less work per iteration
than SYMMLQ (but will take the same number of iterations).


Notation
--------
The following quantities are used in discussing the subroutine
parameters:

Abar   =  (  A   )P,        bbar  =  (b)
          (damp*I)                   (0)

r      =  b - AP*y,         rbar  =  bbar - Abar*y

normr  =  sqrt( norm(r)**2  +  damp**2 * norm(y)**2 )
       =  norm( rbar )

eps    =  the relative precision of floating-point arithmetic.
          On most machines, eps is about 1.0e-7 and 1.0e-16
          in single and double precision respectively.
          We expect eps to be about 1e-16 always.

LSMR  minimizes the function normr with respect to y.
"""
cimport cython
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libc.float cimport DBL_MAX, DBL_EPSILON
from scipy.linalg.cython_blas cimport dscal, ddot, dnrm2

cdef double ZERO = 0.0
cdef double ONE = 1.0
cdef double ONEPT = 1.1
cdef int ONE_INT = 1

ENTER = ' Enter LSMR.'
EXITT = ' Exit  LSMR.'

MSG = ('The exact solution is  x = 0 ',
       'Ax - b is small enough, given atol, btol',
       'The least-squares solution is good enough, given atol ',
       'The estimate of cond(Abar) has exceeded conlim',
       'Ax - b is small enough for this machine ',
       'The LS solution is good enough for this machine ',
       'Cond(Abar) seems to be too large for this machine ',
       'The iteration limit has been reached',
       'Allocation error',
       'Deallocation error ',
       'Error: m or n is out-of-range ')

OUTPUT = {}
OUTPUT['1000'] = """
{status}, Least-squares solution of  Ax = b
       
 The matrix A has {rows} rows and {columns} columns   
    damp   = {damp}         
    options.atol   = {atol} 
    options.btol   = {btol} 
    options.conlim = {conlim}  
    options.itnlim = {itnlim}     
    options.localSize (no. of vectors for local reorthogonalization) = {local_vecs}
"""
OUTPUT['1050'] = OUTPUT['1000']
OUTPUT['1100'] = """
{status},  Least-squares solution of  Ax = b
 The matrix  A  has {rows} rows and {columns} columns
    damp   = {damp}
    options.itnlim = {itnlim}
    options.localSize (no. of vectors for local reorthogonalization) = {local_vecs}
"""
OUTPUT['1150'] = OUTPUT['1100']
OUTPUT['1200'] = """
format(/ "   Itn       y(1)            norm r       P'A'r   ", &
' Compatible    LS      norm AP   cond AP')
"""
OUTPUT['1250'] = """
   Itn       y(1)            norm r       P'A'r    norm AP   cond AP
"""
OUTPUT['1300'] = """
   Itn       y(1)           norm rbar    Abar'rbar Compatible    LS    norm Abar cond Abar
"""
OUTPUT['1350'] = """
   Itn       y(1)           norm rbar    Abar'rbar norm Abar cond Abar')
"""
OUTPUT['1400'] = "   Itn       y[0]"
OUTPUT['1500'] = "{itn:d}  {y0}  {normr} {normAPr} {extra1} {extra2} {extra3} {extra4}"
OUTPUT['1600'] = "{itn:d}  {y0}"
OUTPUT['2000'] = """
{status}    flag    = {flag:d}        itn     = {itn:d}
{status}    normAP  = {normAP}      condAP  = {condAP}
{status}    normb   = {normb}       normy   = {normy}
{status}    normr   = {normr}       normAPr = {normAPr}
"""
OUTPUT['2100'] = "{status}    flag    = {flag:d}        itn     = {itn:d}"
OUTPUT['3000'] = "{status}    {message}"  #  "{a}, {5x}, {a}"

cdef enum termination_flag:
    lsmr_stop_x0 = 0
    lsmr_stop_compatible = 1
    lsmr_stop_LS_atol = 2
    lsmr_stop_ill = 3
    lsmr_stop_Ax = 4
    lsmr_stop_LS = 5
    lsmr_stop_condAP = 6
    lsmr_stop_itnlim = 7
    lsmr_stop_allocation = 8
    lsmr_stop_deallocation = 9
    lsmr_stop_m_oor = 10

cdef inline double dot_product(double*a, double*b, int n):
    """Convenience function to make porting simpler"""
    return ddot(&n, a, &ONE_INT, b, &ONE_INT)

cdef bint allocate(double *array, int *array_size, int size):
    """Convenience function to make porting simpler"""
    array = <double *> malloc(size * sizeof(double))
    array_size[0] = size
    return array == NULL and size > 0

cdef bint allocate_2d(double *array, int *array_rows, int *array_cols, int rows, int cols):
    """Convenience function to make porting simpler"""
    array = <double *> malloc(rows * cols * sizeof(double))
    array_rows[0] = rows
    array_cols[0] = cols
    return array == NULL and cols > 0 and rows > 0

cdef bint allocated(int array_size):
    """Convenience function to make porting simpler"""
    return array_size > 0

cdef bint allocated_2d(int array_rows, int array_cols):
    """Convenience function to make porting simpler"""
    return array_rows > 0 and array_cols > 0

cdef bint deallocate(double *array, int *array_size):
    """Convenience function to make porting simpler"""
    if array_size[0] > 0:
        free(array)
    array_size[0] = 0
    return False

cdef bint deallocate_2d(double *array, int *array_rows, int *array_cols):
    """Convenience function to make porting simpler"""
    if array_rows[0] > 0 and array_cols[0] > 0:
        free(array)
    array_rows[0] = 0
    array_cols[0] = 0
    return False

# control data type
# These may be set before first call and must not be altered between calls.
# type :: lsmr_options
cdef struct lsmr_options_s:
    double atol
    # real(wp)    :: atol = sqrt(epsilon(one))
    # Used if ctest = 3 (Fong and Saunders stopping criteria)
    # In which case, must hold an estimate of the relative error in the data
    # defining the matrix A.  For example, if A is accurate to about 6 digits,
    # set atol = 1.0e-6.

    double btol
    # real(wp)    :: btol = sqrt(epsilon(one))
    # Used if ctest = 3 (Fong and Saunders stopping criteria)
    # In which case, must hold an estimate of the relative error in the data
    # defining the rhs b.  For example, if b is
    # accurate to about 6 digits, set btol = 1.0e-6.

    double conlim
    # real(wp)    :: conlim = 1/(10*sqrt(epsilon(one)))
    # Used if ctest = 3 (Fong and Saunders stopping criteria)
    # In which case, must hold an upper limit on cond(Abar), the apparent
    # condition number of the matrix Abar. Iterations will be terminated
    # if a computed estimate of cond(Abar) exceeds conlim.
    # This is intended to prevent certain small or
    # zero singular values of A or Abar from
    # coming into effect and causing unwanted growth in the computed solution.

    # conlim and damp may be used separately or
    # together to regularize ill-conditioned systems.

    # Normally, conlim should be in the range 1000 to 1/eps.
    # Suggested value:
    # conlim = 1/(100*eps)  for compatible systems,
    # conlim = 1/(10*sqrt(eps)) for least squares.

    # Note: Any or all of atol, btol, conlim may be set to zero.
    # The effect will be the same as the values eps, eps, 1/eps.

    int ctest
    # integer(ip) :: ctest = 3
    # Used to control convergence test. Possible values:
    # 1 : User may test for convergence when action = 1 is returned.
    #     The code does NOT compute normAP, condAP, normr, normAPr, normy.
    #     The code will only terminate if allocation (or deallocation) error
    #     or if itnlim is reached. Thus user is responsible for
    #     for determining when to stop.
    # 2 : As 1 but inform holds the latest estimates of normAP, condAP, normr,
    #     normAPr, normy  which the user may wish to use to monitor convergence.
    # 3 : The code determines if convergence has been achieved using
    #     Fong and Saunders stopping criteria.
    #     inform holds the latest estimates of normAP, condAP, normr, normAPr,
    #     normy.

    int itnlim
    # integer(ip) :: itnlim = -1
    # must hold an upper limit on the number of iterations.
    # Suggested value:
    # itnlim = n/2  for well-conditioned systems with clustered singular values
    # itnlim = 4*n  otherwise. If itnlim = -1 then we use 4*n.

    int itn_test
    # integer(ip) :: itn_test = -1
    # Determines how often control is passed to use to allow convergence
    # testing. It itn_test = -1 we use min(n,10).

    int localSize
    # integer(ip) :: localSize = 0
    # No. of vectors for local reorthogonalization.
    # 0       No reorthogonalization is performed.
    # >0      This many n-vectors "v" (the most recent ones)
    #         are saved for reorthogonalizing the next v.
    #         localSize need not be more than min(m,n).
    #         At most min(m,n) vectors will be allocated.

    int print_freq_head
    # integer(ip) :: print_freq_head = 20 # print frequency for the heading

    int print_freq_itn
    # integer(ip) :: print_freq_itn  = 10 # print frequency (iterations)

    int unit_diagnostics
    # integer(ip) :: unit_diagnostics = 6 # unit number for diagnostic printing.
    #  Printing is suppressed if <0.

    int unit_error
    # integer(ip) :: unit_error = 6 # unit number for error printing.
    #  Printing is suppressed if <0.

ctypedef lsmr_options_s lsmr_options

cdef void initialize_lsmr_options(lsmr_options *options):
    options.atol = sqrt(DBL_EPSILON)  # sqrt(epsilon(one))
    options.btol = sqrt(DBL_EPSILON)  # sqrt(epsilon(one))
    options.conlim = 1 / (10 * sqrt(DBL_EPSILON))  # 1/(10*sqrt(epsilon(one)))
    options.ctest = 3  # ctest = 3
    options.itnlim = -1  # itnlim = -1
    options.itn_test = -1  # itn_test = -1
    options.localSize = 0  # localSize = 0
    options.print_freq_head = 20  # print_freq_head = 20
    options.print_freq_itn = 10  # print_freq_itn  = 10
    options.unit_diagnostics = 6  # unit_diagnostics = 6
    options.unit_error = 6  # unit_error = 6

# information data type
# type :: lsmr_inform
cdef struct lsmr_inform_s:
    int flag
    # integer(ip) :: flag # Gives reason for termination (negative = failure):
    #
    # lsmr_stop_x0       x = 0  is the exact solution.
    #                    No iterations were performed.
    #
    # lsmr_stop_compatible     The equations A*x = b are probably compatible.
    #                    Norm(A*x - b) is sufficiently small, given the
    #                    values of atol and btol.
    #                    options%ctest = 3 only.
    #
    # lsmr_stop_LS_atol  If damp is zero:  A least-squares solution has
    #                    been obtained that is sufficiently accurate,
    #                    given the value of atol.
    #                    If damp is nonzero:  A damped least-squares
    #                    solution has been obtained that is sufficiently
    #                    accurate, given the value of atol.
    #                    options%ctest = 3 only.
    #
    # lsmr_stop_ill      An estimate of cond(Abar) has exceeded conlim.
    #                    The system A*x = b appears to be ill-conditioned,
    #                    or there could be an error in the computation of
    #                    A*v or A'*v.
    #                    options%ctest = 3 only.
    #
    #  lsmr_stop_Ax       Ax - b is small enough for this machine.
    #                    options%ctest = 3 only.
    #
    #  lsmr_stop_LS      The least-squares solution is good enough for this
    #                    machine. options%ctest = 3 only.
    #
    # lsmr_stop_condAP    The estimate of cond(Abar) seems to be too large
    #                    for this machine. options%ctest = 3 only.
    #
    # lsmr_stop_itnlim       :  The iteration limit has been reached.
    #
    # lsmr_stop_allocation   :  An array allocation failed.
    #
    # lsmr_stop_deallocation :  An array deallocation failed.
    #
    # lsmr_stop_m_oor        :  n < 1 or m < 1
    int itn
    # integer(ip) :: itn    # The number of iterations performed
    int stat
    # integer(ip) :: stat   # Fortran stat parameter
    double normb
    # real(wp)    :: normb  # holds norm of rhs
    double normAP
    # real(wp)    :: normAP  # Only holds information if options%ctest = 2 or 3.
    # In this case, holds estimate of the Frobenius norm of Abar.
    # This is the square-root of the sum of squares of the elements of Abar.
    # If damp is small and the columns of A have all been scaled to have
    # length 1.0, normAP should increase to roughly sqrt(n).
    # A radically different value for normAP may
    # indicate an error in the user-supplied
    # products with A or A' (or P or P'). A negative value
    # indicates that no estimate is currently available.
    double condAP
    # real(wp)    :: condAP  # Only holds information if options%ctest = 2 or 3.
    # In this case, holds estimate of cond(Abar), the condition
    # number of Abar.  A very high value of condAP
    # may again indicate an error in the products
    # with A or A'. A negative value indicates
    # that no estimate is currently available.
    double normr
    #real(wp)    :: normr  # Only holds information if options%ctest = 2 or 3.
    # In this case, holds estimate of the value of norm(rbar),
    # the function being minimized (see notation above).  This will be
    # small if A*x = b has a solution. A negative value
    # indicates that no estimate is currently available.
    double normAPr
    # real(wp)    :: normAPr  # Only holds information if options%ctest = 2 or 3.
    # In this case, holds estimate of the value of
    # norm( Abar'*rbar ), the norm of the residual for the normal equations.
    # This should be small in all cases.  (normAPr  will often be smaller
    # than the true value computed from the output vector x.) A negative value
    # indicates that no estimate is currently available.
    #
    double normy
# real(wp)    :: normy # Only holds information if options%ctest = 2 or 3.
# In this case, holds estimate of  norm(y) for the solution y.
# A negative value indicates that no estimate is currently available.

ctypedef lsmr_inform_s lsmr_inform

# define derived type to ensure local variables are saved safely
cdef struct lsmr_keep_s:
    double *h
    double *hbar
    double *localV
    # These are bounds for the allocatable arrays above
    int h_n
    int hbar_n
    int localV_n
    int localV_m
    bint damped
    bint localOrtho
    bint localVQueueFull
    bint show
    bint show_err
    int flag
    int itnlim
    int localOrthoCount
    int localOrthoLimit
    int localPointer
    int localVecs
    int pcount
    int itn_test
    int test_count
    int nout
    int nout_err
    double alpha
    double alphabar
    double beta
    double betad
    double betadd
    double cbar
    double ctol
    double d
    double maxrbar
    double minrbar
    double normA2
    double rho
    double rhobar
    double rhodold
    double sbar
    double tautildeold
    double thetatilde
    double zeta
    double zetabar
    double zetaold
    int branch

ctypedef lsmr_keep_s lsmr_keep

cdef struct lsmr_extra_s:
    # Additional types to pass between functions
    double test1
    double test2
    double test3
    double rtol
    double damp
    bint damped

ctypedef lsmr_extra_s lsmr_extra

cdef void initialize_lsmr_keep(lsmr_keep *keep):
    keep.branch = 0
    keep.h_n = 0
    keep.hbar_n = 0
    keep.localV_n = 0
    keep.localV_m = 0

cdef double d2norm(double a, double b):
    """
    Returns sqrt( a**2 + b**2 ) with precautions to avoid overflow
    """
    scale = abs(a) + abs(b)
    if scale > 0:
        return scale * sqrt((a / scale) ** 2 + (b / scale) ** 2)
    return 0.0

cdef bint lsmr_free_double(lsmr_keep *keep):
    """
    Deallocate components of keep.
    """
    cdef bint status = False
    print('h')
    if keep.h_n > 0:
        print(keep.h_n)
        free(keep.h)
        keep.h_n = 0
    print('hbar')
    if keep.hbar_n > 0:
        print(keep.hbar_n)
        free(keep.hbar)
        keep.hbar_n = 0
    print('localV')
    if keep.localV_n > 0 and keep.localV_m > 0:
        print(keep.localV_n, keep.localV_m)
        free(keep.localV)
        keep.localV_n = 0
        keep.localV_m = 0

    return status

cdef void localVEnqueue(lsmr_keep *keep, int n, double*v):
    """Store v into the circular buffer keep.localV."""
    cdef int i, offset
    if keep.localPointer < keep.localVecs:
        keep.localPointer += 1
    else:
        keep.localPointer = 0
        keep.localVQueueFull = True
    offset = keep.localPointer * n
    for i in range(n):
        keep.localV[offset + i] = v[i]

cdef void localVOrtho(lsmr_keep *keep, int n, double*v):
    """Perform local reorthogonalization of current v."""
    # do localOrthoCount = 1, keep%localOrthoLimit
    # real(wp)  :: d
    cdef double d
    cdef int localOrthoCount, offset, i

    # if (keep%localVQueueFull) then
    if keep.localVQueueFull:
        # keep%localOrthoLimit = keep%localVecs
        keep.localOrthoLimit = keep.localVecs
    # else
    else:
        # keep%localOrthoLimit = keep%localPointer
        keep.localOrthoLimit = keep.localPointer

    # do localOrthoCount = 1, keep%localOrthoLimit
    for localOrthoCount in range(keep.localOrthoLimit):
        offset = localOrthoCount * n
        # d = dot_product(v,keep%localV(1:n,localOrthoCount))
        d = dot_product(v, &(keep.localV[offset]), n)
        # v(1:n) = v(1:n) - d * keep%localV(1:n,localOrthoCount)
        for i in range(n):
            v[i] -= d * keep.localV[offset + i]

cdef void goto_10(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                  lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_10')
    cdef int i
    cdef double alpha_inv
    # keep%alpha = dnrm2 (n, v, 1)
    keep.alpha = dnrm2(&n, v, &ONE_INT)
    # if (keep%alpha .gt. zero) call dscal (n, (one/keep%alpha), v, 1)
    if keep.alpha > 0:
        alpha_inv = ONE / keep.alpha
        dscal(&n, &alpha_inv, v, &ONE_INT)
    # Exit if A'b = 0.
    # inform%normr  = keep%beta
    inform.normr = keep.beta
    # inform%normAPr = keep%alpha * keep%beta
    inform.normAPr = keep.alpha * keep.beta
    # if (inform%normAPr .eq. zero) then
    #   inform%normy = zero
    #   goto 800
    # end if
    if inform.normAPr == ZERO:
        inform.normy = ZERO
        goto_800(action, m, n, u, v, y, keep, options, inform, extra)
        return
    # Initialization for local reorthogonalization.
    # keep%localOrtho = .false.
    keep.localOrtho = False
    # if (keep%localVecs .gt. 0) then
    if keep.localVecs > 0:
        # keep%localPointer    = 1
        keep.localPointer = 1
        # keep%localOrtho      = .true.
        keep.localOrtho = True
        # keep%localVQueueFull = .false.
        keep.localVQueueFull = False
        # keep%localV(1:n,1)   = v(1:n)
        for i in range(n):
            keep.localV[i] = v[i]

    # Initialize variables for 1st iteration.
    # keep%zetabar  = keep%alpha*keep%beta
    keep.zetabar = keep.alpha * keep.beta
    # keep%alphabar = keep%alpha
    keep.alphabar = keep.alpha
    # keep%rho      = 1
    keep.rho = 1
    # keep%rhobar   = 1
    keep.rhobar = 1
    # keep%cbar     = 1
    keep.cbar = 1
    # keep%sbar     = 0
    keep.sbar = 0

    # keep%h(1:n)    = v(1:n)
    # keep%hbar(1:n) = zero
    for i in range(n):
        keep.h[i] = v[i]
        keep.hbar[i] = ZERO

    # Initialize variables for estimation of ||r||.
    # keep%betadd      = keep%beta
    keep.betadd = keep.beta
    # keep%betad       = 0
    keep.betad = 0
    # keep%rhodold     = 1
    keep.rhodold = 1
    # keep%tautildeold = 0
    keep.tautildeold = 0
    # keep%thetatilde  = 0
    keep.thetatilde = 0
    # keep%zeta        = 0
    keep.zeta = 0
    # keep%d           = 0
    keep.d = 0

    # Initialize variables for estimation of ||AP|| and cond(AP).
    # keep%normA2  = keep%alpha**2
    keep.normA2 = keep.alpha ** 2
    # keep%maxrbar = zero
    keep.maxrbar = ZERO
    # keep%minrbar = huge(one)
    keep.minrbar = DBL_MAX

    # inform%normb  = keep%beta
    inform.normb = keep.beta

    # Items for use in stopping rules (needed for control%options = 3 only).
    # keep%ctol   = zero
    keep.ctol = ZERO
    # if (options%conlim .gt. zero) keep%ctol = one/options%conlim
    if options.conlim > 0:
        keep.ctol = ONE / options.conlim

    # ! Heading for iteration log.
    # if (keep%show) then
    if keep.show:
        # if (options%ctest .eq. 3) then
        if options.ctest == 3:
            # if (keep%damped) then
            if keep.damped:
                # write (keep%nout,1300)
                print(OUTPUT['1300'])
            # else
            else:
                # write (keep%nout,1200)
                print(OUTPUT['1200'])
            # test1 = one
            extra.test1 = ONE
            # test2 = keep%alpha / keep%beta
            extra.test2 = keep.alpha / keep.beta
            # write (keep%nout,1500) &
            #   inform%itn,y(1),inform%normr,inform%normAPr,test1,test2
            print(OUTPUT['1500'].format(itn=inform.itn, y0=y[0], normr=inform.normr,
                                        normAPr=inform.normAPr, extra1=extra.test1,
                                        extra2=extra.test2, extra3='', extra4=''))
            # else if (options%ctest .eq. 2) then
        elif options.ctest == 2:
            # if (keep%damped) then
            if keep.damped:
                # write (keep%nout,1350)
                print(OUTPUT['1350'])
            # else
            else:
                # write (keep%nout,1250)
                print(OUTPUT['1250'])
            # write (keep%nout,1500) inform%itn,y(1),inform%normr,inform%normAPr
            print(OUTPUT['1500'].format(itn=inform.itn, y0=y[0], normr=inform.normr,
                                        normAPr=inform.normAPr, extra1='', extra2='',
                                        extra3='', extra4=''))
        # else
        else:
            # simple printing
            # write (keep%nout,1400)
            print(OUTPUT['1400'])
            # write (keep%nout,1600) inform%itn,y(1)
            print(OUTPUT['1600'].format(itn=inform.itn, y0=y[0]))

    goto_100(action, m, n, u, v, y, keep, options, inform, extra)
    return

cdef void goto_20(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                  lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_20')
    cdef double beta_inv, neg_beta
    # keep%beta   = dnrm2 (m, u, 1)
    keep.beta = dnrm2(&m, u, &ONE_INT)

    # if (keep%beta .gt. zero) then
    if keep.beta > 0:
        beta_inv = ONE / keep.beta
        # call dscal (m, (one/keep%beta), u, 1)
        dscal(&m, &beta_inv, u, &ONE_INT)
        # if (keep%localOrtho) then ! Store v into the circular buffer localV
        if keep.localOrtho:
            # call localVEnqueue     ! Store old v for local reorthog'n of new            localVEnqueue(keep, n, v)
            localVEnqueue(keep, n, v)
            # call dscal (n, (- keep%beta), v, 1)
            neg_beta = -keep.beta
            dscal(&n, &neg_beta, v, &ONE_INT)
            # action = 1            ! call Aprod2(m, n, v, u), i.e., v = v + P'A'*u
            action[0] = 1
            # keep%branch = 3
            keep.branch = 3
            return
    goto_30(action, m, n, u, v, y, keep, options, inform, extra)

cdef void goto_30(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                  lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_30')
    cdef double inv_alpha, alphahat, shat, chat
    # if (keep%beta .gt. zero) then
    if keep.beta > 0:
        # if (keep%localOrtho) then ! Perform local reorthogonalization of V.
        if keep.localOrtho:
            localVOrtho(keep, n, v)
            # call localVOrtho       ! Local-reorthogonalization of new v.
        # keep%alpha  = dnrm2 (n, v, 1)
        keep.alpha = dnrm2(&n, v, &ONE_INT)
        # if (keep%alpha .gt. zero) then
        if keep.alpha > 0:
            # call dscal (n, (one/keep%alpha), v, 1)
            inv_alpha = ONE / keep.alpha
            dscal(&n, &inv_alpha, v, &ONE_INT)

    # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
    #
    # ----------------------------------------------------------------
    # Construct rotation Qhat_{k,2k+1}.
    # alphahat = keep%alphabar
    alphahat = keep.alphabar
    # chat     = keep%alphabar/alphahat
    chat = keep.alphabar / alphahat
    # shat     = zero
    shat = ZERO
    # if (present_damp) then
    if extra.damped:
        # if (damp .ne. zero) then
        if extra.damp > 0:
            # alphahat = d2norm(keep%alphabar, damp)
            alphahat = d2norm(keep.alphabar, extra.damp)
            # chat     = keep%alphabar/alphahat
            chat = keep.alphabar / alphahat
            # shat     = damp/alphahat
            shat = extra.damp / alphahat

    # Use a plane rotation (Q_i) to turn B_i to R_i.
    # rhoold   = keep%rho
    rhoold = keep.rho
    # keep%rho = d2norm(alphahat, keep%beta)
    keep.rho = d2norm(alphahat, keep.beta)
    # c        = alphahat/keep%rho
    c = alphahat / keep.rho
    # s        = keep%beta/keep%rho
    s = keep.beta / keep.rho
    # thetanew = s*keep%alpha
    thetanew = s * keep.alpha
    # keep%alphabar = c*keep%alpha
    keep.alphabar = c * keep.alpha

    # Use a plane rotation (Qbar_i) to turn R_i^T into R_i^bar.
    #
    # rhobarold      = keep%rhobar
    rhobarold = keep.rhobar
    # keep%zetaold   = keep%zeta
    keep.zetaold = keep.zeta
    # thetabar       = keep%sbar*keep%rho
    thetabar = keep.sbar * keep.rho
    # rhotemp        = keep%cbar*keep%rho
    rhotemp = keep.cbar * keep.rho
    # keep%rhobar    = d2norm(keep%cbar*keep%rho, thetanew)
    keep.rhobar = d2norm(keep.cbar * keep.rho, thetanew)
    # keep%cbar      = keep%cbar*keep%rho/keep%rhobar
    keep.cbar = keep.cbar * keep.rho / keep.rhobar
    # keep%sbar      = thetanew/keep%rhobar
    keep.sbar = thetanew / keep.rhobar
    # keep%zeta      =   keep%cbar*keep%zetabar
    keep.zeta = keep.cbar * keep.zetabar
    # keep%zetabar   = - keep%sbar*keep%zetabar
    keep.zetabar = - keep.sbar * keep.zetabar

    # Update h, h_hat, y.
    # keep%hbar(1:n)  = keep%h(1:n) - &
    #      (thetabar*keep%rho/(rhoold*rhobarold))*keep%hbar(1:n)
    for i in range(n):
        keep.hbar[i] -= keep.hbar[i] - (thetabar * keep.rho / (rhoold * rhobarold)) * keep.hbar[i]
    # y(1:n)          = y(1:n) +      &
    #      (keep%zeta/(keep%rho*keep%rhobar))*keep%hbar(1:n)
    # keep%h(1:n)     = v(1:n) - (thetanew/keep%rho)*keep%h(1:n)
    for i in range(n):
        y[i] += (keep.zeta / (keep.rho * keep.rhobar)) * keep.hbar[i]
        keep.h[i] = v[i] - (thetanew / keep.rho) * keep.h[i]

    # Estimate ||r||.

    # Apply rotation Qhat_{k,2k+1}.
    # betaacute =   chat* keep%betadd
    betaacute = chat * keep.betadd
    # betacheck = - shat* keep%betadd
    betacheck = - shat * keep.betadd

    # ! Apply rotation Q_{k,k+1}.
    # betahat      =   c*betaacute
    betahat = c * betaacute
    # keep%betadd  = - s*betaacute
    keep.betadd = - s * betaacute

    # ! Apply rotation Qtilde_{k-1}.
    # ! betad = betad_{k-1} here.

    # thetatildeold = keep%thetatilde
    thetatildeold = keep.thetatilde
    # rhotildeold   = d2norm(keep%rhodold, thetabar)
    rhotildeold = d2norm(keep.rhodold, thetabar)
    # ctildeold     = keep%rhodold/rhotildeold
    ctildeold = keep.rhodold / rhotildeold
    # stildeold     = thetabar/rhotildeold
    stildeold = thetabar / rhotildeold
    # keep%thetatilde    = stildeold* keep%rhobar
    keep.thetatilde = stildeold * keep.rhobar
    # keep%rhodold       =   ctildeold* keep%rhobar
    keep.rhodold = ctildeold * keep.rhobar
    # keep%betad         = - stildeold*keep%betad + ctildeold*betahat
    keep.betad = - stildeold * keep.betad + ctildeold * betahat
    #
    # ! betad   = betad_k here.
    # ! rhodold = rhod_k  here.
    #
    # keep%tautildeold                                                       &
    #          = (keep%zetaold - thetatildeold*keep%tautildeold)/rhotildeold
    keep.tautildeold = (keep.zetaold - thetatildeold * keep.tautildeold) / rhotildeold
    # taud     = (keep%zeta - keep%thetatilde*keep%tautildeold)/keep%rhodold
    taud = (keep.zeta - keep.thetatilde * keep.tautildeold) / keep.rhodold
    # keep%d   = keep%d + betacheck**2
    keep.d = keep.d + betacheck ** 2
    #
    # if ((options%ctest .eq. 2) .or. (options%ctest .eq. 3)) then
    if options.ctest == 2 or options.ctest == 3:
        # inform%normr  = sqrt(keep%d + (keep%betad - taud)**2 + keep%betadd**2)
        inform.normr = sqrt(keep.d + (keep.betad - taud) ** 2 + keep.betadd ** 2)
        # ! Estimate ||A||.
        # keep%normA2   = keep%normA2 + keep%beta**2
        keep.normA2 = keep.normA2 + keep.beta ** 2
        # inform%normAP  = sqrt(keep%normA2)
        inform.normAP = sqrt(keep.normA2)
        # keep%normA2   = keep%normA2 + keep%alpha**2
        keep.normA2 = keep.normA2 + keep.alpha ** 2

        # Estimate cond(A).
        # keep%maxrbar    = max(keep%maxrbar,rhobarold)
        keep.maxrbar = max(keep.maxrbar, rhobarold)
        # if (inform%itn .gt. 1) then
        if inform.itn > 1:
            #    keep%minrbar = min(keep%minrbar,rhobarold)
            keep.minrbar = min(keep.minrbar, rhobarold)
        # inform%condAP    = max(keep%maxrbar,rhotemp)/min(keep%minrbar,rhotemp)
        inform.condAP = max(keep.maxrbar, rhotemp) / min(keep.minrbar, rhotemp)

        # Compute norms for convergence testing.
        # inform%normAPr  = abs(keep%zetabar)
        inform.normAPr = abs(keep.zetabar)
        # inform%normy   = dnrm2(n, y, 1)
        inform.normy = dnrm2(&n, y, &ONE_INT)

    # if (inform%itn .ge. keep%itnlim) inform%flag = lsmr_stop_itnlim
    if inform.itn >= keep.itnlim:
        inform.flag = lsmr_stop_itnlim
    # if (options%ctest.eq.3) then
    if options.ctest == 3:
        # ----------------------------------------------------------------
        #  Test for convergence.
        # ----------------------------------------------------------------
        #
        #  Now use these norms to estimate certain other quantities,
        #  some of which will be small near a solution.

        # test1   = inform%normr / inform%normb
        extra.test1 = inform.normr / inform.normb
        # test2   = inform%normAPr / (inform%normAP*inform%normr)
        extra.test2 = inform.normAPr / (inform.normAP * inform.normr)
        # test3   = one/inform%condAP
        extra.test3 = ONE / inform.condAP

        # t1      = test1 / (one + inform%normAP*inform%normy/inform%normb)
        t1 = extra.test1 / (ONE + inform.normAP * inform.normy / inform.normb)
        # rtol    = options%btol + &
        #                options%atol*inform%normAP*inform%normy/inform%normb
        extra.rtol = options.btol + options.atol * inform.normAP * inform.normy / inform.normb

        #  The following tests guard against extremely small values of
        #  atol, btol or ctol.  (The user may have set any or all of
        #  the parameters atol, btol, conlim  to 0.)
        #  The effect is equivalent to the normal tests using
        #  atol = eps,  btol = eps,  conlim = 1/eps.
        #
        #    if (one+test3 .le. one) inform%flag = lsmr_stop_condAP
        if ONE + extra.test3 <= ONE:
            inform.flag = lsmr_stop_condAP
        #    if (one+test2 .le. one) inform%flag = lsmr_stop_LS
        if ONE + extra.test2 <= ONE:
            inform.flag = lsmr_stop_LS
        #    if (one+t1    .le. one) inform%flag = lsmr_stop_Ax
        if ONE + t1 <= ONE:
            inform.flag = lsmr_stop_Ax

        #  Allow for tolerances set by the user.
        #    if (  test3   .le. keep%ctol   ) inform%flag = lsmr_stop_ill
        if extra.test3 <= keep.ctol:
            inform.flag = lsmr_stop_ill
        #    if (  test2   .le. options%atol) inform%flag = lsmr_stop_LS_atol
        if extra.test2 <= options.atol:
            inform.flag = lsmr_stop_LS_atol
        #    if (  test1   .le. rtol        ) inform%flag = lsmr_stop_compatible
        if extra.test1 <= extra.rtol:
            inform.flag = lsmr_stop_compatible
    # else
    else:
        # see if it is time to return to the user to test convergence
        # if (mod(keep%test_count,keep%itn_test) .eq. 0) then
        if (keep.test_count % keep.itn_test) == 0:
            # keep%test_count = 0
            keep.test_count = 0
            # action[0] = 3
            action[0] = 3
            # keep%branch = 4
            keep.branch = 4
            # return
            return
    goto_40(action, m, n, u, v, y, keep, options, inform, extra)
    return

cdef void goto_40(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                  lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_40')
    # prnt = .false.
    cdef bint prnt = False
    # if (keep%show) then
    if keep.show:
        # if (inform%itn .le.               options%print_freq_itn) prnt = .true.
        prnt = inform.itn <= options.print_freq_itn
        # if (inform%itn .ge. (keep%itnlim-options%print_freq_itn)) prnt = .true.
        prnt |= inform.itn >= (keep.itnlim - options.print_freq_itn)
        # if (mod(inform%itn,options%print_freq_itn)  .eq.       0) prnt = .true.
        prnt |= (inform.itn % options.print_freq_itn) == 0
        # if (options%ctest .eq. 3) then
        if options.ctest == 3:
            pass
            # if (test3 .le.  (onept*keep%ctol   )) prnt = .true.
            prnt |= extra.test3 <= ONEPT * keep.ctol
            # if (test2 .le.  (onept*options%atol)) prnt = .true.
            prnt |= extra.test2 <= ONEPT * options.atol
            # if (test1 .le.  (onept*rtol        )) prnt = .true.
            prnt |= extra.test1 <= ONEPT * extra.rtol
        # if (inform%flag .ne.  0) prnt = .true.
        prnt |= inform.flag != 0
    # if (prnt) then        ! Print a line for this iteration
    if prnt:
        # ! Check whether to print a heading first
        # if (keep%pcount .ge. options%print_freq_head) then
        if keep.pcount >= options.print_freq_head:
            # keep%pcount = 0
            keep.pcount = 0
            # if (options%ctest .eq. 3) then
            if options.ctest == 3:
                # if (keep%damped) then
                if keep.damped:
                    # write (keep%nout,1300)
                    print(OUTPUT['1300'])
                # else:
                else:
                    # write (keep%nout,1200)
                    print(OUTPUT['1200'])

            # else if (options%ctest .eq. 2) then
            elif options.ctest == 2:
                # if (keep%damped) then
                if keep.damped:
                    # write (keep%nout,1350)
                    print(OUTPUT['1350'])
                # else:
                else:
                    # write (keep%nout,1200)
                    print(OUTPUT['1250'])
            else:
                # write (keep%nout,1400)
                print(OUTPUT['1400'])
        #    keep%pcount = keep%pcount + 1
        keep.pcount += 1
        # if (options%ctest .eq. 3) then
        if options.ctest == 3:
            # write (keep%nout,1500) &
            #    inform%itn,y(1),inform%normr,inform%normAPr,    &
            #    test1,test2,inform%normAP,inform%condAP
            print(OUTPUT['1500'].format(itn=inform.itn,y0=y[0],normr=inform.normr,normAPr=inform.normAPr,
                                        extra1=extra.test1,extra2=extra.test2,
                                        extra3=inform.normAP,extra4=inform.condAP))
        # else if (options%ctest.eq.2) then
        elif options.ctest == 2:
            # write (keep%nout,1500) inform%itn,y(1),inform%normr, &
            #      inform%normAPr,inform%normAP,inform%condAP
            print(OUTPUT['1500'].format(itn=inform.itn,y0=y[0],normr=inform.normr,
                                        normAPr=inform.normAPr, extra1=inform.normAP,
                                        extra2=inform.condAP, extra3='', extra4=''))
        # else
        else:
            # write (keep%nout,1600) inform%itn,y(1)
            print(OUTPUT['1600'].format(inform.itn,y[0]))

    # if (inform%flag .eq. 0) goto 100
    if inform.flag == 0:
        goto_100(action, m, n, u, v, y, keep, options, inform, extra)
        return
    goto_800(action, m, n, u, v, y, keep, options, inform, extra)
    return

cdef void goto_100(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                   lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_100')
    cdef double neg_alpha

    # inform%itn = inform%itn + 1
    inform.itn += 1
    # ----------------------------------------------------------------
    # Perform the next step of the bidiagonalization to obtain the
    # next beta, u, alpha, v.  These satisfy
    #     beta*u = A*v  - alpha*u,
    #    alpha*v = A'*u -  beta*v.
    # ---------------------------------------------------------------
    # call dscal (m,(- keep%alpha), u, 1)
    neg_alpha = -keep.alpha
    dscal(&m, &neg_alpha, u, &ONE_INT)
    # action = 2             !  call Aprod1(m, n, v, u), i.e., u = u + AP*v
    action[0] = 2
    # keep%branch = 2
    keep.branch = 2
    # return
    return

cdef void goto_800(int *action, int m, int n, double *u, double *v, double *y, lsmr_keep *keep,
                   lsmr_options *options, lsmr_inform *inform, lsmr_extra *extra) except *:
    print('goto_800')
    # keep%flag = inform%flag
    keep.flag = inform.flag
    # if (keep%show) then ! Print the stopping condition.
    if keep.show:
        # if ((options%ctest .eq. 2) .or. (options%ctest .eq. 3)) then
        if options.ctest == 2 or options.ctest == 3:
            print(OUTPUT['2000'].format(status=EXITT, flag=inform.flag, itn=inform.itn,
                                        normAP=inform.normAP, condAP=inform.condAP,
                                        normb=inform.normb, normy=inform.normy,
                                        normr=inform.normr, normAPr=inform.normAPr))
        # else
        else:
            # write (keep%nout, 2100) exitt,inform%flag,inform%itn
            msg = OUTPUT['2100'].format(status=EXITT, flag=inform.flag, itn=inform.itn)
        # write (keep%nout, 3000) exitt, msg(inform%flag)
        print(OUTPUT['3000'].format(status=EXITT, message=MSG[inform.flag]))
    # terminate
    action[0] = 0

cdef void lsmr_error(int *action, lsmr_keep *keep, lsmr_inform *inform, int error_code) except *:
    inform.flag = error_code
    keep.flag = error_code
    if keep.show_err:
        print(MSG[error_code])
        action[0] = 0
    return

cdef void lsmr_solve_double(int *action, int m, int n, double *u, double *v, double *y,
                            lsmr_keep *keep, lsmr_options *options, lsmr_inform *inform,
                            lsmr_extra *extra, double damp=0.0) except *:
    cdef bint present_damp = damp > 0.0
    cdef int localOrthoCount
    cdef bint prnt
    cdef double alphahat, betaacute, betacheck, betahat, c, chat, \
        ctildeold, rhobarold, rhoold, rhotemp, rhotildeold, \
        rtol, s, shat, stildeold, t1, taud, test1, test2, test3, \
        thetabar, thetanew, thetatildeold

    cdef double beta_inv

    # on first call, initialize keep%branch and keep%flag
    if action[0] == 0:
        keep.branch = 0
        keep.flag = 0

    # Immediate return if we have already had an error
    if keep.flag > 0:
        return

    if keep.branch == 1:
        goto_10(action, m, n, u, v, y, keep, options, inform, extra)
        return
    elif keep.branch == 2:
        goto_20(action, m, n, u, v, y, keep, options, inform, extra)
        return
    elif keep.branch == 3:
        goto_30(action, m, n, u, v, y, keep, options, inform, extra)
        return
    elif keep.branch == 4:
        goto_40(action, m, n, u, v, y, keep, options, inform, extra)
        return

    #     ! Initialize.
    inform.flag = 0
    keep.flag = 0
    inform.itn = 0
    inform.normb = -ONE
    inform.normr = -ONE
    inform.normy = -ONE
    inform.normAP = -ONE
    inform.normAPr = -ONE
    inform.condAP = -ONE
    inform.stat = 0

    keep.localVecs = min(options.localSize, m, n)
    keep.nout = options.unit_diagnostics
    keep.show = keep.nout >= 0

    keep.nout_err = options.unit_error
    keep.show_err = keep.nout_err >= 0


    #if (keep%show) then
    if keep.show:
        # if (options%ctest .eq. 3) then
        if options.ctest == 3:
            # if (present(damp)) then
            if damp > ZERO:
                # write (keep%nout, 1000) enter,m,n,damp,options%atol,&
                #      options%btol,options%conlim,options%itnlim,keep%localVecs
                print(OUTPUT['1000'].format(status=ENTER, rows=m, columns=n, damp=damp, atol=options.atol,
                                            btol=options.btol, conlim=options.conlim,
                                            itnlim=options.itnlim, local_vecs=keep.localVecs))
            # else
            else:
                # write (keep%nout, 1050) enter,m,n,options%atol,&
                #      options%btol,options%conlim,options%itnlim,keep%localVecs
                print(OUTPUT['1050'].format(status=ENTER, rows=m, columns=n, damp='N/A', atol=options.atol,
                                            btol=options.btol, conlim=options.conlim,
                                            itnlim=options.itnlim, local_vecs=keep.localVecs))
        # else
        else:
            if damp > ZERO:
                # write (keep%nout, 1100) enter,m,n,damp,options%itnlim,keep%localVecs
                OUTPUT['1100'].format(status=ENTER, rows=m, columns=n, damp=damp, itnlim=options.itnlim,
                                      local_vecs=keep.localVecs)
            # else
            else:
                # write (keep%nout, 1150) enter,m,n,options%itnlim,keep%localVecs
                OUTPUT['1150'].format(status=ENTER, rows=m, columns=n, damp='N/A', itnlim=options.itnlim,
                                      local_vecs=keep.localVecs)

    print(1)
    # quick check of m and n
    if (n < 1) or (m < 1):
        lsmr_error(action, keep, inform, lsmr_stop_m_oor)
        return
    print(2)
    keep.pcount = 0  # print counter
    keep.test_count = 0  # testing for convergence counter
    keep.itn_test = options.itn_test  # how often to test for convergence
    if keep.itn_test <= 0:
        keep.itn_test = min(n, 10)
    keep.damped = False
    keep.damped = damp > ZERO
    keep.itnlim = options.itnlim
    print(3)
    if options.itnlim <= 0:
        keep.itnlim = 4 * n
    #
    #  allocate workspace (only do this if arrays not already allocated
    #  eg for an earlier problem)
    #
    print('h')
    print(keep.h_n)
    if not allocated(keep.h_n):
        print(keep.h_n)
        inform.stat = allocate(keep.h, &keep.h_n, n)
        print(keep.h_n)
        if inform.stat != 0:
            lsmr_error(action, keep, inform, lsmr_stop_allocation)
            return
    elif keep.h_n < n:
        print(keep.h_n)
        inform.stat = deallocate(keep.h, &keep.h_n)
        print(keep.h_n)
        if inform.stat != 0:
            lsmr_error(action, keep, inform, lsmr_stop_deallocation)
            return
        else:
            inform.stat = allocate(keep.h, &keep.h_n, n)
            if inform.stat != 0:
                lsmr_error(action, keep, inform, lsmr_stop_allocation)
                return
    #
    print('h_bar')
    if not allocated(keep.hbar_n):
        inform.stat = allocate(keep.hbar, &keep.hbar_n, n)
        if inform.stat != 0:
            lsmr_error(action, keep, inform, lsmr_stop_allocation)
            return
    elif keep.hbar_n < n:
        inform.stat = deallocate(keep.hbar, &keep.hbar_n)
        if inform.stat != 0:
            lsmr_error(action, keep, inform, lsmr_stop_deallocation)
            return
        else:
            inform.stat = allocate(keep.hbar, &keep.hbar_n, n)
            if inform.stat != 0:
                lsmr_error(action, keep, inform, lsmr_stop_allocation)
                return
    print('localVecs')
    print(keep.localVecs)
    if keep.localVecs > 0:
        if not allocated_2d(keep.localV_n, keep.localV_m):
            inform.stat = allocate_2d(keep.localV, &keep.localV_n, &keep.localV_m, n,
                                      keep.localVecs)
            if inform.stat != 0:
                lsmr_error(action, keep, inform, lsmr_stop_allocation)
                return
    elif keep.localV_n < n or keep.localV_m < keep.localVecs:
        inform.stat = deallocate_2d(keep.localV, &keep.localV_m, &keep.localV_n)
        if inform.stat != 0:
            lsmr_error(action, keep, inform, lsmr_stop_deallocation)
            return
        else:
            inform.stat = allocate_2d(keep.localV, &keep.localV_n, &keep.localV_m, n,
                                      keep.localVecs)
            if inform.stat != 0:
                lsmr_error(action, keep, inform, lsmr_stop_allocation)
                return
    print('Done!!')
    #-------------------------------------------------------------------
    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
    #-------------------------------------------------------------------
    for i in range(n):
        v[i] = ZERO
        y[i] = ZERO

    keep.alpha = ZERO
    keep.beta = dnrm2(&m, u, &ONE_INT)

    if keep.beta > ZERO:
        beta_inv = ONE / keep.beta
        dscal(&m, &beta_inv, u, &ONE_INT)
        action[0] = 1
        keep.branch = 1
        return

    else:
        # Exit if b=0.
        inform.normAP = -ONE
        inform.condAP = -ONE
        goto_800(action, m, n, u, v, y, keep, options, inform, extra)
        return


# Takes b and computes u = u + A * v (A in CSC format)
cdef void matrix_mult(int m, int n, int *ptr, int *row, double *val, double *v, double *u):
    cdef int i, j, k
    for j in range(0, n):
        for k in range(ptr[j], ptr[j+1]):
            i = row[k]
            u[i] += val[k]*v[j]


# Takes b and computes v = v + A^T * u (A in CSC format) */
cdef void matrix_mult_trans(int m, int n, int *ptr, int *row, double *val, double *u, double *v):
    cdef int i, j, k
    for j in range(n):
        for k in range(ptr[j], ptr[j+1]):
            i = row[k]
            v[j] += val[k]*u[i]


def experiment():
    cdef lsmr_extra extra
    cdef lsmr_options options
    cdef lsmr_keep keep
    cdef lsmr_inform inform

    initialize_lsmr_options(&options)
    initialize_lsmr_keep(&keep)

    cdef int m = 5, n = 3
    cdef int ptr[4]
    ptr[:] = [   0,             3,         5,                 9 ]
    cdef int row[9]
    row[:] = [  0,   2,   3,   1,   3,    0,   2,    3,   4 ]
    cdef double val[9]
    val[:] = [ 1.0, 2.0, 5.0, 2.0, 3.0, -1.0, 2.0, -2.0, 6.0 ]
    #  Data for rhs b
    cdef double b[5]
    b[:] = [1.0, 1.0, 1.0, 1.0, 1.0]
    cdef double damp = 0.0
    # prepare for LSMR calls (using no preconditioning) */
    cdef int action = 0;
    cdef double u[5]
    cdef double v[3]
    cdef double x[3]
    for i in range(m):
        u[i] = b[i]
    cdef bint done = False
    options.print_freq_itn = 1
    options.print_freq_head = 1
    lsmr_solve_double(&action, m, n, &u[0], &v[0], &x[0], &keep, &options, &inform, &extra, damp)
    print(action)
    print('Out!')
    print(keep.h_n)
    print(keep.hbar_n)
    print(keep.localV_n)
    print(keep.localV_m)
    #lsmr_free_double(&keep)
