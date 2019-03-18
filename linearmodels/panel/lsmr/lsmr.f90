! Copyright (c) 2010, 2012 David Fong, Michael Saunders, Stanford University
! Copyright (c) 2014-2016 Science and Technology Facilites Council (STFC)
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice,
!    this list of conditions and the following disclaimer.
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
! ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
! LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
! CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
! SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
! INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
! CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
! ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE. 
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!                                LSMR
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
! LSMR   solves Ax = b or min ||Ax - b|| with or without damping,
! using the iterative algorithm of David Fong and Michael Saunders:
!     http://www.stanford.edu/group/SOL/software/lsmr.html
!
! The original LSMR code can be found at
! http://web.stanford.edu/group/SOL/software/lsmr/
! It is maintained by
!     David Fong       <clfong@stanford.edu>
!     Michael Saunders <saunders@stanford.edu>
!     Systems Optimization Laboratory (SOL)
!     Stanford University
!     Stanford, CA 94305-4026, USA

! This version uses reverse communication so that control is passed back
! to the user to perform products with A or A' and to perform preconditioning.
! In addition, the user may choose to use their own stopping rule(s)
! or to apply the stopping criteria of Fong and Saunders.
! A number of options are available (see the derived type lsmr_type_options).
!
!    Dec 2015: SPRAL version developed to allow user more control
!              over preconditioning. Also comprehensive test deck written.
! 23 Nov 2015: revised reverse communication interface written by
!              Nick Gould and Jennifer Scott.
!              lsmrDataModule.f90 no longer used.
!              Automatic arrays no longer used.
!              Option for user to test convergence (or to use Fong and 
!              Saunders test).
!              Uses dnrm2 and dscal (for closer compatibility with 
!              Fong and Saunders code ... easier to compare results).
!              Sent to Saunders; made available on above SOL webpage.
! 20 May 2014: initial reverse-communication version written by Nick Gould
! 02 May 2014: With damp>0, flag=2 was incorrectly set to flag=3
!              (so incorrect stopping message was printed).  Fixed.
! 28 Jan 2014: In lsmrDataModule.f90:
!              ip added for integer(ip) declarations.
!              dnrm2 and dscal coded directly
!              (no longer use lsmrblasInterface.f90 or lsmrblas.f90).
! 07 Sep 2010: Local reorthogonalization now works (localSize > 0).
! 17 Jul 2010: F90 LSMR derived from F90 LSQR and lsqr.m.
!              Aprod1, Aprod2 implemented via f90 interface.
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ! Precision
    ! ---------
    ! The number of iterations required by LSMR will decrease
    ! if the computation is performed in higher precision.
    ! At least 15-digit arithmetic should normally be used.
    ! "real(wp)" declarations should normally be 8-byte words.

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

module spral_lsmr
  implicit none

  private
  public :: lsmr_keep, lsmr_options, lsmr_inform
  public :: lsmr_solve, lsmr_free

  integer(4),  parameter :: ip = kind( 0 )
  integer(4),  parameter :: wp = kind( 0.0d+0 )
  real(wp),    parameter :: zero = 0.0_wp, one = 1.0_wp, onept = 1.1_wp

  ! termination flags(explained below)
  integer(ip), parameter :: lsmr_stop_x0              = 0
  integer(ip), parameter :: lsmr_stop_compatible      = 1
  integer(ip), parameter :: lsmr_stop_LS_atol         = 2
  integer(ip), parameter :: lsmr_stop_ill             = 3
  integer(ip), parameter :: lsmr_stop_Ax              = 4
  integer(ip), parameter :: lsmr_stop_LS              = 5
  integer(ip), parameter :: lmsr_stop_condAP          = 6
  integer(ip), parameter :: lsmr_stop_itnlim          = 7
  integer(ip), parameter :: lsmr_stop_allocation      = 8
  integer(ip), parameter :: lsmr_stop_deallocation    = 9
  integer(ip), parameter :: lsmr_stop_m_oor           = 10

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Make interfaces generic.
  interface lsmr_solve
     module procedure lsmr_solve_double
  end interface lsmr_solve

  interface lsmr_free
     module procedure lsmr_free_double
  end interface lsmr_free

  !-------------------------------------------------------------------
  ! LSMR  finds a solution x to the following problems:
  !
  ! 1. Unsymmetric equations:    Solve  A*x = b
  !
  ! 2. Linear least squares:     Solve  A*x = b
  !                              in the least-squares sense
  !
  ! 3. Damped least squares:     Solve  (   A    )*x = ( b )
  !                                     ( damp*I )     ( 0 )
  !                              in the least-squares sense
  !
  ! where A is a matrix with m rows and n columns, b is an m-vector,
  ! and damp is a scalar.  (All quantities are real.)
  ! The matrix A is treated as a linear operator.  It is accessed
  ! by reverse communication, that is requests for matrix-vector 
  ! products are passed back to the user.
  !
  ! If preconditioning used, solves
  ! 1. Unsymmetric equations:    Solve  AP*y = b
  !
  ! 2. Linear least squares:     Solve  AP*y = b
  !                              in the least-squares sense
  !
  ! 3. Damped least squares:     Solve  (   A    )*P*y = ( b )
  !                                     ( damp*I )       ( 0 )
  !                              in the least-squares sense
  ! Code computes y and user must recover x = Py
  !
  ! LSMR uses an iterative method to approximate the solution.
  ! The number of iterations required to reach a certain accuracy
  ! depends strongly on the scaling of the problem.  Poor scaling of
  ! the rows or columns of A should therefore be avoided where
  ! possible.
  !
  ! For example, in problem 1 the solution is unaltered by
  ! row-scaling.  If a row of A is very small or large compared to
  ! the other rows of A, the corresponding row of ( A  b ) should be
  ! scaled up or down.
  !
  ! In problems 1 and 2, the solution x is easily recovered
  ! following column-scaling.  Unless better information is known,
  ! the nonzero columns of A should be scaled so that they all have
  ! the same Euclidean norm (e.g., 1.0).
  !
  ! In problem 3, there is no freedom to re-scale if damp is
  ! nonzero.  However, the value of damp should be assigned only
  ! after attention has been paid to the scaling of A.
  !
  ! The parameter damp is intended to help regularize
  ! ill-conditioned systems, by preventing the true solution from
  ! being very large.  Another aid to regularization is provided by
  ! the parameter condAP, which may be used to terminate iterations
  ! before the computed solution becomes very large.
  !
  ! Note that x is not an input parameter.
  ! If some initial estimate x0 is known and if damp = 0,
  ! one could proceed as follows:
  !
  ! 1. Compute a residual vector     r0 = b - A*x0.
  ! 2. Use LSMR to solve the system  A*dx = r0.
  ! 3. Add the correction dx to obtain a final solution x = x0 + dx.
  !
  ! This requires that x0 be available before the first call 
  ! to LSMR and after the final call. 
  ! To judge the benefits, suppose LSMR takes k1 iterations
  ! to solve A*x = b and k2 iterations to solve A*dx = r0.
  ! If x0 is "good", norm(r0) will be smaller than norm(b).
  ! If the same stopping tolerances atol and btol are used for each
  ! system, k1 and k2 will be similar, but the final solution x0 + dx
  ! should be more accurate.  The only way to reduce the total work
  ! is to use a larger stopping tolerance for the second system.
  ! If some value btol is suitable for A*x = b, the larger value
  ! btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
  !
  ! Preconditioning is another way to reduce the number of iterations.
  ! If it is possible to solve a related system M*x = b efficiently,
  ! where M approximates A in some helpful way
  ! (e.g. M - A has low rank or its elements are small relative to
  ! those of A), LSMR may converge more rapidly on the system
  !       A*M(inverse)*y = b, ie AP*y = b with P = M(inverse)
  ! after which x can be recovered by solving M*x = y (ie x = Py).
  ! Observe that, if options%ctest = 3 is used (Fong and Saunders stopping)
  ! then the convergence will be in the preconditioner norm.
  ! If options%ctest /= 3, the user may choose the norm that is
  ! used for the stopping criteria.
  !
  ! Currently, the user must combine applying the preconditioner
  ! with performing matrix-vector products
  ! eg suppose we have an incomplete factorization LL^T for A'A and we
  ! are going to use this as a preconditioner.
  ! Let the LS problem be  min||Ay - b||.
  ! When LSMR requires products A*v, the user should compute AL^{-T}*v.
  ! And when A'*v is required, the user should compute L^{-1}A'*v.
  ! On completion, the user must recover the required solution y
  ! by solving y = L^T*x, where x is the solution returned by LSMR.
  ! In a future release, we may include separate returns for preconditioning
  ! operations by extending the range of action values.
  ! This will allow LSMR to return y to the user.
  !
  ! NOTE: If A is symmetric, LSMR should not be used!
  ! Alternatives are the symmetric conjugate-gradient method (CG)
  ! and/or SYMMLQ.
  ! SYMMLQ is an implementation of symmetric CG that applies to
  ! any symmetric A and will converge more rapidly than LSMR.
  ! If A is positive definite, there are other implementations of
  ! symmetric CG that require slightly less work per iteration
  ! than SYMMLQ (but will take the same number of iterations).
  !
  !
  ! Notation
  ! --------
  ! The following quantities are used in discussing the subroutine
  ! parameters:
  !
  ! Abar   =  (  A   )P,        bbar  =  (b)
  !           (damp*I)                   (0)
  !
  ! r      =  b - AP*y,         rbar  =  bbar - Abar*y
  !
  ! normr  =  sqrt( norm(r)**2  +  damp**2 * norm(y)**2 )
  !        =  norm( rbar )
  !
  ! eps    =  the relative precision of floating-point arithmetic.
  !           On most machines, eps is about 1.0e-7 and 1.0e-16
  !           in single and double precision respectively.
  !           We expect eps to be about 1e-16 always.
  !
  ! LSMR  minimizes the function normr with respect to y.

! ------------------------------------------------------------------

  ! control data type
  ! These may be set before first call and must not be altered between calls.
  type :: lsmr_options

     real(wp)    :: atol = sqrt(epsilon(one))  
     ! Used if ctest = 3 (Fong and Saunders stopping criteria)
     ! In which case, must hold an estimate of the relative error in the data
     ! defining the matrix A.  For example, if A is accurate to about 6 digits,
     ! set atol = 1.0e-6.

     real(wp)    :: btol = sqrt(epsilon(one)) 
     ! Used if ctest = 3 (Fong and Saunders stopping criteria)
     ! In which case, must hold an estimate of the relative error in the data
     ! defining the rhs b.  For example, if b is
     ! accurate to about 6 digits, set btol = 1.0e-6.

     real(wp)    :: conlim = 1/(10*sqrt(epsilon(one))) 
     ! Used if ctest = 3 (Fong and Saunders stopping criteria)
     ! In which case, must hold an upper limit on cond(Abar), the apparent
     ! condition number of the matrix Abar. Iterations will be terminated 
     ! if a computed estimate of cond(Abar) exceeds conlim.
     ! This is intended to prevent certain small or
     ! zero singular values of A or Abar from
     ! coming into effect and causing unwanted growth in the computed solution.
    
     ! conlim and damp may be used separately or
     ! together to regularize ill-conditioned systems.
    
     ! Normally, conlim should be in the range 1000 to 1/eps.
     ! Suggested value:
     ! conlim = 1/(100*eps)  for compatible systems,
     ! conlim = 1/(10*sqrt(eps)) for least squares.
    
     ! Note: Any or all of atol, btol, conlim may be set to zero.
     ! The effect will be the same as the values eps, eps, 1/eps.

     integer(ip) :: ctest = 3
     ! Used to control convergence test. Possible values:
     ! 1 : User may test for convergence when action = 1 is returned.
     !     The code does NOT compute normAP, condAP, normr, normAPr, normy.
     !     The code will only terminate if allocation (or deallocation) error
     !     or if itnlim is reached. Thus user is responsible for
     !     for determining when to stop.
     ! 2 : As 1 but inform holds the latest estimates of normAP, condAP, normr,  
     !     normAPr, normy  which the user may wish to use to monitor convergence.
     ! 3 : The code determines if convergence has been achieved using 
     !     Fong and Saunders stopping criteria.
     !     inform holds the latest estimates of normAP, condAP, normr, normAPr, 
     !     normy.

     integer(ip) :: itnlim = -1
     ! must hold an upper limit on the number of iterations.
     ! Suggested value:
     ! itnlim = n/2  for well-conditioned systems with clustered singular values
     ! itnlim = 4*n  otherwise. If itnlim = -1 then we use 4*n.

     integer(ip) :: itn_test = -1
     ! Determines how often control is passed to use to allow convergence 
     ! testing. It itn_test = -1 we use min(n,10).
    
     integer(ip) :: localSize = 0 
     ! No. of vectors for local reorthogonalization.
     ! 0       No reorthogonalization is performed.
     ! >0      This many n-vectors "v" (the most recent ones)
     !         are saved for reorthogonalizing the next v.
     !         localSize need not be more than min(m,n).
     !         At most min(m,n) vectors will be allocated.
    
     integer(ip) :: print_freq_head = 20 ! print frequency for the heading

     integer(ip) :: print_freq_itn  = 10 ! print frequency (iterations)

     integer(ip) :: unit_diagnostics = 6 ! unit number for diagnostic printing. 
     !  Printing is suppressed if <0. 

     integer(ip) :: unit_error = 6 ! unit number for error printing. 
     !  Printing is suppressed if <0. 

  end type lsmr_options

! ------------------------------------------------------------------

  ! information data type
  type :: lsmr_inform

     integer(ip) :: flag ! Gives reason for termination (negative = failure):
     !
     ! lsmr_stop_x0       x = 0  is the exact solution.
     !                    No iterations were performed.
     !
     ! lsmr_stop_compatible     The equations A*x = b are probably compatible.
     !                    Norm(A*x - b) is sufficiently small, given the
     !                    values of atol and btol. 
     !                    options%ctest = 3 only. 
     !
     ! lsmr_stop_LS_atol  If damp is zero:  A least-squares solution has
     !                    been obtained that is sufficiently accurate,
     !                    given the value of atol.  
     !                    If damp is nonzero:  A damped least-squares
     !                    solution has been obtained that is sufficiently
     !                    accurate, given the value of atol. 
     !                    options%ctest = 3 only. 
     !
     ! lsmr_stop_ill      An estimate of cond(Abar) has exceeded conlim.
     !                    The system A*x = b appears to be ill-conditioned,
     !                    or there could be an error in the computation of
     !                    A*v or A'*v.
     !                    options%ctest = 3 only.
     !
     !  lsmr_stop_Ax       Ax - b is small enough for this machine.
     !                    options%ctest = 3 only. 
     !
     !  lsmr_stop_LS      The least-squares solution is good enough for this
     !                    machine. options%ctest = 3 only.
     !
     ! lmsr_stop_condAP    The estimate of cond(Abar) seems to be too large 
     !                    for this machine. options%ctest = 3 only.
     !
     ! lsmr_stop_itnlim       :  The iteration limit has been reached.
     !
     ! lsmr_stop_allocation   :  An array allocation failed.
     !
     ! lsmr_stop_deallocation :  An array deallocation failed.
     !
     ! lsmr_stop_m_oor        :  n < 1 or m < 1
     !
     !
     integer(ip) :: itn    ! The number of iterations performed

     integer(ip) :: stat   ! Fortran stat parameter

     real(wp)    :: normb  ! holds norm of rhs

     real(wp)    :: normAP  ! Only holds information if options%ctest = 2 or 3.
     ! In this case, holds estimate of the Frobenius norm of Abar.
     ! This is the square-root of the sum of squares of the elements of Abar.
     ! If damp is small and the columns of A have all been scaled to have 
     ! length 1.0, normAP should increase to roughly sqrt(n).
     ! A radically different value for normAP may
     ! indicate an error in the user-supplied
     ! products with A or A' (or P or P'). A negative value
     ! indicates that no estimate is currently available.
     !
     real(wp)    :: condAP  ! Only holds information if options%ctest = 2 or 3.
     ! In this case, holds estimate of cond(Abar), the condition
     ! number of Abar.  A very high value of condAP
     ! may again indicate an error in the products 
     ! with A or A'. A negative value indicates
     ! that no estimate is currently available.
     !
     real(wp)    :: normr  ! Only holds information if options%ctest = 2 or 3.
     ! In this case, holds estimate of the value of norm(rbar),
     ! the function being minimized (see notation above).  This will be 
     ! small if A*x = b has a solution. A negative value
     ! indicates that no estimate is currently available.
     !
     real(wp)    :: normAPr  ! Only holds information if options%ctest = 2 or 3.
     ! In this case, holds estimate of the value of
     ! norm( Abar'*rbar ), the norm of the residual for the normal equations.
     ! This should be small in all cases.  (normAPr  will often be smaller 
     ! than the true value computed from the output vector x.) A negative value
     ! indicates that no estimate is currently available.
     !
     real(wp)    :: normy ! Only holds information if options%ctest = 2 or 3.
     ! In this case, holds estimate of  norm(y) for the solution y.
     ! A negative value indicates that no estimate is currently available.

  end type lsmr_inform

! ------------------------------------------------------------------

!  define derived type to ensure local variables are saved safely
  type :: lsmr_keep
     private
     real(wp), allocatable :: h(:), hbar(:), localV(:,:)
     logical     :: damped, localOrtho, localVQueueFull
     logical     :: show, show_err
     integer(ip) :: flag
     integer(ip) :: itnlim
     integer(ip) :: localOrthoCount, localOrthoLimit, localPointer,           &
                    localVecs, pcount
     integer(ip) :: itn_test, test_count
     integer(ip) :: nout, nout_err
     real(wp)    :: alpha, alphabar, beta, betad, betadd, cbar, ctol,         &
                    d, maxrbar, minrbar, normA2, rho, rhobar, rhodold, sbar,  &
                    tautildeold, thetatilde, zeta, zetabar, zetaold
     integer(ip) :: branch = 0
  end type lsmr_keep

contains

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lsmr_solve_double(action, m, n, u, v, y, keep, options, inform, &
       damp)

    implicit none

    integer(ip), intent(inout) :: action ! This parameter controls
    ! the action. On initial entry, must be set to 0.
    ! On subsequent call, action should be unchanged by the user UNLESS
    ! the user wishes to terminate the computation. In this case, it
    ! should be set to 3 and one further call to LSMR
    ! then made (see below).
    ! On return, action determines what the user is required to do.
    ! Possible values and their consequences are:
    ! 
    ! 0. Computation has terminated.
    !    action = 0 is returned if either an error has been encountered
    !    or convergence has been achieved with options%ctest = 3.
    !    It is also returned after a call with action = 3 (see below).
    !
    ! 1. The caller must compute v = v + A'*u without altering u, and 
    ! then re-call the subroutine. The vectors u and v are avaialble 
    ! in the arrays u and v (see below). Only v should be altered.
    !
    ! 2. The caller must compute u = u + A*v  without altering v, and 
    ! then re-call the subroutine. The vectors u and v are avaialble 
    ! in the arrays u and v (see below). Only u should be altered.
    !
    ! FINAL CALL: If the user wishes to deallocate components of keep
    ! a call with action = 10 should be made. Note: that if a series of
    ! problems is being solved, a call with action=10 is only needed
    ! after the final problem has been solved.

    integer(ip), intent(in)  :: m  ! the number of rows in A.

    integer(ip), intent(in)  :: n  ! the number of columns in A.

    real(wp),    intent(inout) :: u(m) ! Prior to the first call, u
    ! must hold the rhs vector b. u is then the vector used to communicate u 
    ! when further action is required (see action, above).

    real(wp),    intent(inout) :: v(n) ! The vector used to communicate v 
    !  when further action is required (see action, above).

    real(wp),    intent(inout) :: y(n) ! Returns the computed solution y
    ! of the preconditioned system.

    type ( lsmr_keep ), intent( inout) :: keep 
    ! A variable of type lsmr_keep that is used to
    ! preserve internal data between reverse-communication
    ! calls. The components are private.

    type ( lsmr_options ), intent( inout) :: options
    ! A variable of type lsmr_options that is used to control the options.
    ! It should not be altered by the user.

    type ( lsmr_inform ), intent( inout) :: inform
    ! A variable of type lsmr_info_type that is used to hold information.
    ! It should not be altered by the user.

    real(wp),intent(in), optional  :: damp ! The damping parameter for problem 3 
    ! above. (damp should be absent or set to 0.0 for problems 1 and 2.) 
    ! If the system A*x = b 
    ! is incompatible, values of damp in the range 0 to sqrt(eps)*norm(A)
    ! will probably have a negligible effect. Larger values of damp will tend 
    ! to decrease the norm of x and reduce the number of 
    ! iterations required by LSMR.
    !
    ! The work per iteration and the storage needed
    ! by LSMR are the same for all values of damp.

    intrinsic   :: abs, dot_product, min, max, sqrt

    ! Local arrays and variables
    integer(ip) :: localOrthoCount
    logical     :: prnt
    real(wp)    :: alphahat, betaacute, betacheck, betahat, c, chat,          &
                   ctildeold, rhobarold, rhoold, rhotemp, rhotildeold,        &
                   rtol, s, shat, stildeold, t1, taud, test1, test2, test3,   &
                   thetabar, thetanew, thetatildeold

    real(wp) :: dnrm2

    ! Local constants

    character(len=*), parameter :: enter = ' Enter LSMR.  '
    character(len=*), parameter :: exitt = ' Exit  LSMR.  '
    character(len=*), parameter :: msg(0:10 ) =                   &
      (/ 'The exact solution is  x = 0                          ', &
         'Ax - b is small enough, given atol, btol              ', &
         'The least-squares solution is good enough, given atol ', &
         'The estimate of cond(Abar) has exceeded conlim        ', &
         'Ax - b is small enough for this machine               ', &
         'The LS solution is good enough for this machine       ', &
         'Cond(Abar) seems to be too large for this machine     ', &
         'The iteration limit has been reached                  ', &
         ' Allocation error                                     ', &
         ' Deallocation error                                   ', &
         ' Error: m or n is out-of-range                        ' /)
    !-------------------------------------------------------------------

    ! on first call, initialize keep%branch and keep%flag
    if (action .eq. 0) then
       keep%branch = 0
       keep%flag = 0
    end if

    ! Immediate return if we have already had an error 
    if (keep%flag .gt. 0) return

    ! on other calls, jump to the appropriate place after 
    ! reverse-communication

    select case ( keep%branch )
    case ( 1 )
       goto 10
    case ( 2 )
       goto 20
    case ( 3 )
       goto 30
    case ( 4 )
       goto 40
    end select

    ! Initialize.
    inform%flag = 0
    keep%flag   = 0 
    inform%itn  = 0
    inform%normb  = -one
    inform%normr  = -one
    inform%normy  = -one
    inform%normAP  = -one
    inform%normAPr = -one
    inform%condAP  = -one
    inform%stat   = 0

    keep%localVecs = min(options%localSize,m,n)
    keep%nout      = options%unit_diagnostics
    keep%show      = (keep%nout .ge. 0)

    keep%nout_err  = options%unit_error
    keep%show_err  = (keep%nout_err .ge. 0)

    if (keep%show) then
       if (options%ctest .eq. 3) then
          if (present(damp)) then
             write (keep%nout, 1000) enter,m,n,damp,options%atol,&
                  options%btol,options%conlim,options%itnlim,keep%localVecs
          else
             write (keep%nout, 1050) enter,m,n,options%atol,&
                  options%btol,options%conlim,options%itnlim,keep%localVecs
          end if
       else
          if (present(damp)) then
             write (keep%nout, 1100) enter,m,n,damp,options%itnlim,keep%localVecs
          else
             write (keep%nout, 1150) enter,m,n,options%itnlim,keep%localVecs
          end if
       end if
    end if

    ! quick check of m and n
    if ((n .lt. 1) .or. (m .lt. 1)) then
       inform%flag = lsmr_stop_m_oor
       keep%flag   = lsmr_stop_m_oor
       if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_m_oor)
       action = 0
       return
    end if
    
    keep%pcount     = 0            ! print counter
    keep%test_count = 0            ! testing for convergence counter
    keep%itn_test   = options%itn_test  ! how often to test for convergence
    if (keep%itn_test .le. 0) keep%itn_test = min(n,10)
    keep%damped     = .false.
    if (present(damp)) keep%damped = (damp .gt. zero)  !
    keep%itnlim = options%itnlim
    if (options%itnlim .le. 0) keep%itnlim = 4*n

    !  allocate workspace (only do this if arrays not already allocated
    !  eg for an earlier problem)

    if (.not. allocated(keep%h)) then
       allocate( keep%h(n), stat = inform%stat )
       if ( inform%stat .ne. 0 ) then
          inform%flag = lsmr_stop_allocation
          keep%flag   = lsmr_stop_allocation
          if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
          action = 0
          return
       end if
    else if (size(keep%h) .lt. n) then
       deallocate (keep%h, stat = inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = lsmr_stop_deallocation
          keep%flag   = lsmr_stop_deallocation
          if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_deallocation)
          action = 0
          return
       else
          allocate( keep%h(n), stat = inform%stat )
          if ( inform%stat .ne. 0 ) then
             inform%flag = lsmr_stop_allocation
             keep%flag   = lsmr_stop_allocation
             if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
             action = 0
             return
          end if
       end if
    end if

    if (.not. allocated(keep%hbar)) then
       allocate( keep%hbar(n), stat = inform%stat )
       if ( inform%stat .ne. 0 ) then
          inform%flag = lsmr_stop_allocation
          keep%flag   = lsmr_stop_allocation
          if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
          action = 0
          return
       end if
    else if (size(keep%hbar) .lt. n) then
       deallocate (keep%hbar, stat = inform%stat)
       if (inform%stat .ne. 0) then
          inform%flag = lsmr_stop_deallocation
          keep%flag   = lsmr_stop_deallocation
          if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_deallocation)
          action = 0
          return
       else
          allocate( keep%hbar(n), stat = inform%stat )
          if ( inform%stat .ne. 0 ) then
             inform%flag = lsmr_stop_allocation
             keep%flag   = lsmr_stop_allocation
             if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
             action = 0
             return
          end if
       end if
    end if

    if (keep%localVecs .gt. 0) then
       if (.not. allocated(keep%localV)) then
          allocate( keep%localV(n,keep%localVecs), stat = inform%stat )
          if ( inform%stat .ne. 0 ) then
             inform%flag = lsmr_stop_allocation
             keep%flag   = lsmr_stop_allocation
             if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
             action = 0
             return
          end if
       else if ((size(keep%localV,1) .lt. n) .or. &
            (size(keep%localV,2) .lt. keep%localVecs)) then
          deallocate (keep%localV, stat = inform%stat)
          if (inform%stat .ne. 0) then
             inform%flag = lsmr_stop_deallocation
             keep%flag   = lsmr_stop_deallocation
             if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_deallocation)
             action = 0
             return
          else
             allocate( keep%localV(n,keep%localVecs), stat = inform%stat )
             if ( inform%stat .ne. 0 ) then
                inform%flag = lsmr_stop_allocation
                keep%flag   = lsmr_stop_allocation
                if (keep%show_err) write (keep%nout_err,'(a)') msg(lsmr_stop_allocation)
                action = 0
                return
             end if
          end if
       end if
    end if

    !-------------------------------------------------------------------
    ! Set up the first vectors u and v for the bidiagonalization.
    ! These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
    !-------------------------------------------------------------------
    v(1:n) = zero
    y(1:n) = zero

    keep%alpha  = zero
    keep%beta   = dnrm2 (m, u, 1)

    if (keep%beta .gt. zero) then
       call dscal (m, (one/keep%beta), u, 1)
       action = 1                  ! call Aprod2(m, n, v, u), i.e., v = P'A'*u
       keep%branch = 1
       return

    else
      ! Exit if b=0.
       inform%normAP = -one
       inform%condAP = -one
       goto 800
    end if

 10 continue

    keep%alpha = dnrm2 (n, v, 1)
    if (keep%alpha .gt. zero) call dscal (n, (one/keep%alpha), v, 1)

    ! Exit if A'b = 0.
    inform%normr  = keep%beta
    inform%normAPr = keep%alpha * keep%beta
    if (inform%normAPr .eq. zero) then
       inform%normy = zero
       goto 800
    end if

    ! Initialization for local reorthogonalization.

    keep%localOrtho = .false.
    if (keep%localVecs .gt. 0) then
       keep%localPointer    = 1
       keep%localOrtho      = .true.
       keep%localVQueueFull = .false.
       keep%localV(1:n,1)   = v(1:n)
    end if

    ! Initialize variables for 1st iteration.

    keep%zetabar  = keep%alpha*keep%beta
    keep%alphabar = keep%alpha
    keep%rho      = 1
    keep%rhobar   = 1
    keep%cbar     = 1
    keep%sbar     = 0

    keep%h(1:n)    = v(1:n)
    keep%hbar(1:n) = zero

    ! Initialize variables for estimation of ||r||.

    keep%betadd      = keep%beta
    keep%betad       = 0
    keep%rhodold     = 1
    keep%tautildeold = 0
    keep%thetatilde  = 0
    keep%zeta        = 0
    keep%d           = 0

    ! Initialize variables for estimation of ||AP|| and cond(AP).

    keep%normA2  = keep%alpha**2
    keep%maxrbar = zero
    keep%minrbar = huge(one)

    inform%normb  = keep%beta

    ! Items for use in stopping rules (needed for control%options = 3 only).
    keep%ctol   = zero
    if (options%conlim .gt. zero) keep%ctol = one/options%conlim

    ! Heading for iteration log.

    if (keep%show) then
       if (options%ctest .eq. 3) then
          if (keep%damped) then
             write (keep%nout,1300)
          else
             write (keep%nout,1200)
          end if
          test1 = one
          test2 = keep%alpha / keep%beta
          write (keep%nout,1500) &
               inform%itn,y(1),inform%normr,inform%normAPr,test1,test2
       else if (options%ctest .eq. 2) then
          if (keep%damped) then
             write (keep%nout,1350)
          else
             write (keep%nout,1250)
          end if
          write (keep%nout,1500) inform%itn,y(1),inform%normr,inform%normAPr
       else
          ! simple printing
          write (keep%nout,1400)
          write (keep%nout,1600) inform%itn,y(1)
       end if
    end if

    !===================================================================
    ! Main iteration loop.
    !===================================================================
100 continue

       inform%itn = inform%itn + 1

       !----------------------------------------------------------------
       ! Perform the next step of the bidiagonalization to obtain the
       ! next beta, u, alpha, v.  These satisfy
       !     beta*u = A*v  - alpha*u,
       !    alpha*v = A'*u -  beta*v.
       !----------------------------------------------------------------
       call dscal (m,(- keep%alpha), u, 1)
       action = 2             !  call Aprod1(m, n, v, u), i.e., u = u + AP*v
       keep%branch = 2
       return

 20    continue
       keep%beta   = dnrm2 (m, u, 1)

       if (keep%beta .gt. zero) then
          call dscal (m, (one/keep%beta), u, 1)
          if (keep%localOrtho) then ! Store v into the circular buffer localV
             call localVEnqueue     ! Store old v for local reorthog'n of new v.
          end if
          call dscal (n, (- keep%beta), v, 1)
          action = 1            ! call Aprod2(m, n, v, u), i.e., v = v + P'A'*u
          keep%branch = 3
          return
       end if

 30    continue
       if (keep%beta .gt. zero) then
          if (keep%localOrtho) then ! Perform local reorthogonalization of V.
             call localVOrtho       ! Local-reorthogonalization of new v.
          end if
          keep%alpha  = dnrm2 (n, v, 1)
          if (keep%alpha .gt. zero) then
             call dscal (n, (one/keep%alpha), v, 1)
          end if
       end if
    
       ! At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
    
       !----------------------------------------------------------------
       ! Construct rotation Qhat_{k,2k+1}.
    
       alphahat = keep%alphabar
       chat     = keep%alphabar/alphahat
       shat     = zero
       if (present(damp)) then
          if (damp .ne. zero) then
             alphahat = d2norm(keep%alphabar, damp)
             chat     = keep%alphabar/alphahat
             shat     = damp/alphahat
          end if
       end if

       ! Use a plane rotation (Q_i) to turn B_i to R_i.

       rhoold   = keep%rho
       keep%rho = d2norm(alphahat, keep%beta)
       c        = alphahat/keep%rho
       s        = keep%beta/keep%rho
       thetanew = s*keep%alpha
       keep%alphabar = c*keep%alpha

       ! Use a plane rotation (Qbar_i) to turn R_i^T into R_i^bar.
    
       rhobarold      = keep%rhobar
       keep%zetaold   = keep%zeta
       thetabar       = keep%sbar*keep%rho
       rhotemp        = keep%cbar*keep%rho
       keep%rhobar    = d2norm(keep%cbar*keep%rho, thetanew)
       keep%cbar      = keep%cbar*keep%rho/keep%rhobar
       keep%sbar      = thetanew/keep%rhobar
       keep%zeta      =   keep%cbar*keep%zetabar
       keep%zetabar   = - keep%sbar*keep%zetabar

       ! Update h, h_hat, y.

       keep%hbar(1:n)  = keep%h(1:n) - &
            (thetabar*keep%rho/(rhoold*rhobarold))*keep%hbar(1:n)
       y(1:n)          = y(1:n) +      &
            (keep%zeta/(keep%rho*keep%rhobar))*keep%hbar(1:n)
       keep%h(1:n)     = v(1:n) - (thetanew/keep%rho)*keep%h(1:n)

       ! Estimate ||r||.
    
       ! Apply rotation Qhat_{k,2k+1}.
       betaacute =   chat* keep%betadd
       betacheck = - shat* keep%betadd

       ! Apply rotation Q_{k,k+1}.
       betahat      =   c*betaacute
       keep%betadd  = - s*betaacute

       ! Apply rotation Qtilde_{k-1}.
       ! betad = betad_{k-1} here.

       thetatildeold = keep%thetatilde
       rhotildeold   = d2norm(keep%rhodold, thetabar)
       ctildeold     = keep%rhodold/rhotildeold
       stildeold     = thetabar/rhotildeold
       keep%thetatilde    = stildeold* keep%rhobar
       keep%rhodold       =   ctildeold* keep%rhobar
       keep%betad         = - stildeold*keep%betad + ctildeold*betahat

       ! betad   = betad_k here.
       ! rhodold = rhod_k  here.

       keep%tautildeold                                                       &
                = (keep%zetaold - thetatildeold*keep%tautildeold)/rhotildeold
       taud     = (keep%zeta - keep%thetatilde*keep%tautildeold)/keep%rhodold
       keep%d   = keep%d + betacheck**2

       if ((options%ctest .eq. 2) .or. (options%ctest .eq. 3)) then
          inform%normr  = sqrt(keep%d + (keep%betad - taud)**2 + keep%betadd**2)
    
          ! Estimate ||A||.
          keep%normA2   = keep%normA2 + keep%beta**2
          inform%normAP  = sqrt(keep%normA2)
          keep%normA2   = keep%normA2 + keep%alpha**2
    
          ! Estimate cond(A).
          keep%maxrbar    = max(keep%maxrbar,rhobarold)
          if (inform%itn .gt. 1) then 
             keep%minrbar = min(keep%minrbar,rhobarold)
          end if
          inform%condAP    = max(keep%maxrbar,rhotemp)/min(keep%minrbar,rhotemp)

          ! Compute norms for convergence testing.
          inform%normAPr  = abs(keep%zetabar)
          inform%normy   = dnrm2(n, y, 1)
       end if


       if (inform%itn .ge. keep%itnlim) inform%flag = lsmr_stop_itnlim
       if (options%ctest.eq.3) then

          !----------------------------------------------------------------
          ! Test for convergence.
          !----------------------------------------------------------------

          ! Now use these norms to estimate certain other quantities,
          ! some of which will be small near a solution.

          test1   = inform%normr / inform%normb
          test2   = inform%normAPr / (inform%normAP*inform%normr)
          test3   = one/inform%condAP

          t1      = test1 / (one + inform%normAP*inform%normy/inform%normb)
          rtol    = options%btol + &
                      options%atol*inform%normAP*inform%normy/inform%normb

          ! The following tests guard against extremely small values of
          ! atol, btol or ctol.  (The user may have set any or all of
          ! the parameters atol, btol, conlim  to 0.)
          ! The effect is equivalent to the normal tests using
          ! atol = eps,  btol = eps,  conlim = 1/eps.

          if (one+test3 .le. one) inform%flag = lmsr_stop_condAP
          if (one+test2 .le. one) inform%flag = lsmr_stop_LS
          if (one+t1    .le. one) inform%flag = lsmr_stop_Ax

          ! Allow for tolerances set by the user.

          if (  test3   .le. keep%ctol   ) inform%flag = lsmr_stop_ill
          if (  test2   .le. options%atol) inform%flag = lsmr_stop_LS_atol
          if (  test1   .le. rtol        ) inform%flag = lsmr_stop_compatible

       else
          ! see if it is time to return to the user to test convergence
          if (mod(keep%test_count,keep%itn_test) .eq. 0) then
             keep%test_count = 0
             action = 3
             keep%branch = 4
             return
          end if
       end if

 40    continue

       !----------------------------------------------------------------
       ! See if it is time to print something.
       !----------------------------------------------------------------
       prnt = .false.
       if (keep%show) then
          if (inform%itn .le.               options%print_freq_itn) prnt = .true.
          if (inform%itn .ge. (keep%itnlim-options%print_freq_itn)) prnt = .true.
          if (mod(inform%itn,options%print_freq_itn)  .eq.       0) prnt = .true.
          if (options%ctest .eq. 3) then
             if (test3 .le.  (onept*keep%ctol   )) prnt = .true.
             if (test2 .le.  (onept*options%atol)) prnt = .true.
             if (test1 .le.  (onept*rtol        )) prnt = .true.
          end if
          if (inform%flag .ne.  0) prnt = .true.

          if (prnt) then        ! Print a line for this iteration
             ! Check whether to print a heading first
             if (keep%pcount .ge. options%print_freq_head) then 
                keep%pcount = 0
                if (options%ctest .eq. 3) then
                   if (keep%damped) then
                      write (keep%nout,1300)
                   else
                      write (keep%nout,1200)
                   end if
                else if (options%ctest .eq. 2) then
                   if (keep%damped) then
                      write (keep%nout,1350)
                   else
                      write (keep%nout,1250)
                   end if
                else
                   write (keep%nout,1400)
                end if
             end if
             keep%pcount = keep%pcount + 1
             if (options%ctest .eq. 3) then
                write (keep%nout,1500) &
                     inform%itn,y(1),inform%normr,inform%normAPr,    &
                     test1,test2,inform%normAP,inform%condAP
             else if (options%ctest.eq.2) then
                write (keep%nout,1500) inform%itn,y(1),inform%normr, &
                     inform%normAPr,inform%normAP,inform%condAP
             else
                write (keep%nout,1600) inform%itn,y(1)
             end if
          end if
       end if

       if (inform%flag .eq. 0) goto 100

       !===================================================================
       ! End of iteration loop.
       !===================================================================

       ! Come here if inform%normAPr = 0, or if normal exit, or iteration
       ! count exceeded.

800    continue
       keep%flag = inform%flag

       if (keep%show) then ! Print the stopping condition.
          if ((options%ctest .eq. 2) .or. (options%ctest .eq. 3)) then 
             write (keep%nout, 2000)                 &
                  exitt,inform%flag,inform%itn,      &
                  exitt,inform%normAP,inform%condAP, &
                  exitt,inform%normb, inform%normy,  &
                  exitt,inform%normr,inform%normAPr
          else
             write (keep%nout, 2100)                 &
                  exitt,inform%flag,inform%itn
          end if
          write (keep%nout, 3000) exitt, msg(inform%flag)
       end if

    ! terminate
    action = 0
    return

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 1000 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' damp   =', es22.14         &
     / ' options%atol   =', es10.2, &
     / ' options%btol   =', es10.2, &
     / ' options%conlim =', es10.2  &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1050 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' options%atol   =', es10.2, &
     / ' options%btol   =', es10.2, &
     / ' options%conlim =', es10.2  &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1100 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' damp   =', es22.14         &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1150 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1200 format(/ "   Itn       y(1)            norm r       P'A'r   ", &
      ' Compatible    LS      norm AP   cond AP')
 1250 format(/ "   Itn       y(1)            norm r       P'A'r   ", &
      ' norm AP   cond AP')
 1300 format(/ "   Itn       y(1)           norm rbar    Abar'rbar", &
      ' Compatible    LS    norm Abar cond Abar')
 1350 format(/ "   Itn       y(1)           norm rbar    Abar'rbar", &
      ' norm Abar cond Abar')
 1400 format(/ "   Itn       y(1)")
 1500 format(i6, 2es17.9, 5es10.2)
 1600 format(i6, es17.9)
 2000 format(/ a, 5x, 'flag    =', i2,    15x, 'itn     =', i8      &
      /      a, 5x, 'normAP  =', es12.5, 5x, 'condAP  =', es12.5   &
      /      a, 5x, 'normb   =', es12.5, 5x, 'normy   =', es12.5   &
      /      a, 5x, 'normr   =', es12.5, 5x, 'normAPr =', es12.5)
 2100 format(/ a, 5x, 'flag    =', i2,   15x,  'itn     =', i8)
 3000 format(a, 5x, a)

  contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    function d2norm( a, b )

      implicit none

      real(wp)             :: d2norm
      real(wp), intent(in) :: a, b

      !-------------------------------------------------------------------
      ! d2norm returns sqrt( a**2 + b**2 )
      ! with precautions to avoid overflow.
      !
      ! 21 Mar 1990: First version.
      ! 17 Sep 2007: Fortran 90 version.
      ! 24 Oct 2007: User real(wp) instead of compiler option -r8.
      !-------------------------------------------------------------------

      intrinsic            :: abs, sqrt
      real(wp)             :: scale

      d2norm = zero
      scale = abs(a) + abs(b)
      if (scale .gt. zero) d2norm = scale*sqrt((a/scale)**2 + (b/scale)**2)

    end function d2norm

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine localVEnqueue

      implicit none

      ! Store v into the circular buffer keep%localV.

      if (keep%localPointer < keep%localVecs) then
         keep%localPointer = keep%localPointer + 1
      else
         keep%localPointer = 1
         keep%localVQueueFull = .true.
      end if
      keep%localV(1:n,keep%localPointer) = v

    end subroutine localVEnqueue

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine localVOrtho

      implicit none

      ! Perform local reorthogonalization of current v.

      real(wp)  :: d

      if (keep%localVQueueFull) then
         keep%localOrthoLimit = keep%localVecs
      else
         keep%localOrthoLimit = keep%localPointer
      end if

      do localOrthoCount = 1, keep%localOrthoLimit
         d = dot_product(v,keep%localV(1:n,localOrthoCount))
         v(1:n) = v(1:n)    -    d * keep%localV(1:n,localOrthoCount)
      end do

    end subroutine localVOrtho

  end subroutine lsmr_solve_double

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine lsmr_free_double (keep, stat)

    implicit none

    ! Routine to deallocate components of keep. Failure is indicated
    ! by nonzero stat value. No printing.

    integer(ip), intent(out)           :: stat ! Fortran stat parameter
    type ( lsmr_keep ), intent( inout) :: keep 

    integer :: st


    stat = 0
    if (allocated(keep%hbar)) then
       deallocate( keep%hbar, stat = st )
       stat = max(stat,st)
    end if

    if (allocated(keep%h)) then
       deallocate( keep%h, stat = st )
       stat = max(stat,st)
    end if

    if (allocated(keep%localV)) then
       deallocate( keep%localV, stat = st )
       stat = max(stat,st)
    end if

  end subroutine lsmr_free_double


end module spral_lsmr
