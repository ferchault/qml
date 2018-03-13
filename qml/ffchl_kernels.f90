! http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions
module ffchl_kernels

    implicit none

    public :: kernel

contains


subroutine gaussian_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2

    l2 = s11 + s22 - 2.0d0*s12 
    do i = 1, size(k)
        k(i) = exp(l2 * parameters(1,i)) 
    enddo

end subroutine gaussian_kernel 


subroutine linear_kernel(s12, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = s12
    enddo

end subroutine linear_kernel 


subroutine polynomial_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = (parameters(1,i) * s12 + parameters(2,i))**parameters(3,i)
    enddo

end subroutine polynomial_kernel 


subroutine sigmoid_kernel(s12, parameters, k)

    implicit none
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i

    do i = 1, size(k)
        k(i) = tanh(parameters(1,i) * s12 + parameters(2,i))
    enddo

end subroutine sigmoid_kernel 


subroutine multiquadratic_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = sqrt(l2 + parameters(1,i)**2)
    enddo

end subroutine multiquadratic_kernel 


subroutine inverse_multiquadratic_kernel(s11, s22, s12, parameters, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = 1.0d0 / sqrt(l2 + parameters(i,1)**2)
    enddo

end subroutine inverse_multiquadratic_kernel 


subroutine bessel_kernel(s12, parameters, k)

    implicit none

    double precision, intent(in) :: s12
    double precision, intent(in), dimension(:,:) :: parameters

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    double precision :: sigma
    integer :: v 
    integer :: n
    
    do i = 1, size(k)
        k(i) = BESSEL_JN(int(parameters(2,i)), parameters(1,i) *s12) & 
            & / (s12**(-parameters(3,i)*(parameters(2,i) + 1)))
    enddo

end subroutine bessel_kernel 

subroutine l2_kernel(s11, s22, s12, k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12

    double precision, intent(out), dimension(:) :: k

    integer :: i
    double precision :: l2
    
    l2 = s11 + s22 - 2.0d0*s12 

    do i = 1, size(k)
        k(i) = l2
    enddo

end subroutine l2_kernel 


function kernel(s11, s22, s12, kernel_idx, parameters) result(k)

    implicit none

    double precision, intent(in) :: s11
    double precision, intent(in) :: s22
    double precision, intent(in) :: s12
    integer, intent(in) :: kernel_idx
    double precision, intent(in), dimension(:,:) :: parameters

    integer :: n
    double precision, allocatable, dimension(:) :: k

    n = size(parameters, dim=2)
    allocate(k(n))

    if (kernel_idx == 1) then
        call gaussian_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 2) then
        call linear_kernel(s12, k)
    
    else if (kernel_idx == 3) then
        call polynomial_kernel(s12, parameters, k)

    else if (kernel_idx == 4) then
        call sigmoid_kernel(s12, parameters, k)

    else if (kernel_idx == 5) then
        call multiquadratic_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 6) then
        call inverse_multiquadratic_kernel(s11, s22, s12, parameters, k)

    else if (kernel_idx == 7) then
        call bessel_kernel(s12, parameters, k)
    
    else if (kernel_idx == 8) then
        call l2_kernel(s11, s22, s12, k)

    else
        write (*,*) "QML ERROR: Unknown kernel function requested:", kernel_idx
        stop
    endif


end function kernel

end module ffchl_kernels
