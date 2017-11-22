subroutine fget_atomic_force_alphas_fchl(x1, forces, nneigh1, &
       & sigmas, lambda, na1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, alphas)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:), intent(in) :: forces

    double precision, allocatable, dimension(:,:,:,:,:) :: x1_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas
    double precision, intent(in) :: lambda

    ! Number of molecules
    integer, intent(in) :: na1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    double precision, allocatable, dimension(:,:) :: y
    !DEC$ attributes align: 64:: y

    double precision, allocatable, dimension(:,:,:)  :: kernel_delta
    !DEC$ attributes align: 64:: kernel_delta

    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
    !DEC$ attributes align: 64:: kernel_scratch

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar1_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi1_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: info

    double precision :: ang_norm2

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    double precision :: dx_sign

    integer :: maxneigh1

    write (*,*) "INIT"

    maxneigh1 = maxval(nneigh1)
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x1, dim=1)
    dim2 = size(x1, dim=2)
    dim3 = size(x1, dim=3)

    allocate(x1_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na1
        x1_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x1(i,:,:), nneigh1(i), dx_numm, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, cut_start, cut_distance, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    write (*,*) "SELF SCALAR"
    !$OMP PARALLEL DO
    do i = 1, na1
        self_scalar1(i) = scalar(x1(i,:,:), x1(i,:,:), &
            & nneigh1(i), nneigh1(i), ksi1(i,:), ksi1(i,:), &
            & sinp1(i,:,:,:), sinp1(i,:,:,:), &
            & cosp1(i,:,:,:), cosp1(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO

    write (*,*) "ALLOCATE AND CLEAR"
    allocate(kernel_delta(na1,na1,nsigmas))

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    allocate(ksi1_displaced(maxneigh1))
    ksi1_displaced = 0.0d0

    allocate(fourier_displaced(2, pmax1, order, maxneigh1))
    fourier_displaced = 0.0d0

    allocate(y(na1,nsigmas))
    y = 0.0d0

    alphas = 0.0d0

    do xyz = 1, 3

        kernel_delta = 0.0d0

        ! Plus/minus displacemenets
        do pm = 1, 2

            ! Get the sign and magnitude of displacement
            dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

            write (*,*) "DERIVATIVE", xyz, ((dble(pm) - 1.5d0) * 2.0d0)

            !$OMP PARALLEL DO schedule(dynamic), &
            !$OMP& PRIVATE(l2dist,self_scalar1_displaced,ksi1_displaced,fourier_displaced)
            do i = 1, na1

                ksi1_displaced(:) = &
                    & get_twobody_weights(x1_displaced(i,:,:,xyz,pm), nneigh1(i), &
                    & two_body_power, cut_start, cut_distance, maxneigh1)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x1_displaced(i,:,:,xyz,pm), &
                    & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

                self_scalar1_displaced = scalar(x1_displaced(i,:,:,xyz,pm), &
                    & x1_displaced(i,:,:,xyz,pm), nneigh1(i), nneigh1(i), &
                    & ksi1_displaced(:), ksi1_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x1_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh1(i), nneigh1(j), ksi1_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    do k = 1, nsigmas
                        kernel_delta(i,j,k) = kernel_delta(i,j,k) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
            !$OMP END PARALLEL DO
        enddo

        do k = 1, nsigmas

            write (*,*) "    DSYRK", sigmas(k)
            ! DSYRK call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
            call dsyrk("U", "T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & 1.0d0, kernel_scratch(:,:,k), na1)

            write (*,*) "    DGEMV", sigmas(k)
            ! DGEMV call corresponds to y := 1.0 * K^T * F + 1.0 * y
            call dgemv("T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                            & forces(:,xyz), 1, 1.0d0, y(:,k), 1)
        enddo

    enddo

    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + lambda
        enddo
    enddo

    do k = 1, nsigmas
        write (*,*) "  DPOTRF"
        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        write (*,*) "  DPOTRS"
        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        alphas(k,:) = y(:,k)
    enddo

    deallocate(kernel_delta)
    deallocate(kernel_scratch)
    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(x1_displaced)

end subroutine fget_atomic_force_alphas_fchl


subroutine fget_atomic_force_kernels_fchl(x1, x2, nneigh1, nneigh2, &
       & sigmas, na1, na2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (na1,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1

    ! fchl descriptors for the prediction set, format (na2,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x2

    double precision, allocatable, dimension(:,:,:,:,:) :: x2_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1
    integer, dimension(:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: na1
    integer, intent(in) :: na2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,3,na2,na1), intent(out) :: kernels
    ! double precision, allocatable, dimension(:,:,:,:)  :: l2_displaced

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar2_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi2_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm

    double precision :: ang_norm2

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0



    maxneigh1 = maxval(nneigh1(:))
    maxneigh2 = maxval(nneigh2(:))
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    pmax2 = 0
    do a = 1, na2
        pmax2 = max(pmax2, int(maxval(x2(a,2,:nneigh2(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x2, dim=1)
    dim2 = size(x2, dim=2)
    dim3 = size(x2, dim=3)

    allocate(x2_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na2
        x2_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x2(i,:,:), nneigh2(i), dx_numm, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, cut_start, cut_distance, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO


    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        self_scalar1(i) = scalar(x1(i,:,:), x1(i,:,:), &
            & nneigh1(i), nneigh1(i), ksi1(i,:), ksi1(i,:), &
            & sinp1(i,:,:,:), sinp1(i,:,:,:), &
            & cosp1(i,:,:,:), cosp1(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO


    allocate(ksi2_displaced(maxneigh2))
    allocate(fourier_displaced(2, pmax2, order, maxneigh2))
    ksi2_displaced = 0.0d0
    fourier_displaced = 0.0d0

    write (*,*) "KERNEL DERIVATIVES"
    do xyz = 1, 3
        do pm = 1, 2

            ! Get the sign and magnitude of displacement
            dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

            write (*,*) "DERIVATIVE", xyz, nint(sign(1.0d0, dx_sign))

            !$OMP PARALLEL DO schedule(dynamic), &
            !$OMP& PRIVATE(l2dist,self_scalar2_displaced,ksi2_displaced,fourier_displaced)
            do i = 1, na2

                ksi2_displaced(:) = &
                    & get_twobody_weights(x2_displaced(i,:,:,xyz,pm), nneigh2(i), &
                    & two_body_power, cut_start, cut_distance, maxneigh2)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x2_displaced(i,:,:,xyz,pm), &
                    & nneigh2(i), order, three_body_power, cut_start, cut_distance, pmax2, order, maxneigh2)

                self_scalar2_displaced = scalar(x2_displaced(i,:,:,xyz,pm), &
                    & x2_displaced(i,:,:,xyz,pm), nneigh2(i), nneigh2(i), &
                    & ksi2_displaced(:), ksi2_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x2_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh2(i), nneigh1(j), ksi2_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar2_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    do k = 1, nsigmas
                        kernels(k,xyz,i,j) = kernels(k,xyz,i,j) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
            !$OMP END PARALLEL DO
        enddo
    enddo

    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(x2_displaced)

end subroutine fget_atomic_force_kernels_fchl


subroutine fget_scalar_vector_alphas_fchl(x1, forces, energies, nneigh1, &
       & sigmas, lambda, nm1, na1, n1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, alphas)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    use omp_lib, only: omp_get_wtime

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:), intent(in) :: x1
    double precision, dimension(:,:), intent(in) :: forces
    double precision, dimension(:), intent(in) :: energies

    double precision, allocatable, dimension(:,:,:,:,:) :: x1_displaced

    ! Number of neighbors for each atom in each compound
    integer, dimension(:), intent(in) :: nneigh1

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas
    double precision, intent(in) :: lambda

    ! Number of molecules
    integer, intent(in) :: nm1

    ! Number of atoms
    integer, intent(in) :: na1

    ! Number of atoms
    integer, dimension(:), intent(in) :: n1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    double precision, allocatable, dimension(:,:) :: y
    !DEC$ attributes align: 64:: y

    double precision, allocatable, dimension(:,:,:)  :: kernel_delta
    !DEC$ attributes align: 64:: kernel_delta

    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
    !DEC$ attributes align: 64:: kernel_scratch

    ! Internal counters
    integer :: i, j, k
    ! integer :: ni, nj
    integer :: a! , b



    double precision :: force_weight
    double precision :: energy_weight
    double precision, dimension(na1) :: energy_A

    ! Temporary variables necessary for parallelization
    double precision :: l2dist
    ! double precision, allocatable, dimension(:,:) :: atomic_distance

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:) :: self_scalar1
    double precision :: self_scalar1_displaced

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:) :: ksi1
    double precision, allocatable, dimension(:) :: ksi1_displaced

    double precision, allocatable, dimension(:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:) :: fourier_displaced

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: nneighi

    integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: info

    double precision :: ang_norm2

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    double precision :: dx_sign

    integer :: maxneigh1

    integer, dimension(nm1) :: istart, iend
    ! double precision, allocatable, dimension(:,:,:) :: kernel_molecular
    double precision, allocatable, dimension(:,:,:) :: kernel_MA

    double precision :: t_start, t_end

    write (*,*) "INIT"

    maxneigh1 = maxval(nneigh1)
    ang_norm2 = get_angular_norm2(t_width)

    pmax1 = 0
    do a = 1, na1
        pmax1 = max(pmax1, int(maxval(x1(a,2,:nneigh1(a)))))
    enddo

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "DISPLACED REPS"

    dim1 = size(x1, dim=1)
    dim2 = size(x1, dim=2)
    dim3 = size(x1, dim=3)

    allocate(x1_displaced(dim1, dim2, dim3, 3, 2))

    !$OMP PARALLEL DO
    do i = 1, na1
        x1_displaced(i, :, :, :, :) = &
            & get_displaced_representaions(x1(i,:,:), nneigh1(i), dx_numm, dim2, dim3)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KSI1"
    allocate(ksi1(na1, maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO
    do i = 1, na1
        ksi1(i, :) = get_twobody_weights(x1(i,:,:), nneigh1(i), &
            & two_body_power, cut_start, cut_distance, maxneigh1)
    enddo
    !$OMP END PARALLEL do

    write (*,*) "FOURIER"
    allocate(cosp1(na1, pmax1, order, maxneigh1))
    allocate(sinp1(na1, pmax1, order, maxneigh1))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(fourier)
    do i = 1, na1

        fourier = get_threebody_fourier(x1(i,:,:), &
            & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

        cosp1(i,:,:,:) = fourier(1,:,:,:)
        sinp1(i,:,:,:) = fourier(2,:,:,:)

    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(na1))

    self_scalar1 = 0.0d0

    write (*,*) "SELF SCALAR"
    !$OMP PARALLEL DO
    do i = 1, na1
        self_scalar1(i) = scalar(x1(i,:,:), x1(i,:,:), &
            & nneigh1(i), nneigh1(i), ksi1(i,:), ksi1(i,:), &
            & sinp1(i,:,:,:), sinp1(i,:,:,:), &
            & cosp1(i,:,:,:), cosp1(i,:,:,:), &
            & t_width, d_width, cut_distance, order, &
            & pd, ang_norm2,distance_scale, angular_scale, alchemy)
    enddo
    !$OMP END PARALLEL DO

    write (*,*) "ALLOCATE AND CLEAR"
    allocate(kernel_delta(na1,na1,nsigmas))

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    allocate(ksi1_displaced(maxneigh1))
    ksi1_displaced = 0.0d0

    allocate(fourier_displaced(2, pmax1, order, maxneigh1))
    fourier_displaced = 0.0d0

    allocate(y(na1,nsigmas))
    y = 0.0d0

    alphas = 0.0d0

    do xyz = 1, 3

        kernel_delta = 0.0d0


        ! Plus/minus displacemenets
        do pm = 1, 2

            ! Get the sign and magnitude of displacement
            dx_sign = ((dble(pm) - 1.5d0) * 2.0d0) * inv_2dx

            write (*,*) "DERIVATIVE", xyz, nint(sign(1.0d0, dx_sign))

            !$OMP PARALLEL DO schedule(dynamic), &
            !$OMP& PRIVATE(l2dist,self_scalar1_displaced,ksi1_displaced,fourier_displaced)
            do i = 1, na1

                ksi1_displaced(:) = &
                    & get_twobody_weights(x1_displaced(i,:,:,xyz,pm), nneigh1(i), &
                    & two_body_power, cut_start, cut_distance, maxneigh1)

                fourier_displaced(:,:,:,:) = get_threebody_fourier(x1_displaced(i,:,:,xyz,pm), &
                    & nneigh1(i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

                self_scalar1_displaced = scalar(x1_displaced(i,:,:,xyz,pm), &
                    & x1_displaced(i,:,:,xyz,pm), nneigh1(i), nneigh1(i), &
                    & ksi1_displaced(:), ksi1_displaced(:), &
                    & fourier_displaced(2,:,:,:), fourier_displaced(2,:,:,:), &
                    & fourier_displaced(1,:,:,:), fourier_displaced(1,:,:,:), &
                    & t_width, d_width, cut_distance, order, &
                    & pd, ang_norm2,distance_scale, angular_scale, alchemy)

                do j = 1, na1

                    l2dist = scalar(x1_displaced(i,:,:,xyz,pm), x1(j,:,:), &
                        & nneigh1(i), nneigh1(j), ksi1_displaced(:), ksi1(j,:), &
                        & fourier_displaced(2,:,:,:), sinp1(j,:,:,:), &
                        & fourier_displaced(1,:,:,:), cosp1(j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)

                    l2dist = self_scalar1_displaced &
                        & + self_scalar1(j) - 2.0d0 * l2dist

                    do k = 1, nsigmas
                        kernel_delta(i,j,k) = kernel_delta(i,j,k) + &
                            & exp(l2dist * inv_sigma2(k)) * dx_sign
                    enddo

                enddo
            enddo
            !$OMP END PARALLEL DO
        enddo

        do k = 1, nsigmas

            write (*,"(A,F12.4)", advance="no") "     DSYRK()    sigma =", sigmas(k)
            t_start = omp_get_wtime()
            ! DSYRK call corresponds to: C := 1.0 *  K^T * K + 1.0 * C
            call dsyrk("U", "T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & 1.0d0, kernel_scratch(:,:,k), na1)

            t_end = omp_get_wtime()

            write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

            write (*,*) "    DGEMV", sigmas(k)
            ! DGEMV call corresponds to alphas := 1.0 * K^T * F + 1.0 * alphas
            call dgemv("T", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                            & forces(:,xyz), 1, 1.0d0, y(:,k), 1)

        enddo

    enddo

    kernel_delta = 0.0d0

    write (*,*) "NORMAL KERNEL"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(l2dist)
    do j = 1, na1
        do i = j, na1
            l2dist = scalar(x1(i,:,:), x1(j,:,:), &
                & nneigh1(i), nneigh1(j), ksi1(i,:), ksi1(j,:), &
                & sinp1(i,:,:,:), sinp1(j,:,:,:), &
                & cosp1(i,:,:,:), cosp1(j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(i) + self_scalar1(j) - 2.0d0 * l2dist

            do k = 1, nsigmas
                ! kernel_delta(i,j,k) = kernel_delta(i,j,k) + sqrt(l2dist + 0.1)
                kernel_delta(i, j, k) =  exp(l2dist * inv_sigma2(k))
                kernel_delta(j, i, k) =  kernel_delta(i, j, k)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    istart(:) = 0
    iend(:) = 0

    istart(1) = 1
    iend(1) = n1(1)



    do i = 2, nm1
        istart(i) = istart(i-1) + n1(i)
        iend(i) = iend(i-1) + n1(i)

    enddo

    write (*,*) "HERE1"
    do i = 1, nm1
        energy_A(istart(i):iend(i)) = energies(i)! / n1(i)
    enddo
    write (*,*) "HERE1"

    deallocate(self_scalar1)
    deallocate(cosp1)
    deallocate(sinp1)
    deallocate(ksi1)
    deallocate(ksi1_displaced)
    deallocate(fourier_displaced)
    deallocate(x1_displaced)

    write (*,*) "HERE1"
    allocate(kernel_MA(nm1,na1,nsigmas))

    write (*,*) "HERE1"
    do k = 1, nsigmas
        do j = 1, nm1

            do i = 1, na1

                l2dist = sum(kernel_delta(i, istart(j):iend(j), k))
                kernel_MA(j, i, k) = l2dist

            enddo
        enddo
    enddo

    write (*,*) "HERE1"

    ! write(*,*) "Y"
    ! do k = 1, nsigmas
    !     write(*,*) y(:5,k)
    ! enddo

    do k = 1, nsigmas
        write (*,*) "Kernel", k, sigmas(k)
        do i = 1, 10
            write(*,*) kernel_delta(:10,i,k)
        enddo
    enddo

    force_weight  = 1.0d0
    energy_weight = 0.0d0

    do k = 1, nsigmas

        kernel_scratch(:,:,k) = kernel_scratch(:,:,k) * force_weight + kernel_delta(:,:,k) * energy_weight
        y(:,k) = 1.0d0 * y(:,k) * force_weight + energy_A(:) * energy_weight

        ! write (*,*) "    DGEMM", sigmas(k)

        ! DGEMM call corresponds to: C := w_E*K^T*K + w_F*C
        ! call dgemm("t", "n", na1, na1, nm1, energy_weight, kernel_MA(:,:,k), nm1, &
        !   & kernel_MA(:,:,k), nm1, force_weight, kernel_scratch(:,:,k), na1)

        ! write (*,*) "    DGEMV", sigmas(k)

        ! ! DGEMV call corresponds to: Y := w_E*K^T*E  + w_F*Y
        ! call dgemv("T", nm1, na1, energy_weight, kernel_ma(:,:,k), nm1, &
        !               & energies(:), 1, force_weight, y(:,k), 1)

    enddo

    ! write(*,*) "Y"
    ! do k = 1, nsigmas
    !     write(*,*) y(:5,k)
    ! enddo

    deallocate(kernel_delta)


    write(*,*) "LLAMBDA", lambda

    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + lambda
        enddo
    enddo

    do k = 1, nsigmas
        write (*,*) "  DPOTRF", sigmas(k)
        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        write (*,*) "  DPOTRS", sigmas(k)
        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        alphas(k,:) = y(:,k)
    enddo

    deallocate(kernel_scratch)
    deallocate(y)

end subroutine fget_scalar_vector_alphas_fchl


subroutine fget_local_force_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, naq1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: naq1
    integer, intent(in) :: naq2
    
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power
    
    double precision, intent(in) :: dx

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,naq1,naq2), intent(out) :: kernels
    ! double precision, allocatable, dimension(:,:,:,:)  :: l2_displaced

    ! Internal counters
    integer :: i, j, k, i1, i2, j1, j2
    integer :: na, nb ! , ni, nj
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    integer :: xyz_pm
    integer :: xyz_pm1
    integer :: xyz_pm2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    integer :: idx1, idx2
    ! integer :: nneighi

    ! integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: xyz1, pm1
    integer :: xyz2, pm2

    double precision :: ang_norm2
    ! double precision :: l2_dist

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    ! double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 10
    pmax2 = 10

    write (*,*) "KSI1", nm1,3,2,maxval(n1),maxval(n1),maxneigh1
    allocate(ksi1(nm1,3,2,maxval(n1),maxval(n1),maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
                        ksi1(a, xyz, pm, i, j, :) = get_twobody_weights( &
                & x1(a,xyz,pm,i,j,:,:), nneigh1(a, xyz, pm, i, j), &
                & two_body_power, cut_start, cut_distance, maxneigh1)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do


    write (*,*) "KSI2",nm2,3,2,maxval(n2),maxval(n2),maxneigh2
    allocate(ksi2(nm2,3,2,maxval(n2),maxval(n2),maxneigh2))

    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
                        ksi2(a, xyz, pm, i, j, :) = get_twobody_weights( &
                & x2(a,xyz,pm,i,j,:,:), nneigh2(a, xyz, pm, i, j), &
                & two_body_power, cut_start, cut_distance, maxneigh2)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    allocate(cosp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    
    !$OMP PARALLEL DO PRIVATE(na, fourier, xyz_pm) schedule(dynamic)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na

                xyz_pm = 2*xyz + pm - 2

                fourier = get_threebody_fourier(x1(a,xyz,pm,i,j,:,:), &
                    & nneigh1(a, xyz, pm, i, j), &
                    & order, three_body_power, cut_start, cut_distance, &
                    & pmax1, order, maxneigh1)

            cosp1(a,xyz_pm,i,j,:,:,:) = fourier(1,:,:,:)
            sinp1(a,xyz_pm,i,j,:,:,:) = fourier(2,:,:,:)

            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    
    !$OMP PARALLEL DO PRIVATE(na, fourier, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na

                xyz_pm = 2*xyz + pm - 2

                fourier = get_threebody_fourier(x2(a,xyz,pm,i,j,:,:), &
                    & nneigh2(a, xyz, pm, i, j), &
                    & order, three_body_power, cut_start, cut_distance, &
                    & pmax2, order, maxneigh2)

            cosp2(a,xyz_pm,i,j,:,:,:) = fourier(1,:,:,:)
            sinp2(a,xyz_pm,i,j,:,:,:) = fourier(2,:,:,:)

            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(nm1, 3, 2, maxval(n1), maxval(n1)))
    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na, xyz_pm) schedule(dynamic)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na
                
            xyz_pm = 2*xyz + pm - 2

            self_scalar1(a,xyz,pm,i,j) = scalar(x1(a,xyz,pm,i,j,:,:), x1(a,xyz,pm,i,j,:,:), &
                & nneigh1(a,xyz,pm,i,j), nneigh1(a,xyz,pm,i,j), &
                & ksi1(a,xyz,pm,i,j,:), ksi1(a,xyz,pm,i,j,:), &
                & sinp1(a,xyz_pm,i,j,:,:,:), sinp1(a,xyz_pm,i,j,:,:,:), &
                & cosp1(a,xyz_pm,i,j,:,:,:), cosp1(a,xyz_pm,i,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    write (*,*) "SELF SCALAR"
    allocate(self_scalar2(nm2, 3, 2, maxval(n2), maxval(n2)))
    self_scalar2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na
                
            xyz_pm = 2*xyz + pm - 2

            self_scalar2(a,xyz,pm,i,j) = scalar(x2(a,xyz,pm,i,j,:,:), x2(a,xyz,pm,i,j,:,:), &
                & nneigh2(a,xyz,pm,i,j), nneigh2(a,xyz,pm,i,j), &
                & ksi2(a,xyz,pm,i,j,:), ksi2(a,xyz,pm,i,j,:), &
                & sinp2(a,xyz_pm,i,j,:,:,:), sinp2(a,xyz_pm,i,j,:,:,:), &
                & cosp2(a,xyz_pm,i,j,:,:,:), cosp2(a,xyz_pm,i,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do


    write (*,*) "KERNEL"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            l2dist = scalar(x1(a,xyz1,pm1,i1,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,xyz1,pm1,i1,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,xyz1,pm1,i1,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,xyz_pm1,i1,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,xyz_pm1,i1,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(a,xyz1,pm1,i1,j1) + self_scalar2(b,xyz2,pm2,i2,j2) &
                & - 2.0d0 * l2dist

            if (pm1 == pm2) then

                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & + exp(inv_sigma2(k)  * l2dist)
                enddo
            else
                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & - exp(inv_sigma2(k)  * l2dist)
                enddo

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (4 * dx**2)

end subroutine fget_local_force_kernels_fchl


subroutine fget_local_symmetric_force_kernels_fchl(x1, n1, nneigh1, &
       & sigmas, nm1, naq1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x1

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh1
    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: naq1
    
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power
    
    double precision, intent(in) :: dx

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,naq1,naq1), intent(out) :: kernels
    ! double precision, allocatable, dimension(:,:,:,:)  :: l2_displaced

    ! Internal counters
    integer :: i, j, k, i1, i2, j1, j2
    integer :: na, nb!, ni, nj
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar1
    ! double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi1
    !double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp1

    !double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    !double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    integer :: xyz_pm
    integer :: xyz_pm1
    integer :: xyz_pm2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    ! integer :: pmax2
    integer :: idx1, idx2
    ! integer :: nneighi

    ! integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: xyz1, pm1
    integer :: xyz2, pm2

    double precision :: ang_norm2
    ! double precision :: l2_dist

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    ! double precision :: dx_sign

    integer :: maxneigh1
    ! integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)

    pmax1 = 10

    write (*,*) "KSI1", nm1,3,2,maxval(n1),maxval(n1),maxneigh1
    allocate(ksi1(nm1,3,2,maxval(n1),maxval(n1),maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
                        ksi1(a, xyz, pm, i, j, :) = get_twobody_weights( &
                & x1(a,xyz,pm,i,j,:,:), nneigh1(a, xyz, pm, i, j), &
                & two_body_power, cut_start, cut_distance, maxneigh1)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    allocate(cosp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)))
    
    !$OMP PARALLEL DO PRIVATE(na, fourier, xyz_pm) schedule(dynamic)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na

                xyz_pm = 2*xyz + pm - 2

                fourier = get_threebody_fourier(x1(a,xyz,pm,i,j,:,:), &
                    & nneigh1(a, xyz, pm, i, j), &
                    & order, three_body_power, cut_start, cut_distance, &
                    & pmax1, order, maxneigh1)

            cosp1(a,xyz_pm,i,j,:,:,:) = fourier(1,:,:,:)
            sinp1(a,xyz_pm,i,j,:,:,:) = fourier(2,:,:,:)

            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(nm1, 3, 2, maxval(n1), maxval(n1)))
    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na, xyz_pm) schedule(dynamic)
    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na
                
            xyz_pm = 2*xyz + pm - 2

            self_scalar1(a,xyz,pm,i,j) = scalar(x1(a,xyz,pm,i,j,:,:), x1(a,xyz,pm,i,j,:,:), &
                & nneigh1(a,xyz,pm,i,j), nneigh1(a,xyz,pm,i,j), &
                & ksi1(a,xyz,pm,i,j,:), ksi1(a,xyz,pm,i,j,:), &
                & sinp1(a,xyz_pm,i,j,:,:,:), sinp1(a,xyz_pm,i,j,:,:,:), &
                & cosp1(a,xyz_pm,i,j,:,:,:), cosp1(a,xyz_pm,i,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KERNEL"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    do j1 = 1, na
            
        do b = a, nm1
        nb = n1(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n1(:b)) - n1(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            l2dist = scalar(x1(a,xyz1,pm1,i1,j1,:,:), x1(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,xyz1,pm1,i1,j1), nneigh1(b,xyz2,pm2,i2,j2), &
                & ksi1(a,xyz1,pm1,i1,j1,:), ksi1(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,xyz_pm1,i1,j1,:,:,:), sinp1(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,xyz_pm1,i1,j1,:,:,:), cosp1(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(a,xyz1,pm1,i1,j1) + self_scalar1(b,xyz2,pm2,i2,j2) &
                & - 2.0d0 * l2dist

            

            if (pm1 == pm2) then

                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & + exp(inv_sigma2(k)  * l2dist)
                    if (a /= b) then
                        kernels(k,idx2,idx1) = kernels(k,idx2,idx1) & 
                            & + exp(inv_sigma2(k)  * l2dist)
                    endif
                enddo
            else
                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & - exp(inv_sigma2(k)  * l2dist)
                
                    if (a /= b) then
                        kernels(k,idx2,idx1) = kernels(k,idx2,idx1) & 
                            & - exp(inv_sigma2(k)  * l2dist)
                    endif
                enddo

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (4 * dx**2)

end subroutine fget_local_symmetric_force_kernels_fchl


subroutine fget_local_gradient_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: naq2
    
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power
    
    double precision, intent(in) :: dx

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting alpha vector
    double precision, dimension(nsigmas,nm1,naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, i2, j1, j2
    integer :: na, nb, ni!, nj
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    integer :: xyz_pm
    ! integer :: xyz_pm1
    integer :: xyz_pm2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    integer :: idx1, idx2
    ! integer :: nneighi

    ! integer :: dim1, dim2, dim3
    integer :: xyz, pm
    ! integer :: xyz1, pm1
    integer :: xyz2, pm2

    double precision :: ang_norm2
    ! double precision :: l2_dist

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    ! double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 10
    pmax2 = 10

    write (*,*) "KSI1", nm1,maxval(n1),maxneigh1
    allocate(ksi1(nm1, maxval(n1), maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
                & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do


    write (*,*) "KSI2",nm2,3,2,maxval(n2),maxval(n2),maxneigh2
    allocate(ksi2(nm2,3,2,maxval(n2),maxval(n2),maxneigh2))

    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
                        ksi2(a, xyz, pm, i, j, :) = get_twobody_weights( &
                & x2(a,xyz,pm,i,j,:,:), nneigh2(a, xyz, pm, i, j), &
                & two_body_power, cut_start, cut_distance, maxneigh2)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    
    !$OMP PARALLEL DO PRIVATE(na, fourier, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na

                xyz_pm = 2*xyz + pm - 2

                fourier = get_threebody_fourier(x2(a,xyz,pm,i,j,:,:), &
                    & nneigh2(a, xyz, pm, i, j), &
                    & order, three_body_power, cut_start, cut_distance, &
                    & pmax2, order, maxneigh2)

            cosp2(a,xyz_pm,i,j,:,:,:) = fourier(1,:,:,:)
            sinp2(a,xyz_pm,i,j,:,:,:) = fourier(2,:,:,:)

            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(nm1, maxval(n1)))
    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            self_scalar1(a,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
        enddo
    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar2(nm2, 3, 2, maxval(n2), maxval(n2)))
    self_scalar2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na
                
            xyz_pm = 2*xyz + pm - 2

            self_scalar2(a,xyz,pm,i,j) = scalar(x2(a,xyz,pm,i,j,:,:), x2(a,xyz,pm,i,j,:,:), &
                & nneigh2(a,xyz,pm,i,j), nneigh2(a,xyz,pm,i,j), &
                & ksi2(a,xyz,pm,i,j,:), ksi2(a,xyz,pm,i,j,:), &
                & sinp2(a,xyz_pm,i,j,:,:,:), sinp2(a,xyz_pm,i,j,:,:,:), &
                & cosp2(a,xyz_pm,i,j,:,:,:), cosp2(a,xyz_pm,i,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KERNEL"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    ! do xyz1 = 1, 3
    ! do pm1 = 1, 2
    ! xyz_pm1 = 2*xyz1 + pm1 - 2
    ! do i1 = 1, na
    ! idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    idx1 = a
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2
        do j2 = 1, nb


            l2dist = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(a,j1) + self_scalar2(b,xyz2,pm2,i2,j2) &
                & - 2.0d0 * l2dist

            if (pm2 == 2) then

                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & + exp(inv_sigma2(k)  * l2dist)
                enddo
            else
                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & - exp(inv_sigma2(k)  * l2dist)
                enddo

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    ! enddo
    ! enddo
    ! enddo
    enddo
    !$OMP END PARALLEL do

    kernels = kernels / (2 * dx)

end subroutine fget_local_gradient_kernels_fchl



subroutine fget_local_full_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2, dx_numm

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: naq2
    
    
    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: n1
    integer, dimension(:), intent(in) :: n2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    double precision, intent(in) :: two_body_power
    double precision, intent(in) :: three_body_power
    
    double precision, intent(in) :: dx

    double precision, intent(in) :: t_width
    double precision, intent(in) :: d_width
    double precision, intent(in) :: cut_start
    double precision, intent(in) :: cut_distance
    integer, intent(in) :: order
    double precision, intent(in) :: distance_scale
    double precision, intent(in) :: angular_scale
    logical, intent(in) :: alchemy

    ! -1.0 / sigma^2 for use in the kernel
    double precision, dimension(nsigmas) :: inv_sigma2

    double precision, dimension(:,:), intent(in) :: pd

    ! Resulting kernel matrix
    double precision, dimension(nsigmas,nm1+naq2,nm1+naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, i1, i2, j1, j2
    integer :: na, nb, ni, nj
    integer :: a, b

    ! Temporary variables necessary for parallelization
    double precision :: l2dist

    ! Pre-computed terms in the full distance matrix
    double precision, allocatable, dimension(:,:) :: self_scalar1
    double precision, allocatable, dimension(:,:,:,:,:) :: self_scalar2

    ! Pre-computed terms
    double precision, allocatable, dimension(:,:,:) :: ksi1
    double precision, allocatable, dimension(:,:,:,:,:,:) :: ksi2

    double precision, allocatable, dimension(:,:,:,:,:) :: sinp1
    double precision, allocatable, dimension(:,:,:,:,:) :: cosp1

    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: sinp2
    double precision, allocatable, dimension(:,:,:,:,:,:,:) :: cosp2

    integer :: xyz_pm
    integer :: xyz_pm1
    integer :: xyz_pm2

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    integer :: idx1, idx2
    ! integer :: nneighi

    ! integer :: dim1, dim2, dim3
    integer :: xyz, pm
    integer :: xyz1, pm1
    integer :: xyz2, pm2

    double precision :: ang_norm2
    ! double precision :: l2_dist

    double precision, parameter :: inv_2dx = 1.0d0 / (2.0d0 * dx_numm)
    ! double precision :: dx_sign

    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 10
    pmax2 = 10

    write (*,*) "KSI1", nm1,maxval(n1),maxneigh1
    allocate(ksi1(nm1, maxval(n1), maxneigh1))

    ksi1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            ksi1(a, i, :) = get_twobody_weights(x1(a,i,:,:), nneigh1(a, i), &
                & two_body_power, cut_start, cut_distance, maxneigh1)
        enddo
    enddo
    !$OMP END PARALLEL do


    write (*,*) "KSI2",nm2,3,2,maxval(n2),maxval(n2),maxneigh2
    allocate(ksi2(nm2,3,2,maxval(n2),maxval(n2),maxneigh2))

    ksi2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
                        ksi2(a, xyz, pm, i, j, :) = get_twobody_weights( &
                & x2(a,xyz,pm,i,j,:,:), nneigh2(a, xyz, pm, i, j), &
                & two_body_power, cut_start, cut_distance, maxneigh2)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    
    allocate(cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
    allocate(sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

    cosp1 = 0.0d0
    sinp1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni

            fourier = get_threebody_fourier(x1(a,i,:,:), &
                & nneigh1(a, i), order, three_body_power, cut_start, cut_distance, pmax1, order, maxneigh1)

            cosp1(a,i,:,:,:) = fourier(1,:,:,:)
            sinp1(a,i,:,:,:) = fourier(2,:,:,:)

        enddo
    enddo
    !$OMP END PARALLEL DO

    
    write (*,*) "ALLOCATE KSI3", &
        & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    
    !$OMP PARALLEL DO PRIVATE(na, fourier, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na

                xyz_pm = 2*xyz + pm - 2

                fourier = get_threebody_fourier(x2(a,xyz,pm,i,j,:,:), &
                    & nneigh2(a, xyz, pm, i, j), &
                    & order, three_body_power, cut_start, cut_distance, &
                    & pmax2, order, maxneigh2)

            cosp2(a,xyz_pm,i,j,:,:,:) = fourier(1,:,:,:)
            sinp2(a,xyz_pm,i,j,:,:,:) = fourier(2,:,:,:)

            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do
    
    write (*,*) "SELF SCALAR"
    allocate(self_scalar1(nm1, maxval(n1)))
    self_scalar1 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(ni)
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            self_scalar1(a,i) = scalar(x1(a,i,:,:), x1(a,i,:,:), &
                & nneigh1(a,i), nneigh1(a,i), ksi1(a,i,:), ksi1(a,i,:), &
                & sinp1(a,i,:,:,:), sinp1(a,i,:,:,:), &
                & cosp1(a,i,:,:,:), cosp1(a,i,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
        enddo
    enddo
    !$OMP END PARALLEL DO

    write (*,*) "SELF SCALAR"
    allocate(self_scalar2(nm2, 3, 2, maxval(n2), maxval(n2)))
    self_scalar2 = 0.0d0

    !$OMP PARALLEL DO PRIVATE(na, xyz_pm) schedule(dynamic)
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
        do pm = 1, 2
        do i = 1, na
            do j = 1, na
                
            xyz_pm = 2*xyz + pm - 2

            self_scalar2(a,xyz,pm,i,j) = scalar(x2(a,xyz,pm,i,j,:,:), x2(a,xyz,pm,i,j,:,:), &
                & nneigh2(a,xyz,pm,i,j), nneigh2(a,xyz,pm,i,j), &
                & ksi2(a,xyz,pm,i,j,:), ksi2(a,xyz,pm,i,j,:), &
                & sinp2(a,xyz_pm,i,j,:,:,:), sinp2(a,xyz_pm,i,j,:,:,:), &
                & cosp2(a,xyz_pm,i,j,:,:,:), cosp2(a,xyz_pm,i,j,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)
            enddo
        enddo
        enddo
        enddo
    enddo
    !$OMP END PARALLEL do

    write (*,*) "KERNEL EVALUATION"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,l2dist)
    do a = 1, nm1
    na = n1(a)
    do j1 = 1, na
            
        do b = 1, nm1
        nb = n1(b)
        do j2 = 1, nb


            l2dist = scalar(x1(a,j1,:,:), x1(b,j2,:,:), &
                & nneigh1(a,j1), nneigh1(b,j2), &
                & ksi1(a,j1,:), ksi1(b,j2,:), &
                & sinp1(a,j1,:,:,:), sinp1(b,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp1(b,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(a,j1) + self_scalar1(b,j2) &
                & - 2.0d0 * l2dist

            do k = 1, nsigmas
                kernels(k,a,b) = kernels(k,a,b) & 
                    & + exp(inv_sigma2(k)  * l2dist)
            enddo

        enddo
        enddo
    enddo
    enddo
    !$OMP END PARALLEL do







    write (*,*) "KERNEL GRADIENT"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    ! do xyz1 = 1, 3
    ! do pm1 = 1, 2
    ! xyz_pm1 = 2*xyz1 + pm1 - 2
    ! do i1 = 1, na
    ! idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1
    idx1 = a
    do j1 = 1, na
            
        do b = 1, nm2
        nb = n2(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n2(:b)) - n2(b))* 3 + (i2 - 1) * 3  + xyz2 + nm1
        do j2 = 1, nb


            l2dist = scalar(x1(a,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh1(a,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi1(a,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp1(a,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp1(a,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar1(a,j1) + self_scalar2(b,xyz2,pm2,i2,j2) &
                & - 2.0d0 * l2dist

            if (pm2 == 2) then

                do k = 1, nsigmas

                   kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                       & + exp(inv_sigma2(k)  * l2dist)
                
                   kernels(k,idx2,idx1) = kernels(k,idx1,idx2)

                enddo
            else
                do k = 1, nsigmas

                    kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                         - exp(inv_sigma2(k)  * l2dist)
                
                    kernels(k,idx2,idx1) = kernels(k,idx1,idx2)

                enddo

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    ! enddo
    ! enddo
    ! enddo
    enddo
    !$OMP END PARALLEL do

    kernels(:,:nm1,nm1+1:) = kernels(:,:nm1,nm1+1:) / (2 * dx)
    kernels(:,nm1+1:,:nm1) = kernels(:,nm1+1:,:nm1) / (2 * dx)
    
    write (*,*) "KERNEL HESSIAN"
    
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm1,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2)
    do a = 1, nm1
    na = n1(a)
    do xyz1 = 1, 3
    do pm1 = 1, 2
    xyz_pm1 = 2*xyz1 + pm1 - 2
    do i1 = 1, na
    idx1 = (sum(n1(:a)) - n1(a))* 3 + (i1 - 1) * 3  + xyz1 + nm1
    do j1 = 1, na
            
        do b = a, nm1
        nb = n1(b)
        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb
        idx2 = (sum(n1(:b)) - n1(b))* 3 + (i2 - 1) * 3  + xyz2 + nm1
        do j2 = 1, nb


            l2dist = scalar(x2(a,xyz1,pm1,i1,j1,:,:), x2(b,xyz2,pm2,i2,j2,:,:), &
                & nneigh2(a,xyz1,pm1,i1,j1), nneigh2(b,xyz2,pm2,i2,j2), &
                & ksi2(a,xyz1,pm1,i1,j1,:), ksi2(b,xyz2,pm2,i2,j2,:), &
                & sinp2(a,xyz_pm1,i1,j1,:,:,:), sinp2(b,xyz_pm2,i2,j2,:,:,:), &
                & cosp2(a,xyz_pm1,i1,j1,:,:,:), cosp2(b,xyz_pm2,i2,j2,:,:,:), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2,distance_scale, angular_scale, alchemy)

            l2dist = self_scalar2(a,xyz1,pm1,i1,j1) + self_scalar2(b,xyz2,pm2,i2,j2) &
                & - 2.0d0 * l2dist

            

            if (pm1 == pm2) then

                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & + exp(inv_sigma2(k)  * l2dist)
                    if (a /= b) then
                        kernels(k,idx2,idx1) = kernels(k,idx2,idx1) & 
                            & + exp(inv_sigma2(k)  * l2dist)
                    endif
                enddo
            else
                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                    & - exp(inv_sigma2(k)  * l2dist)
                
                    if (a /= b) then
                        kernels(k,idx2,idx1) = kernels(k,idx2,idx1) & 
                            & - exp(inv_sigma2(k)  * l2dist)
                    endif
                enddo

            end if


        enddo
        enddo
        enddo
        enddo
        enddo
    enddo
    enddo
    enddo
    enddo
    enddo
    !$OMP END PARALLEL do

    kernels(:,nm1+1:,nm1+1:) = kernels(:,nm1+1:,nm1+1:) / (4 * dx**2)

end subroutine fget_local_full_kernels_fchl
