subroutine fget_local_hessian_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, naq1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

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

    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 0 

    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax1 = max(pmax1, int(maxval(x1(a,xyz,pm,i,j,2,:nneigh1(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

    write (*,*) "PMAX =", pmax1

    pmax2 = 0 

    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax2 = max(pmax2, int(maxval(x2(a,xyz,pm,i,j,2,:nneigh2(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

    write (*,*) "PMAX =", pmax2

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

end subroutine fget_local_hessian_kernels_fchl


subroutine fget_local_symmetric_hessian_kernels_fchl(x1, n1, nneigh1, &
       & sigmas, nm1, naq1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

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

    integer :: maxneigh1
    ! integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)

    pmax1 = 0 

    do a = 1, nm1
        na = n1(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax1 = max(pmax1, int(maxval(x1(a,xyz,pm,i,j,2,:nneigh1(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

    write (*,*) "PMAX =", pmax1

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

end subroutine fget_local_symmetric_hessian_kernels_fchl


subroutine fget_local_gradient_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

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


    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT", dx

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 0
    pmax2 = 0

    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            pmax1 = max(pmax1, int(maxval(x1(a,i,2,:nneigh1(a,i)))))
        end do
    end do
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax2 = max(pmax2, int(maxval(x2(a,xyz,pm,i,j,2,:nneigh2(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

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
                        & get_displaced_representaions, get_angular_norm2

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
    integer :: na, nb, ni
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

    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT"

    write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 0
    pmax2 = 0
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            pmax1 = max(pmax1, int(maxval(x1(a,i,2,:nneigh1(a,i)))))
        end do
    end do
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax2 = max(pmax2, int(maxval(x2(a,xyz,pm,i,j,2,:nneigh2(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

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


subroutine fget_atomic_gradient_kernels_fchl(x1, x2, n1, n2, nneigh1, nneigh2, &
       & sigmas, nm1, nm2, na1, naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, kernels)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

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
    
    integer, intent(in) :: na1
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
    double precision, dimension(nsigmas,na1,naq2), intent(out) :: kernels

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
    integer :: idx1_start,idx1_end
    integer :: idx2_start,idx2_end
    ! integer :: nneighi

    ! integer :: dim1, dim2, dim3
    integer :: xyz, pm
    ! integer :: xyz1, pm1
    integer :: xyz2, pm2

    double precision :: ang_norm2
    ! double precision :: l2_dist


    integer :: maxneigh1
    integer :: maxneigh2

    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    ! write (*,*) "INIT  DX =", dx

    ! write (*,*) "CLEARING KERNEL MEM"
    kernels = 0.0d0

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 0
    pmax2 = 0
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            pmax1 = max(pmax1, int(maxval(x1(a,i,2,:nneigh1(a,i)))))
        end do
    end do
    do a = 1, nm2
        na = n2(a)
        do xyz = 1, 3
            do pm = 1, 2
                do i = 1, na
                    do j = 1, na
            pmax2 = max(pmax2, int(maxval(x2(a,xyz,pm,i,j,2,:nneigh2(a, xyz, pm, i, j)))))
                    end do
                end do
            end do
        end do
    end do

    ! write (*,*) "KSI2 1"!, nm1,maxval(n1),maxneigh1
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


    ! write (*,*) "KSI2 2"!,nm2,3,2,maxval(n2),maxval(n2),maxneigh2
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
    
    ! write (*,*) "KSI3 1"!, &
        ! & nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
    
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

    
    ! write (*,*) "KSI3 2"!, &
        !& nm1, 3*2, maxval(n1), maxval(n1), pmax1, order, maxval(nneigh1)
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
    
    ! write (*,*) "SELF SCALAR 1"
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

    ! write (*,*) "SELF SCALAR 2"
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

    ! write (*,*) "KERNEL"
    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,l2dist),&
    !$OMP& PRIVATE(idx1,idx2,idx1_start,idx1_end,idx2_start,idx2_end)
    do a = 1, nm1
    na = n1(a)
    
    idx1_end = sum(n1(:a))
    idx1_start = idx1_end - na + 1 
    
    do j1 = 1, na
        idx1 = idx1_start - 1 + j1
            
        do b = 1, nm2
        nb = n2(b)

        ! idx2_start = sum(n2(:b)) - n2(b) + 1 
        ! idx2_end = sum(n2(:b))

        idx2_end = sum(n2(:b))
        idx2_start = idx2_end - nb + 1 

        do xyz2 = 1, 3
        do pm2 = 1, 2
        xyz_pm2 = 2*xyz2 + pm2 - 2
        do i2 = 1, nb

        idx2 = (idx2_start-1)*3 + (i2-1)*3 + xyz2

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
                ! kernels(k,idx1_start:idx1_end,idx2) = kernels(k,idx1_start:idx1_end,idx2) & 
                    & + exp(inv_sigma2(k)  * l2dist)
                enddo
            else
                do k = 1, nsigmas
                kernels(k,idx1,idx2) = kernels(k,idx1,idx2) & 
                ! kernels(k,idx1_start:idx1_end,idx2) = kernels(k,idx1_start:idx1_end,idx2) & 
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

end subroutine fget_atomic_gradient_kernels_fchl


subroutine fget_local_invariant_alphas_fchl(x1, x2, forces, energies, n1, n2, &
        & nneigh1, nneigh2, sigmas, nm1, nm2, na1, nsigmas, &
        & t_width, d_width, cut_start, cut_distance, order, pd, &
        & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
        & llambda, alphas)

    use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
                        & get_displaced_representaions, get_angular_norm2

    use omp_lib, only: omp_get_wtime

    implicit none

    double precision, allocatable, dimension(:,:,:,:) :: fourier

    ! fchl descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
    double precision, dimension(:,:,:,:), intent(in) :: x1
    double precision, dimension(:,:,:,:,:,:,:), intent(in) :: x2

    ! Number of neighbors for each atom in each compound
    integer, dimension(:,:), intent(in) :: nneigh1
    integer, dimension(:,:,:,:,:), intent(in) :: nneigh2

    double precision, dimension(:,:), intent(in) :: forces
    double precision, dimension(:), intent(in) :: energies

    ! Sigma in the Gaussian kernel
    double precision, dimension(:), intent(in) :: sigmas

    ! Regularization Lambda
    double precision, intent(in) :: llambda

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2
    
    integer, intent(in) :: na1
    ! integer, intent(in) :: naq2
    
    
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
    ! double precision, dimension(nsigmas,nm1+naq2,nm1+naq2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, i2, j1, j2
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
    
    double precision, dimension(nsigmas,na1), intent(out) :: alphas

    integer :: xyz_pm
    ! integer :: xyz_pm1
    integer :: xyz_pm2

    integer :: info

    ! Value of PI at full FORTRAN precision.
    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! counter for periodic distance
    integer :: pmax1
    integer :: pmax2
    integer :: idx1
    integer :: idx2
    integer :: idx1_start
    integer :: idx2_start
    
    integer :: xyz, pm
    integer :: xyz2, pm2

    double precision :: ang_norm2
    double precision :: inv_2dx

    integer :: maxneigh1
    integer :: maxneigh2
    
    double precision, allocatable, dimension(:,:) :: y
    
    double precision, allocatable, dimension(:,:,:)  :: kernel_delta

    double precision, allocatable, dimension(:,:,:)  :: kernel_scratch
    
    double precision :: t_start, t_end
    
    double precision, allocatable, dimension(:,:,:) :: kernel_ma

    inv_2dx = 1.0d0 / (2.0d0 * dx)
    ang_norm2 = get_angular_norm2(t_width)
    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2

    write (*,*) "INIT, DX =", dx

    maxneigh1 = maxval(nneigh1)
    maxneigh2 = maxval(nneigh2)

    pmax1 = 0
    do a = 1, nm1
        ni = n1(a)
        do i = 1, ni
            pmax1 = max(pmax1, int(maxval(x1(a,i,2,:nneigh1(a,i)))))
        end do
    end do

    pmax2 = pmax1

    write (*,"(A)", advance="no") "TWO-BODY TERMS"
    t_start = omp_get_wtime()

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

    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                          Time = ", t_end - t_start, " s"

    write (*,"(A)", advance="no") "TWO-BODY GRADIENT"
    t_start = omp_get_wtime()

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
    
    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                       Time = ", t_end - t_start, " s"

    write (*,"(A)", advance="no") "THREE-BODY TERMS"
    t_start = omp_get_wtime()
    
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

    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                        Time = ", t_end - t_start, " s"

    write (*,"(A)", advance="no") "THREE-BODY GRADIENT"
    t_start = omp_get_wtime()
    
    allocate(cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    allocate(sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxval(nneigh2)))
    cosp2 = 0.0d0
    sinp2 = 0.0d0
    
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
    
    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                     Time = ", t_end - t_start, " s"
    
    write (*,"(A)", advance="no") "SELF-SCALAR TERMS"
    t_start = omp_get_wtime()

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
    
    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                       Time = ", t_end - t_start, " s"

    write (*,"(A)", advance="no") "SELF-SCALAR GRADIENT"
    t_start = omp_get_wtime()
    
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
    
    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                    Time = ", t_end - t_start, " s"

    allocate(kernel_delta(na1,na1,nsigmas))
    allocate(y(na1,nsigmas))
    y = 0.0d0

    allocate(kernel_scratch(na1,na1,nsigmas))
    kernel_scratch = 0.0d0

    do xyz2 = 1, 3
        
        write (*,"(A,I3,A)", advance="no") "KERNEL GRADIENT", xyz2, " / 3"
        t_start = omp_get_wtime()
        
        kernel_delta = 0.0d0

        !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,l2dist), &
        !$OMP& PRIVATE(idx1,idx2,idx1_start,idx2_start)
        do a = 1, nm1
        na = n1(a)
        idx1_start = sum(n1(:a)) - na
        do j1 = 1, na
        idx1 = idx1_start + j1
                
            do b = 1, nm2
            nb = n2(b)
            idx2_start = (sum(n2(:b)) - nb)

            do pm2 = 1, 2
            xyz_pm2 = 2*xyz2 + pm2 - 2
            do i2 = 1, nb
            idx2 = idx2_start + i2
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

                       kernel_delta(idx1,idx2,k) = kernel_delta(idx1,idx2,k) & 
                           & + exp(inv_sigma2(k)  * l2dist) * inv_2dx

                    enddo
                else
                    do k = 1, nsigmas
                       kernel_delta(idx1,idx2,k) = kernel_delta(idx1,idx2,k) & 
                           & - exp(inv_sigma2(k)  * l2dist) * inv_2dx

                    enddo

                end if


            enddo
            enddo
            enddo
            enddo
            enddo
        enddo
        !$OMP END PARALLEL do
        
        t_end = omp_get_wtime()
        write (*,"(A,F12.4,A)") "                  Time = ", t_end - t_start, " s"

        do k = 1, nsigmas

            write (*,"(A,F12.4)", advance="no") "     DSYRK()    sigma =", sigmas(k)
            t_start = omp_get_wtime()
            
            call dsyrk("U", "N", na1, na1, 1.0d0, kernel_delta(1,1,k), na1, &
               & 1.0d0, kernel_scratch(1,1,k), na1)

            ! kernel_scratch(:,:,k) = kernel_scratch(:,:,k) &
            !    & + matmul(kernel_delta(:,:,k),transpose(kernel_delta(:,:,k)))! * inv_2dx*inv_2dx

            t_end = omp_get_wtime()
            write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

            write (*,"(A,F12.4)", advance="no") "     DGEMV()    sigma =", sigmas(k)
            t_start = omp_get_wtime()
            
            call dgemv("N", na1, na1, 1.0d0, kernel_delta(:,:,k), na1, &
                & forces(:,xyz2), 1, 1.0d0, y(:,k), 1)
            
            ! y(:,k) = y(:,k) + matmul(kernel_delta(:,:,k), forces(:,xyz2))!* inv_2dx

            t_end = omp_get_wtime()
            write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        enddo

    enddo

    deallocate(kernel_delta)
    deallocate(self_scalar2)
    deallocate(ksi2)
    deallocate(cosp2)
    deallocate(sinp2)

    allocate(kernel_MA(nm1,na1,nsigmas))
    kernel_MA = 0.0d0
 
    write (*,"(A)", advance="no") "KERNEL"

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(ni,nj,idx1,l2dist,idx1_start)
    do a = 1, nm1
        ni = n1(a)
        idx1_start = sum(n1(:a)) - ni
        do i = 1, ni
        
            idx1 = idx1_start + i
            
            do b = 1, nm1
                nj = n1(b)
                do j = 1, nj
 
                    l2dist = scalar(x1(a,i,:,:), x1(b,j,:,:), &
                        & nneigh1(a,i), nneigh1(b,j), ksi1(a,i,:), ksi1(b,j,:), &
                        & sinp1(a,i,:,:,:), sinp1(b,j,:,:,:), &
                        & cosp1(a,i,:,:,:), cosp1(b,j,:,:,:), &
                        & t_width, d_width, cut_distance, order, &
                        & pd, ang_norm2, distance_scale, angular_scale, alchemy)
 
                    l2dist = self_scalar1(a,i) + self_scalar1(b,j) - 2.0d0 * l2dist
 
                    do k = 1, nsigmas
                       kernel_MA(b, idx1, k) = kernel_MA(b, idx1, k) &
                            & + exp(l2dist * inv_sigma2(k))
                    enddo
 
                enddo
            enddo
 
        enddo
    enddo
    !$OMP END PARALLEL DO
    
    t_end = omp_get_wtime()
    write (*,"(A,F12.4,A)") "                                  Time = ", t_end - t_start, " s"
    
    deallocate(self_scalar1)
    deallocate(ksi1)
    deallocate(cosp1)
    deallocate(sinp1)
 
    do k = 1, nsigmas
        
        ! kernel_scratch(:,:,k) = kernel_scratch(:,:,k) &
        !    & + matmul(transpose(kernel_MA(:,:,k)),kernel_MA(:,:,k))
 
        ! y(:,k) = y(:,k) + matmul(transpose(kernel_MA(:,:,k)), energies(:))

        write (*,"(A,F12.4)", advance="no") "     DSYRK()    sigma =", sigmas(k)
        t_start = omp_get_wtime()

        call dsyrk("U", "T", na1, nm1, 1.0d0, kernel_MA(:,:,k), nm1, &
            & 1.0d0, kernel_scratch(:,:,k), na1)

        t_end = omp_get_wtime()
        write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"
 
        write (*,"(A,F12.4)", advance="no") "     DGEMV()    sigma =", sigmas(k)
        t_start = omp_get_wtime()
            
        call dgemv("T", nm1, na1, 1.0d0, kernel_ma(:,:,k), nm1, &
                      & energies(:), 1, 1.0d0, y(:,k), 1)

        t_end = omp_get_wtime()
        write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"
 
    enddo

    deallocate(kernel_ma)

    do k = 1, nsigmas
        do i = 1, na1
            kernel_scratch(i,i,k) = kernel_scratch(i,i,k) + llambda
        enddo
    enddo

    alphas = 0.0d0

    write (*,"(A)") "CHOLESKY DECOMPOSITION"
    do k = 1, nsigmas

        write (*,"(A,F12.4)", advance="no") "     DPOTRF()   sigma =", sigmas(k)
        t_start = omp_get_wtime()

        call dpotrf("U", na1, kernel_scratch(:,:,k), na1, info)
        if (info > 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", info, "-th leading order is not positive definite."
        else if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        t_end = omp_get_wtime()
        write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        write (*,"(A,F12.4)", advance="no") "     DPOTRS()   sigma =", sigmas(k)
        t_start = omp_get_wtime()

        call dpotrs("U", na1, 1, kernel_scratch(:,:,k), na1, y(:,k), na1, info)
        if (info < 0) then
            write (*,*) "QML WARNING: Error in LAPACK Cholesky solver DPOTRS()."
            write (*,*) "QML WARNING: The", -info, "-th argument had an illegal value."
        endif

        t_end = omp_get_wtime()
        write (*,"(A,F12.4,A)") "     Time = ", t_end - t_start, " s"

        alphas(k,:) = y(:,k)
    enddo

    deallocate(y)
    deallocate(kernel_scratch)

end subroutine fget_local_invariant_alphas_fchl
