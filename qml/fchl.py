# MIT License
#
# Copyright (c) 2017 Felix Faber and Anders Steen Christensen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import copy

from .ffchl_module import fget_kernels_fchl
from .ffchl_module import fget_symmetric_kernels_fchl
from .ffchl_module import fget_global_kernels_fchl
from .ffchl_module import fget_global_symmetric_kernels_fchl
from .ffchl_module import fget_atomic_kernels_fchl
from .ffchl_module import fget_atomic_symmetric_kernels_fchl

from .ffchl_module import fget_local_full_kernels_fchl
from .ffchl_module import fget_local_gradient_kernels_fchl
from .ffchl_module import fget_local_hessian_kernels_fchl
from .ffchl_module import fget_local_symmetric_hessian_kernels_fchl

from .ffchl_module import fget_local_invariant_alphas_fchl
from .ffchl_module import fget_atomic_gradient_kernels_fchl
from .ffchl_module import fget_local_atomic_kernels_fchl

from .ffchl_module import fget_kernels_fchl_ef

from .alchemy import get_alchemy
from .fchl_kernels import get_kernel_parameters

def generate_displaced_representations(coordinates, nuclear_charges,
                    max_size=23, neighbors=23, cut_distance = 5.0, cell=None, dx=0.005): 
    """ Generates displaced representations for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param dx: Real-space displacement for numerical derivatives, in units of angstrom.
    :type dx: float
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """
    size = max_size
    if cell is None:
        neighbors=size
    reps = np.zeros((3,2,size,size,5,neighbors))

    compound_size = len(nuclear_charges)

    for xyz in range(3):

        for i in range(compound_size):
            for idisp, disp in enumerate([-dx, dx]):

                displaced_coordinates = copy.deepcopy(coordinates)
                displaced_coordinates[i,xyz] += disp

                rep = generate_representation(displaced_coordinates, nuclear_charges,
                    max_size=size, neighbors=neighbors, cut_distance=cut_distance, cell=cell)

                reps[xyz,idisp,i,:,:,:] = rep[:,:,:]

    return reps

def generate_representation_ef(coordinates, nuclear_charges, partial_charges,
        max_size=23, neighbors=23, cut_distance = 5.0, cell=None):
    """ Generates a representation for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """

    size = max_size

    if cell is None:
        neighbors=size

    L = len(coordinates)
    coords = np.asarray(coordinates)
    ocupationList = np.asarray(nuclear_charges)
    partial_charges = np.asarray(partial_charges)
    M =  np.zeros((size,6,neighbors))

    if cell is not None:
        print "ERROR: sorry, I can only do EF for molecules!"
        exit()

    coordsExt = copy.copy(coords)
    partialExt = copy.copy(partial_charges)
    ocupationListExt = copy.copy(ocupationList)

    M[:,0,:] = 1E+100

    for i in range(L):
        cD = - coords[i] + coordsExt[:]

        ocExt =  np.asarray(ocupationListExt)
        qExt =  np.asarray(partialExt)
        
        D1 = np.sqrt(np.sum(cD**2, axis = 1))
        args = np.argsort(D1)
        D1 = D1[args]
        
        ocExt = np.asarray([ocExt[l] for l in args])
        qExt = np.asarray([qExt[l] for l in args])

        cD = cD[args]


        args = np.where(D1 < cut_distance)[0]
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        qExt = np.asarray([qExt[l] for l in args])

        cD = cD[args]
        M[i,0,: len(D1)] = D1
        M[i,1,: len(D1)] = ocExt[:]
        M[i,2:5,: len(D1)] = cD.T
        M[i,5,: len(D1)] = qExt[:]


    return M


def generate_representation(coordinates, nuclear_charges,
        max_size=23, neighbors=23, cut_distance = 5.0, cell=None):
    """ Generates a representation for the FCHL kernel module.

    :param coordinates: Input coordinates.
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param max_size: Max number of atoms in representation.
    :type max_size: integer
    :param neighbors: Max number of atoms within the cut-off around an atom. (For periodic systems)
    :type neighbors: integer
    :param cell: Unit cell vectors. The presence of this keyword argument will generate a periodic representation.
    :type cell: numpy array
    :param cut_distance: Spatial cut-off distance - must be the same as used in the kernel function call.
    :type cut_distance: float
    :return: FCHL representation, shape = (size,5,neighbors).
    :rtype: numpy array
    """

    size = max_size

    if cell is None:
        neighbors=size

    L = len(coordinates)
    coords = np.asarray(coordinates)
    ocupationList = np.asarray(nuclear_charges)
    M =  np.zeros((size,5,neighbors))

    if cell is not None:
        coords = np.dot(coords,cell)
        nExtend = (np.floor(cut_distance/np.linalg.norm(cell,2,axis = 0)) + 1).astype(int)

        for i in range(-nExtend[0],nExtend[0] + 1):
            for j in range(-nExtend[1],nExtend[1] + 1):
                for k in range(-nExtend[2],nExtend[2] + 1):
                    if i == -nExtend[0] and j  == -nExtend[1] and k  == -nExtend[2]:
                        coordsExt = coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:]
                        ocupationListExt = copy.copy(ocupationList)
                    else:
                        ocupationListExt = np.append(ocupationListExt,ocupationList)
                        coordsExt = np.append(coordsExt,coords + i*cell[0,:] + j*cell[1,:] + k*cell[2,:],axis = 0)
    else:
        coordsExt = copy.copy(coords)
        ocupationListExt = copy.copy(ocupationList)

    M[:,0,:] = 1E+100

    for i in range(L):
        cD = - coords[i] + coordsExt[:]

        ocExt =  np.asarray(ocupationListExt)
        D1 = np.sqrt(np.sum(cD**2, axis = 1))
        args = np.argsort(D1)
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        cD = cD[args]

        args = np.where(D1 < cut_distance)[0]
        D1 = D1[args]
        ocExt = np.asarray([ocExt[l] for l in args])
        cD = cD[args]
        M[i,0,: len(D1)] = D1
        M[i,1,: len(D1)] = ocExt[:]
        M[i,2:5,: len(D1)] = cD.T
    return M


def get_local_kernels(A, B, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    return fget_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, nm1, nm2, n_kernels, \
                three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)


def get_local_symmetric_kernels(A, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6,
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`A_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, N),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    nm1 = A.shape[0]
    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)
    kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    return fget_symmetric_kernels_fchl(A, N1, neighbors1, nm1, n_kernels, \
                three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)


def get_global_symmetric_kernels(A, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6,
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`A_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, N),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    nm1 = A.shape[0]
    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)
    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    return fget_global_symmetric_kernels_fchl(A, N1, neighbors1, nm1, nsigmas, \
            three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)


def get_global_kernels(A, B, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6,
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes!"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes!"

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)
    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    return fget_global_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, nm1, nm2, nsigmas, \
            three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)


def get_atomic_kernels(A, B, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6,
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, size).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, size).
        :type B: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    assert len(A.shape) == 3
    assert len(B.shape) == 3

    # assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    # assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"

    na1 = A.shape[0]
    na2 = B.shape[0]

    neighbors1 = np.zeros((na1), dtype=np.int32)
    neighbors2 = np.zeros((na2), dtype=np.int32)

    for i, x in enumerate(A):
        neighbors1[i] = len(np.where(x[0]< cut_distance)[0])

    for i, x in enumerate(B):
        neighbors2[i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)
    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)
    
    return fget_atomic_kernels_fchl(A, B, neighbors1, neighbors2, na1, na2, nsigmas, \
                three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)


def get_atomic_symmetric_kernels(A, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6,
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, size).
        :type A: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    assert len(A.shape) == 3

    na1 = A.shape[0]

    neighbors1 = np.zeros((na1), dtype=np.int32)

    for i, x in enumerate(A):
        neighbors1[i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)
    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    return fget_atomic_symmetric_kernels_fchl(A, neighbors1, na1, nsigmas, \
                three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, kernel_idx, kernel_parameters)

def get_local_full_kernels(A, B, dx=0.005, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    assert B.shape[1] == 3
    assert B.shape[2] == 2
    assert B.shape[5] == 5
    assert A.shape[2] == 5
    # assert B.shape[2] == 2
    # assert B.shape[5] == 5

    atoms_max = B.shape[4]
    assert A.shape[1] == atoms_max
    assert B.shape[3] == atoms_max

    neighbors_max = B.shape[6]
    assert A.shape[3] == neighbors_max


    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a,0,0,0,:,1,0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m,xyz,pm,i,:ni]):
                        neighbors2[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    naq2 = np.sum(N2) * 3

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    return fget_local_full_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, naq2, nsigmas, three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, dx, kernel_idx, kernel_parameters)

def get_local_gradient_kernels(A, B, dx=0.005,
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    assert B.shape[1] == 3
    assert B.shape[2] == 2
    assert B.shape[5] == 5
    assert A.shape[2] == 5
    # assert B.shape[2] == 2
    # assert B.shape[5] == 5

    atoms_max = B.shape[4]
    assert A.shape[1] == atoms_max
    assert B.shape[3] == atoms_max

    neighbors_max = B.shape[6]
    assert A.shape[3] == neighbors_max


    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a,0,0,0,:,1,0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m,xyz,pm,i,:ni]):
                        neighbors2[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    naq2 = np.sum(N2) * 3

    return fget_local_gradient_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, naq2, nsigmas, three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, dx, kernel_idx, kernel_parameters)


def get_local_hessian_kernels(A, B, dx=0.005,
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    assert A.shape[1] == 3
    assert A.shape[2] == 2
    assert A.shape[5] == 5
    assert B.shape[1] == 3
    assert B.shape[2] == 2
    assert B.shape[5] == 5

    atoms_max = A.shape[4]
    assert A.shape[3] == atoms_max
    assert B.shape[3] == atoms_max

    neighbors_max = A.shape[6]
    assert B.shape[6] == neighbors_max


    # N1 = np.zeros((nm1, 3, 2, atoms_max),dtype=np.int32)
    # N2 = np.zeros((nm2, 3, 2, atoms_max),dtype=np.int32)

    # for a in range(nm1):
    #     for xyz in range(3):
    #         for pm in range(2):
    #             for i in range(atoms_max):
    #                N1[a,xyz,pm,i] = len(np.where(A[a,xyz,pm,i,:,1,0] > 0.0001)[0])

    # # for a in range(nm2):
    # #    N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    # for a in range(nm2):
    #     for xyz in range(3):
    #         for pm in range(2):
    #             for i in range(atoms_max):
    #                N2[a,xyz,pm,i] = len(np.where(B[a,xyz,pm,i,:,1,0] > 0.0001)[0])

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,0,0,0,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,0,0,0,:,1,0] > 0.0001)[0])

    # for a in range(nm2):
    #    N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, 3, 2, atoms_max, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm1):
        ni = N1[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(A[m,xyz,pm,i,:ni]):
                        neighbors1[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m,xyz,pm,i,:ni]):
                        neighbors2[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)


    naq1 = np.sum(N1) * 3
    naq2 = np.sum(N2) * 3

    # print naq1, naq2, nsigmas
    return fget_local_hessian_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, naq1, naq2, nsigmas, three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, dx, kernel_idx, kernel_parameters)

    
def get_local_symmetric_hessian_kernels(A, dx=0.005,
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):

    nm1 = A.shape[0]

    assert A.shape[1] == 3
    assert A.shape[2] == 2
    assert A.shape[5] == 5

    atoms_max = A.shape[4]
    assert A.shape[3] == atoms_max

    neighbors_max = A.shape[6]

    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,0,0,0,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm1):
        ni = N1[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(A[m,xyz,pm,i,:ni]):
                        neighbors1[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)


    naq1 = np.sum(N1) * 3

    return fget_local_symmetric_hessian_kernels_fchl(A, N1, neighbors1, nm1, naq1, nsigmas,  \
        three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, three_body_power, dx, kernel_idx, kernel_parameters)

def get_local_invariant_alphas(A, B, F, energy=None, dx=0.005, regularization=1e-7,
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):
    nm1 = A.shape[0]
    nm2 = B.shape[0]

    assert B.shape[1] == 3
    assert B.shape[2] == 2
    assert B.shape[5] == 5
    assert A.shape[2] == 5
    # assert B.shape[2] == 2
    # assert B.shape[5] == 5

    atoms_max = B.shape[4]
    assert A.shape[1] == atoms_max
    assert B.shape[3] == atoms_max

    neighbors_max = B.shape[6]
    assert A.shape[3] == neighbors_max


    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a,0,0,0,:,1,0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m,xyz,pm,i,:ni]):
                        neighbors2[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)
    naq2 = np.sum(N2) * 3

    E = np.zeros((nm1))
    if energy is not None:
        E = energy

    return fget_local_invariant_alphas_fchl(A, B, F, E, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, na1, nsigmas,  three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power,
        three_body_power, dx, kernel_idx, kernel_parameters, regularization)

def get_atomic_gradient_kernels(A, B, dx = 0.005,
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    assert B.shape[1] == 3
    assert B.shape[2] == 2
    assert B.shape[5] == 5
    assert A.shape[2] == 5

    atoms_max = B.shape[4]
    assert A.shape[1] == atoms_max
    assert B.shape[3] == atoms_max

    neighbors_max = B.shape[6]
    assert A.shape[3] == neighbors_max


    N1 = np.zeros((nm1),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a,0,0,0,:,1,0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m,xyz,pm,i,:ni]):
                        neighbors2[m,xyz,pm,i,a] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)


    na1 = np.sum(N1)
    naq2 = np.sum(N2) * 3

    return fget_atomic_gradient_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, na1, naq2, nsigmas, three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power,
        three_body_power, dx, kernel_idx, kernel_parameters)

    
def get_local_atomic_kernels(A, B, \
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array
        :param sigma: List of kernel-widths.
        :type sigma: list
        :param t_width: Gaussian width for the angular (theta) terms.
        :type t_width: float
        :param d_width: Gaussian width for the distance terms.
        :type d_width: float
        :param cut_start: The fraction of the cut-off radius at which cut-off damping start
        :type cut_start: float
        :param cut_distance: Cut-off radius.
        :type cut_distance: float
        :param r_width: Gaussian width along rows in the periodic table.
        :type r_width: float
        :param c_width: Gaussian width along columns in the periodic table.
        :type c_width: float
        :param order: Fourier-expansion truncation order.
        :type order: integer
        :param scale_distance: Weight for distance-dependent terms.
        :type scale_distance: float
        :param scale_angular: Weight for angle-dependent terms.
        :type scale_angular: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)

    return fget_local_atomic_kernels_fchl(A, B, N1, N2, neighbors1, neighbors2, \
        nm1, nm2, na1, nsigmas, three_body_width, two_body_width, cut_start, cut_distance, \
        fourier_order, pd, two_body_scaling, three_body_scaling, doalchemy, two_body_power, \
        three_body_power, kernel_idx, kernel_parameters)


def get_local_kernels_ef(A, B, df=1e-5,\
        two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
        two_body_width=0.2, three_body_width=np.pi,
        two_body_power=4.0, three_body_power=2.0,
        cut_start=1.0, cut_distance=5.0,
        fourier_order=1, alchemy="periodic-table",
        alchemy_period_width=1.6, alchemy_group_width=1.6, 
        kernel="gaussian", kernel_args=None):
    """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

            :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

        Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
        K is calculated analytically using an OpenMP parallel Fortran routine.
        Note, that this kernel will ONLY work with FCHL representations as input.

        :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
        :type A: numpy array
        :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
        :type B: numpy array

        :param two_body_scaling: Weight for 2-body terms.
        :type two_body_scaling: float
        :param three_body_scaling: Weight for 3-body terms.
        :type three_body_scaling: float

        :param two_body_width: Gaussian width for 2-body terms
        :type two_body_width: float
        :param three_body_width: Gaussian width for 3-body terms.
        :type three_body_width: float

        :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
        :type two_body_power: float
        :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
        :type three_body_power: float

        :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
        :type cut_start: float
        :param cut_distance: Cut-off radius. (default=5 angstrom)
        :type cut_distance: float

        :param fourier_order: 3-body Fourier-expansion truncation order.
        :type fourier_order: integer
        :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
        :type alchemy: string

        :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
        :type alchemy_period_width: float
        :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
        :type alchemy_group_width: float

        :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
        :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    assert B.shape[1] == atoms_max, "ERROR: Check FCHL representation sizes! code = 2"
    assert B.shape[3] == neighbors_max, "ERROR: Check FCHL representation sizes! code = 3"


    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1),dtype=np.int32)
    N2 = np.zeros((nm2),dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

    # charge1 = np.zeros((nm1, atoms_max))
    # charge2 = np.zeros((nm2, atoms_max))
    # 
    # for a, representation in enumerate(A):
    #     ni = N1[a]
    #     for i, x in enumerate(representation[:ni]):
    #         charge1[a,i] = Q1[a][i]
    # 
    # for a, representation in enumerate(B):
    #     ni = N2[a]
    #     for i, x in enumerate(representation[:ni]):
    #         charge2[a,i] = Q2[a][i]

    # print(charge1)
    # print(charge2)

    doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

    kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    return fget_kernels_fchl_ef(A, B, N1, N2, neighbors1, neighbors2, nm1, nm2, n_kernels, \
                three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling, 
                three_body_scaling, doalchemy, two_body_power, three_body_power, df, kernel_idx, kernel_parameters)
