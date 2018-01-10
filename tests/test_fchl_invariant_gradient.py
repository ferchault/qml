#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append("/home/andersx/dev/qml/fchl_gdml/build/lib.linux-x86_64-2.7")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from time import time
import ast

import scipy
import scipy.stats

from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv
import pandas as pd

import qml
from qml.math import cho_solve
from qml.fchl import generate_fchl_representation
from qml.fchl import generate_displaced_fchl_representations

from qml.fchl import get_local_symmetric_kernels_fchl
from qml.fchl import get_local_kernels_fchl

from qml.fchl import get_local_atomic_kernels_fchl

from qml.fchl import get_local_hessian_kernels_fchl
from qml.fchl import get_local_symmetric_hessian_kernels_fchl

from qml.fchl import get_atomic_gradient_kernels_fchl
from qml.fchl import get_local_gradient_kernels_fchl

from qml.fchl import get_local_full_kernels_fchl
from qml.fchl import get_local_invariant_alphas_fchl

np.set_printoptions(linewidth=19999999999, suppress=True, edgeitems=10)


SIGMAS = [1.28]
FORCE_KEY  = "mopac_forces"
ENERGY_KEY = "mopac_energy"
CSV_FILE = "data/01.csv"

# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/1a_1200.csv"

# SIGMAS = [0.32]
# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/2a_1200.csv"


# SIGMAS = [10.24]
# FORCE_KEY  = "mopac_forces"
# ENERGY_KEY = "mopac_energy"
# CSV_FILE = "data/02.csv"

# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/molecule_300.csv"

FORCE_KEY = "forces"
ENERGY_KEY = "om2_energy"
CSV_FILE = "data/amons_10_300k.csv"
# CSV_FILE = "data/amons_small.csv"
# CSV_FILE = "data/01.csv"

TRAINING = int(sys.argv[1])

CUT_DISTANCE = 1e6
LLAMBDA = 1e-7
LLAMBDA_ENERGY = 1e-10
LLAMBDA_FORCE = 1e-7

SIGMAS = [0.01 * 2**i for i in range(20)]
# SIGMAS = [0.01 * 2**i for i in range(10)]
# SIGMAS = [0.01 * 2**i for i in range(10, 20)]
# SIGMAS = [10.24]
# SIGMAS = [100.0]

DX = 0.05

kernel_args={
    "cut_distance": CUT_DISTANCE, 
    "alchemy": "off",
    # "scale_distance": 1.0,
    # "d_width": 0.15,
    "two_body_power": 2.0,

    # "scale_angular": 0.5,
    "three_body_power": 1.0,
    # "t_width": np.pi/2,
}

SIZE=19



def csv_to_molecular_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):

    df = pd.read_csv(csv_filename)

    c = []
    n = []
    x = []
    f = []
    e = []
    distance = []

    disp_x = []

    # max_atoms = max([len(ast.literal_eval(df["atomtypes"][i])) for i in range(len(df))])
    max_atoms = SIZE

    print("MAX ATOMS:", max_atoms)


    IDX = np.array(range(len(df)))
    np.random.shuffle(IDX)
    IDX = IDX[:TRAINING]

    print(IDX)
    for i in IDX:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))

        dist = norm(coordinates[0] - coordinates[1])
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))
        energy = df[energy_key][i]

        rep = generate_fchl_representation(coordinates, nuclear_charges, 
                size=max_atoms, cut_distance=CUT_DISTANCE)
        disp_rep = generate_displaced_fchl_representations(coordinates, nuclear_charges, 
                size=max_atoms, cut_distance=CUT_DISTANCE, dx=DX)

        distance.append(dist)
        x.append(rep)
        f.append(force)
        e.append(energy)
        n.append(nuclear_charges)
        disp_x.append(disp_rep)
        c.append(coordinates)

    return np.array(x), f, e, distance, np.array(disp_x), n, c

def test_force():
    
    Xall, Fall, Eall, Dall, dXall, Nall, Call = csv_to_molecular_reps(CSV_FILE,
                                force_key=FORCE_KEY, energy_key=ENERGY_KEY)


    X  = Xall [:TRAINING]
    dX = dXall[:TRAINING]
    F  = Fall [:TRAINING]
    E  = np.array(Eall [:TRAINING])
    N  = Nall [:TRAINING]
    
    Xs  = Xall [-1:]
    dXs = dXall[-1:]
    Fs  = Fall [-1:]
    Es  = Eall [-1:]

    print(Es)
    print(Fs)

    Xs_numm = []

    coordinates = Call[-1]
    nuclear_charges = Nall[-1]

    rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
    Xs_numm.append(rep)

    print(coordinates.shape)
    for i in range(coordinates.shape[0]):
        for xyz in range(coordinates.shape[1]):
            
            coord = coordinates[i,xyz]

            coords = deepcopy(coordinates)

            coords[i,xyz] = coord + DX

            rep = generate_fchl_representation(coords, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
            Xs_numm.append(rep)
            
            coords[i,xyz] = coord - DX

            rep = generate_fchl_representation(coords, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )
            Xs_numm.append(rep)
            
            coords[i,xyz] = coord

    Xs_numm = np.array(Xs_numm)
    print(Xs_numm.shape)

    Xs_force = Xs[:1]

    Ftrain = np.concatenate(F)

    alphas      = get_local_invariant_alphas_fchl(X, dX, Ftrain, SIGMAS, energy=E, dx=DX, **kernel_args)
    
    Kt_force = get_atomic_gradient_kernels_fchl(        X,  dX,   SIGMAS, dx=DX, **kernel_args)
    Ks_force = get_atomic_gradient_kernels_fchl(        X, dXs,   SIGMAS, dx=DX, **kernel_args)
    
    Kt_energy = get_local_atomic_kernels_fchl(X, X,   SIGMAS, **kernel_args)
    Ks_energy = get_local_atomic_kernels_fchl(X, Xs,  SIGMAS, **kernel_args)
    Ks_numm_energy = get_local_atomic_kernels_fchl(X, Xs_numm,  SIGMAS, **kernel_args)


    # print(Ks_numm_energy.shape)

    # print("TEST FORCE KERNEL")
    # Ks_force = get_scalar_vector_kernels_fchl(X, Xs_force, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy=ALCHEMY,
    #             )

    # TEST = Xs.shape[0]

    # X = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    # Xs = np.reshape(Xs, (TEST*SIZE,5,SIZE))

    # print("TEST ENERGY KERNEL")
    # Ks_large = get_atomic_kernels_fchl(X, Xs, SIGMAS,
    #                 cut_distance=CUT_DISTANCE, 
    #                 alchemy=ALCHEMY,
    #             )

    # for i, sigma in enumerate(SIGMAS):
    for i, sigma in enumerate(SIGMAS):

        Ft  = np.zeros((Kt_force[i,:,:].shape[1]/3,3))
        Fss = np.zeros((Ks_force[i,:,:].shape[1]/3,3))

        for xyz in range(3):
            
            Ft[:,xyz]  = np.dot(Kt_force[i,:,xyz::3].T, alphas[i])
            Fss[:,xyz] = np.dot(Ks_force[i,:,xyz::3].T, alphas[i])
        
        Ess = np.dot(Ks_numm_energy[i].T, alphas[i])
        Et  = np.dot(Kt_energy[i].T, alphas[i])

        Fss_num = np.zeros((coordinates.shape[0],3))

        idx = 1

        # print(Ess)
        for j in range(coordinates.shape[0]):
            for xyz in range(coordinates.shape[1]):

                Fss_num[j,xyz] += Ess[idx]

                idx += 1
                
                Fss_num[j,xyz] -= Ess[idx]

                idx += 1

                # print(Ess[idx-1] - Ess[idx-2])

        Fss_num /= (2*DX)
  

        # print(Fs[0])
        print(Fss / Fss_num)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Ftrain.flatten(), Ft.flatten())
        print("TRAINING FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
               (np.mean(np.abs(Ft.flatten() - Ftrain.flatten())), sigma, slope, intercept, r_value ))
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())
        print("TRAINING ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Et - E)), sigma, slope, intercept, r_value ))
    #     
    #     if abs (sigma - 1.28) > 0.001: continue

    #     Fss = np.einsum('jkl,l->kj', Ks_force[i], alphas[i])

    #     # print(Fss)
    #     

    #     Ks = np.zeros((TRAINING*SIZE,TEST))

    #     for j in range(TRAINING*SIZE):
    #         for k in range(TEST):

    #            Ks[j,k] = np.sum(Ks_large[i,j,k*SIZE:(k+1)*SIZE])

    #     Ess = np.dot(Ks.T, alphas[i])



    #     # Ess -= intercept

    #     # print(Ess[0]/2)
    #     # print(Ess[1:]/2)

    #     print(Fs)
    #     print(Fss)
    #     print(Fss_num)
    #     print(Fss_num - Fss)
    #     print(Fss_num/Fss)

    #     print(scipy.stats.linregress(Fss_num.flatten(), Fss.flatten()))
    #     print(scipy.stats.linregress(Fss_num.flatten(), Fs.flatten()))
    #     print(scipy.stats.linregress(Fss.flatten(), Fs.flatten()))


    #     print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss_num - Fs)), sigma)
    #     print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss_num - Fss)), sigma)
    #     print("RMSE ENERGY", 
    #             Es/2,
    #             Ess[0]/2,
    #             Ess[0]/2 - Es/2,
    #             sigma,
    #         )


if __name__ == "__main__":

    test_force()
