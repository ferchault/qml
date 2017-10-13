#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append("/home/andersx/dev/qml/fchl_forceenergy/build/lib.linux-x86_64-2.7")

import matplotlib.pyplot as plt

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
from qml.fchl import get_atomic_force_alphas_fchl
from qml.fchl import get_atomic_force_kernels_fchl
from qml.fchl import get_scalar_vector_alphas_fchl
from qml.fchl import get_scalar_vector_kernels_fchl
from qml.fchl import get_local_symmetric_kernels_fchl
from qml.fchl import get_local_kernels_fchl
from qml.fchl import get_atomic_kernels_fchl

np.set_printoptions(linewidth=19999999999, suppress=True, edgeitems=10)



# SIGMAS = [1.28]
# SIZE = 3
# FORCE_KEY  = "mopac_forces"
# ENERGY_KEY = "mopac_energy"
# CSV_FILE = "data/01.csv"

# SIZE = 5
# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/1a_1200.csv"

# SIGMAS = [0.32]
# SIZE = 6
# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/2a_1200.csv"


SIGMAS = [10.24]
SIZE = 6
FORCE_KEY  = "orca_forces"
ENERGY_KEY = "orca_energy"
CSV_FILE = "data/02.csv"

# SIZE = 19
# FORCE_KEY = "forces"
# ENERGY_KEY = "om2_energy"
# CSV_FILE = "data/molecule_300.csv"

TRAINING = 100
TEST     = 200

CUT_DISTANCE = 1e6
LLAMBDA = 1e-6
SIGMAS = [0.01 * 2**i for i in range(20)]
# SIGMAS = [0.01 * 2**i for i in range(10)]
# SIGMAS = [0.01 * 2**i for i in range(10, 20)]
SIGMAS = [10.24]
ENERGY_SCALE = 1.0 #/ SIZE
FORCE_SCALE = 1.0 #/ (SIZE / 3.0 * 2.0)

def csv_to_atomic_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):


    df = pd.read_csv(csv_filename)

    reps = []
    y = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        for j in range(len(atomtypes)):

            reps.append(rep[j])
            y.append(force[j])


    return np.array(reps), np.array(y)


def csv_to_molecular_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):


    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    distance = []

    disp_x = []

    for i in range(len(df)):

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))

        dist = norm(coordinates[0] - coordinates[1])
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))

        energy = df[energy_key][i]

        rep = generate_fchl_representation(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        disp_rep = generate_displaced_fchl_representations(coordinates, nuclear_charges,
                                            size=SIZE,
                                            cut_distance=CUT_DISTANCE,
                                        )

        distance.append(dist)
        x.append(rep)
        f.append(force)
        e.append(energy)

        disp_x.append(disp_rep)
    # distance = np.array(distance)[::2]

    return np.array(x), f, e, distance, np.array(disp_x)


def get_gdml_kernel(X):

    pass


def test_hessian():

    coords = np.array([[0.0, 0.0, 0.0], 
                       [-0.926021, -0.036279, 0.354716], 
                       [0.354364, 0.731748, 0.541187]])
    nuc = np.array([8, 1, 1])
    size=3
    rep = generate_fchl_representation(coords, nuc,
             size=size, neighbors=size, cut_distance=CUT_DISTANCE, cell=None)

    HESS = np.zeros((9,9))

    coordp1 = deepcopy(coords)
    coordm1 = deepcopy(coords)
    coordp2 = deepcopy(coords)
    coordm2 = deepcopy(coords)

    dx = 0.0005

    coordp1[0,0] += dx
    coordm1[0,0] -= dx

    
    coordp2[0,1] += dx
    coordm2[0,1] -= dx

    repp1 = generate_fchl_representation(coordp1, nuc,
             size=size, neighbors=size, cut_distance=CUT_DISTANCE, cell=None)

    repm1 = generate_fchl_representation(coordm1, nuc,
             size=size, neighbors=size, cut_distance=CUT_DISTANCE, cell=None)

    repp2 = generate_fchl_representation(coordp2, nuc,
             size=size, neighbors=size, cut_distance=CUT_DISTANCE, cell=None)

    repm2 = generate_fchl_representation(coordm2, nuc,
             size=size, neighbors=size, cut_distance=CUT_DISTANCE, cell=None)

    sigma = SIGMAS[0]

    Kpp = np.sum(get_atomic_kernels_fchl(repp1, repp2,
            [sigma], cut_distance=CUT_DISTANCE, alchemy="off")[0])

    Kmp = np.sum(get_atomic_kernels_fchl(repm1, repp2,
            [sigma], cut_distance=CUT_DISTANCE, alchemy="off")[0])

    Kpm = np.sum(get_atomic_kernels_fchl(repp1, repm2,
            [sigma], cut_distance=CUT_DISTANCE, alchemy="off")[0])

    Kmm = np.sum(get_atomic_kernels_fchl(repm1, repm2,
            [sigma], cut_distance=CUT_DISTANCE, alchemy="off")[0])

    print("%20.8f %20.8f %20.8f %20.8f %f" %(Kmm, Kmp, Kpm, Kpp, 1.0/(4 * dx**2)))
    print((Kpp - Kpm - Kmp + Kmm))
    print((Kpp - Kpm - Kmp + Kmm) / (4 * dx**2))


def test_new_forces():
    
    Xall, Fall, Eall, Dall, dXall = csv_to_molecular_reps(CSV_FILE,
                                force_key=FORCE_KEY, energy_key=ENERGY_KEY)

    print(Xall.shape)
    print(dXall.shape)

    Eall = np.array(Eall)
    Fall = np.array(Fall)

    # Eall -= Eall.mean()
    train = TRAINING
    test = TEST

    X = Xall[:train]
    dX = dXall[:train]
    F = Fall[:train]
    E = Eall[:train]
    D = Dall[:train]
    
    Xs = Xall[-test:]
    dXs = dXall[-test:]
    Fs = Fall[-test:]
    Es = Eall[-test:]
    Ds = Dall[-test:]

    print("allocate")
    K = np.zeros((len(SIGMAS),3*TRAINING*SIZE,3*TRAINING*SIZE))
    Ks = np.zeros((len(SIGMAS),3*TRAINING*SIZE,3*TEST*SIZE))
    print("kernel")

    dx = 0.0005

    for idisp, dx_i in enumerate([-dx, dx]):

        # Xi = dXs[:,:,idisp,:,i,:,:]

        Xi = deepcopy(dX[:,:,idisp,:,:,:,:])
        # np.swapaxes(Xi, 3, 2)
        Xi = deepcopy(np.swapaxes(Xi, 1, 2))

        Xi = np.reshape(Xi, (TRAINING*3*SIZE, SIZE,5, SIZE))

        for jdisp, dx_j in enumerate([-dx, dx]):
            # Xj = dXs[:,:,jdisp,:,j,:,:]

            Xj = deepcopy(dX[:,:,jdisp,:,:,:,:])
            # np.swapaxes(Xj, 3, 2)
            Xj = deepcopy(np.swapaxes(Xj, 1, 2))

            print(idisp, jdisp, Xj.shape, dXs.shape)
            Xj = np.reshape(Xj, (TRAINING*3*SIZE,SIZE, 5, SIZE))
            print(idisp, jdisp, Xj.shape, dXs.shape)


            K_partial = get_local_kernels_fchl(
                Xi, Xj,
                SIGMAS,
                cut_distance=CUT_DISTANCE, 
                alchemy="off"
            )
            print(K_partial.shape)
            print(1.0/(4.0*dx_i*dx_j))
            print(K_partial[0,:9,:9])
            np.multiply(K_partial, 1.0/(4.0*dx_i*dx_j), out=K_partial)
            # print(K[0,:10,:10])
            print(K_partial.shape)
            K += K_partial
            # print(K[0,:10,:10])

    
    for idisp, dx_i in enumerate([-dx, dx]):

        Xi = deepcopy(dX[:,:,idisp,:,:,:,:])
        Xi = deepcopy(np.swapaxes(Xi, 1, 2))
        Xi = np.reshape(Xi, (TRAINING*3*SIZE, SIZE,5, SIZE))

        for jdisp, dx_j in enumerate([-dx, dx]):

            Xj = deepcopy(dXs[:,:,jdisp,:,:,:,:])
            Xj = deepcopy(np.swapaxes(Xj, 1, 2))
            Xj = np.reshape(Xj, (TEST*3*SIZE,SIZE, 5, SIZE))

            K_partial = get_local_kernels_fchl(
                Xi, Xj,
                SIGMAS,
                cut_distance=CUT_DISTANCE, 
                alchemy="off"
            )

            print(K_partial.shape)
            print(1.0/(4.0*dx_i*dx_j))
            print(K_partial[0,:9,:9])
            np.multiply(K_partial, 1.0/(4.0*dx_i*dx_j), out=K_partial)
            # print(K[0,:10,:10])
            print(K_partial.shape)
            Ks += K_partial
            # print(K[0,:10,:10])

    
    print(K[0,:9,:18])
    print(Ks[0,:9,:18])
    # exit()
    Y = np.reshape(F, (3*TRAINING*SIZE)) 
    print(F)
    print(Y)
    for i, sigma in enumerate(SIGMAS):

        C = deepcopy(K[i])
        C[np.diag_indices_from(C)] += LLAMBDA
        alpha = cho_solve(C, Y)
        print(SIGMAS[i])
        print(alpha[:20])
        Fss = np.dot(np.transpose(Ks[i]), alpha)
        # Fss = np.sum(Yss, axis=1)

        print("FSS.SHAPE", Fss.shape)

        Fs = np.array(Fs.flatten())
        # Fs = np.reshape(Fs, (Fss.shape[0], Fss.shape[1]))

        print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss - Fs)), sigma,
                scipy.stats.linregress(Fs.flatten(), Fss.flatten()),
                )

        # plt.figure(figsize=(10,4))
        # plt.subplot(121)
        plt.title("Forces, sigma = %10.2f" % sigma)
        plt.scatter(Fs.flatten(), Fss.flatten(), label="test")
        # plt.scatter(Fs.flatten(), Fs.flatten(), color="y", label="true")
        plt.legend()
        plt.xlabel("True Gradient [kcal/mol/angstrom]")
        plt.ylabel("GDML Gradient [kcal/mol/angstrom]")
        plt.savefig("F_GDML.png")
        plt.clf()
    
    # Ks_force_atomic = np.zeros([len(SIGMAS),3,2,SIZE,TRAINING*SIZE,TEST*SIZE])

    # X  = np.reshape(X, (TRAINING*SIZE,5,SIZE))
    # print(dXs.shape)

    # for xyz in range(3):
    #     for i in range(SIZE):
    #         for pm in range(2):

    #             Xs_force = dXs[:,xyz,pm,:,i,:,:]

    #             Xs_force = np.reshape(Xs_force, (TEST*SIZE,5,SIZE))
    #             print("KERNEL", xyz, i, pm, X.shape)
    #             print("KERNEL", xyz, i, pm, Xs_force.shape)

    #             Ks_force_atomic[:,xyz,pm,i,:,:] = get_atomic_kernels_fchl(
    #                     X, Xs_force, 
    #                     SIGMAS,
    #                     cut_distance=CUT_DISTANCE, 
    #                     alchemy="off"
    #             )

    # dKs_force_atomic = np.zeros([len(SIGMAS),3,SIZE,TRAINING*SIZE,TEST*SIZE])

    # for i in range(len(SIGMAS)):
    #     for xyz in range(3):
    #         dKs_force_atomic[i,xyz,:,:,:] = \
    #             (Ks_force_atomic[i,xyz,1,:,:,:] - Ks_force_atomic[i,xyz,0,:,:,:]) / (2.0 * 0.0005)

    # print(Ks_force_atomic.shape)
    # print(dKs_force_atomic.shape)

    # L = np.zeros((train*SIZE, train))
    # k = 0
    # for i in range(train):
    #     for j in range(SIZE):

    #         L[k,i] = 1
    #         k += 1
    # 
    # for i, sigma in enumerate(SIGMAS):

    #     for j in range(train):

    #         K[i,j,j] += LLAMBDA

    #     LTCKL = deepcopy(K[i])
    #     E = np.array(E)
    #     alphas = cho_solve(K[i], E)

    #     alpha_atomic = np.dot(L,alphas)
    #     
    #     Yss = np.einsum('jklm,l->mkj', dKs_force_atomic[i], alpha_atomic)
    #     Fss = np.sum(Yss, axis=1)

    #     print("FSS.SHAPE", Fss.shape)

    #     Fs = np.array(Fs)
    #     Fs = np.reshape(Fs, (Fss.shape[0], Fss.shape[1]))

    #     print("RMSE FORCE COMPONENT", np.mean(np.abs(Fss - Fs)), sigma,
    #             scipy.stats.linregress(Fs.flatten(), Fss.flatten()),
    #             )

    #     # plt.figure(figsize=(10,4))
    #     # plt.subplot(121)
    #     plt.title("Forces, sigma = %10.2f" % sigma)
    #     plt.scatter(Fs.flatten(), Fss.flatten(), label="test")
    #     # plt.scatter(Fs.flatten(), Fs.flatten(), color="y", label="true")
    #     plt.legend()
    #     plt.xlabel("True Gradient [kcal/mol/angstrom]")
    #     plt.ylabel("GDML Gradient [kcal/mol/angstrom]")
    #     plt.savefig("F_GDML.png")
    #     plt.clf()

    #     # Ess = np.dot(Ks[i], alphas)

    #     # print("RMSE ENERGY", np.mean(np.abs(Ess - Es)), sigma,
    #     #                  scipy.stats.linregress(Ess, Es),
    #     #                     )
    #     # Et = np.dot(Kt[i], alphas)
    #     # print("RMSE ENERGY Train", np.mean(np.abs(E - Et)), sigma,
    #     #                 scipy.stats.linregress(Et,E),
    #     #                     )

    #     # plt.subplot(122)
    #     # plt.title("Energy, sigma = %10.2f" % sigma)
    #     # plt.scatter(Es, Ess,   label="test")
    #     # # plt.scatter(E,  Et,    label="training" )
    #     # # plt.scatter(E,  E,   color="y", label="true" )
    #     # plt.xlabel("True Energy [kcal/mol]")
    #     # plt.ylabel("ML Energy [kcal/mol]")
    #     # plt.legend()
    #     # plt.savefig("E_scatter.png")
    #     # plt.clf()


if __name__ == "__main__":

    # Xall, Fall, Eall = csv_to_molecular_reps(CSV_FILE, 
    #                             force_key=FORCE_KEY, energy_key=ENERGY_KEY)
    # Eall = np.array(Eall)
    # print("STD ENERGY", Eall.std())

    # test_old_forces()
    # test_old_energy()
    test_hessian()
    test_new_forces()

