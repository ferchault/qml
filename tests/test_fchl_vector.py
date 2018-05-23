from __future__ import print_function

import ast
import time

import scipy
import scipy.stats

from copy import deepcopy

import numpy as np
from numpy.linalg import norm, inv
import pandas as pd

import qml
from qml.math import cho_solve
from qml.fchl import generate_representation
from qml.fchl import generate_displaced_representations
from qml.fchl import generate_displaced_representations_5point

from qml.fchl import get_local_kernels
from qml.fchl import get_local_symmetric_kernels
from qml.fchl import get_local_gradient_kernels
from qml.fchl import get_local_hessian_kernels
from qml.fchl import get_local_symmetric_hessian_kernels
from qml.fchl import get_local_full_kernels
from qml.fchl import get_local_invariant_alphas
from qml.fchl import get_atomic_gradient_kernels
from qml.fchl import get_smooth_atomic_gradient_kernels
from qml.fchl import get_local_atomic_kernels

SIGMAS = [2.5]
FORCE_KEY = "forces"
ENERGY_KEY = "om2_energy"
CSV_FILE = "data/amons_small.csv"
# CSV_FILE = "data/amons_10_300k.csv"

# SIGMAS = [10.24]
# FORCE_KEY  = "mopac_forces"
# ENERGY_KEY = "mopac_energy"
# CSV_FILE = "data/01.csv"

SIGMAS = [0.01 * 2**i for i in range(20)]

TRAINING = 13
TEST     = 7

DX = 0.005
CUT_DISTANCE = 1e6
KERNEL_ARGS = {
    "cut_distance": CUT_DISTANCE, 
    "alchemy": "off",
    "two_body_width": 0.4,
    "two_body_power": 3.0,
    "three_body_power": 2.0,
    # "kernel": "linear",
    # "kernel": "inverse-multiquadratic",
    # "kernel": "multiquadratic",
    # "kernel": "matern",
    "kernel": "cauchy",
    # "kernel": "gaussian",
    "kernel_args": {
            # "c": SIGMAS,
            # "n": [2.0 for _ in SIGMAS],
            "sigma": SIGMAS,
    },
}

LLAMBDA_ENERGY = 1e-7
LLAMBDA_FORCE  = 1e-7

np.set_printoptions(linewidth=19999999999, suppress=True, edgeitems=10)
np.random.seed(667)

def csv_to_molecular_reps(csv_filename, force_key="orca_forces", energy_key="orca_energy"):

    df = pd.read_csv(csv_filename)

    x = []
    f = []
    e = []
    distance = []

    disp_x = []
    disp_x5 = []

    max_atoms = max([len(ast.literal_eval(df["atomtypes"][i])) for i in range(len(df))])

    print("MAX ATOMS:", max_atoms)

    IDX = np.array(range(len(df)))
    np.random.shuffle(IDX)
    IDX = IDX[:TRAINING+TEST]

    print(IDX)
    for i in IDX:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))

        dist = norm(coordinates[0] - coordinates[1])
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df[force_key][i]))
        energy = df[energy_key][i]

        rep = generate_representation(coordinates, nuclear_charges, 
                max_size=max_atoms, cut_distance=CUT_DISTANCE)
        disp_rep = generate_displaced_representations(coordinates, nuclear_charges, 
                max_size=max_atoms, cut_distance=CUT_DISTANCE, dx=DX)
        
        disp_rep5 = generate_displaced_representations_5point(coordinates, nuclear_charges, 
                max_size=max_atoms, cut_distance=CUT_DISTANCE, dx=DX)

        distance.append(dist)
        x.append(rep)
        f.append(force)
        e.append(energy)

        disp_x.append(disp_rep)
        disp_x5.append(disp_rep5)

    return np.array(x), f, e, distance, np.array(disp_x), np.array(disp_x5)


def test_gp_derivative():

    Xall, Fall, Eall, Dall, dXall, dXall5 = csv_to_molecular_reps(CSV_FILE,
                                        force_key=FORCE_KEY, energy_key=ENERGY_KEY)

    # TRAINING = int(len(Eall) * 0.66)
    # TEST = len(Eall) - TRAINING

    Eall = np.array(Eall)
    Fall = np.array(Fall)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    F  = Fall[:TRAINING]
    E  = Eall[:TRAINING]
    D  = Dall[:TRAINING]
    
    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Fs = Fall[-TEST:]
    Es = Eall[-TEST:]
    Ds = Dall[-TEST:]
    
    K = get_local_full_kernels(X, dX, dx=DX, **KERNEL_ARGS)
    Kt = K[:,TRAINING:,TRAINING:]
    Kt_local = K[:,:TRAINING,:TRAINING]
    Kt_energy = K[:,:TRAINING,TRAINING:]
   
    Kt_grad2 = get_local_gradient_kernels(  X,  dX, dx=DX, **KERNEL_ARGS)


    Ks          = get_local_hessian_kernels(     dX, dXs, dx=DX, **KERNEL_ARGS)
    Ks_energy   = get_local_gradient_kernels(  X,  dXs, dx=DX, **KERNEL_ARGS)
     
    Ks_energy2  = get_local_gradient_kernels(  Xs, dX,  dx=DX, **KERNEL_ARGS)
    Ks_local    = get_local_kernels(           X,  Xs,         **KERNEL_ARGS)
    # exit()
    
    F = np.concatenate(F)
    Fs = np.concatenate(Fs)
    
    Y = np.array(F.flatten())
    Y = np.concatenate((E, Y))

    for i, sigma in enumerate(SIGMAS):

        C = deepcopy(K[i])
        
        for j in range(TRAINING):
            C[j,j] += LLAMBDA_ENERGY

        for j in range(TRAINING,K.shape[2]):
            C[j,j] += LLAMBDA_FORCE

        alpha = cho_solve(C, Y)
        beta = alpha[:TRAINING]
        gamma = alpha[TRAINING:]

        Fss = np.dot(np.transpose(Ks[i]), gamma) + np.dot(np.transpose(Ks_energy[i]), beta)
        Ft  = np.dot(np.transpose(Kt[i]), gamma) + np.dot(np.transpose(Kt_energy[i]), beta)

        Ess = np.dot(Ks_energy2[i], gamma) + np.dot(Ks_local[i].T, beta)
        Et  = np.dot(Kt_energy [i], gamma) + np.dot(Kt_local[i].T, beta)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())
        print("TRAINING ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Et - E)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), Ft.flatten())
        print("TRAINING FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ft.flatten() - F.flatten())), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), Ess.flatten())
        print("TEST     ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ess - Es)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), Fss.flatten())
        print("TEST     FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Fss.flatten() - Fs.flatten())), sigma, slope, intercept, r_value ))

def test_gdml_derivative():

    Xall, Fall, Eall, Dall, dXall, dXall5 = csv_to_molecular_reps(CSV_FILE,
                                        force_key=FORCE_KEY, energy_key=ENERGY_KEY)

    Eall = np.array(Eall)
    Fall = np.array(Fall)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    F  = Fall[:TRAINING]
    E  = Eall[:TRAINING]
    D  = Dall[:TRAINING]
    
    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Fs = Fall[-TEST:]
    Es = Eall[-TEST:]
    Ds = Dall[-TEST:]
    
   
    K  = get_local_symmetric_hessian_kernels(dX, dx=DX, **KERNEL_ARGS)
    Ks = get_local_hessian_kernels(dXs, dX, dx=DX, **KERNEL_ARGS)

    Kt_energy   = get_local_gradient_kernels(  X,  dX, dx=DX, **KERNEL_ARGS)
    Ks_energy   = get_local_gradient_kernels(  Xs,  dX, dx=DX, **KERNEL_ARGS)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)
    
    Y = np.array(F.flatten())
    # Y = np.concatenate((E, Y))

    for i, sigma in enumerate(SIGMAS):

        C = deepcopy(K[i])
        for j in range(K.shape[2]):
            C[j,j] += LLAMBDA_FORCE

        alpha = cho_solve(C, Y)
        Fss = np.dot(Ks[i], alpha)
        Ft  = np.dot(K[i],  alpha)

        Ess = np.dot(Ks_energy[i], alpha)
        Et  = np.dot(Kt_energy[i], alpha)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())
        
        Ess -= intercept
        Et  -= intercept

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())

        print("TRAINING ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Et - E)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), Ft.flatten())
        print("TRAINING FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ft.flatten() - F.flatten())), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), Ess.flatten())
        print("TEST     ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ess - Es)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), Fss.flatten())
        print("TEST     FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Fss.flatten() - Fs.flatten())), sigma, slope, intercept, r_value ))

        
def test_general_derivative():

    Xall, Fall, Eall, Dall, dXall, dXall5 = csv_to_molecular_reps(CSV_FILE,
                                        force_key=FORCE_KEY, energy_key=ENERGY_KEY)

    Eall = np.array(Eall)
    Fall = np.array(Fall)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    dX5 = dXall5[:TRAINING]
    F  = Fall[:TRAINING]
    E  = Eall[:TRAINING]
    D  = Dall[:TRAINING]
    
    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    dXs5 = dXall5[-TEST:]
    Fs = Fall[-TEST:]
    Es = Eall[-TEST:]
    Ds = Dall[-TEST:]
    
    Ftrain = np.concatenate(F)
    Etrain = np.array(E)
    alphas      = get_local_invariant_alphas(X, dX, Ftrain, energy=Etrain, dx=DX, 
                    regularization=LLAMBDA_FORCE, **KERNEL_ARGS)

    Kt_force = get_atomic_gradient_kernels(X,  dX, dx=DX, **KERNEL_ARGS)
    Ks_force = get_atomic_gradient_kernels(X, dXs, dx=DX, **KERNEL_ARGS)
    
    Kt_force5 = get_smooth_atomic_gradient_kernels(X,  dX5, dx=DX, **KERNEL_ARGS)
    Ks_force5 = get_smooth_atomic_gradient_kernels(X, dXs5, dx=DX, **KERNEL_ARGS)

    Kt_energy = get_local_atomic_kernels(X, X,   **KERNEL_ARGS)
    Ks_energy = get_local_atomic_kernels(X, Xs,  **KERNEL_ARGS)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)
    Y = np.array(F.flatten())

    for i, sigma in enumerate(SIGMAS):

        Ft  = np.zeros((Kt_force[i,:,:].shape[1]/3,3))
        Fss = np.zeros((Ks_force[i,:,:].shape[1]/3,3))

        Ft5  = np.zeros((Kt_force5[i,:,:].shape[1]/3,3))
        Fss5 = np.zeros((Ks_force5[i,:,:].shape[1]/3,3))

        for xyz in range(3):
            
            Ft[:,xyz]  = np.dot(Kt_force[i,:,xyz::3].T, alphas[i])
            Fss[:,xyz] = np.dot(Ks_force[i,:,xyz::3].T, alphas[i])
            
            Ft5[:,xyz]  = np.dot(Kt_force5[i,:,xyz::3].T, alphas[i])
            Fss5[:,xyz] = np.dot(Ks_force5[i,:,xyz::3].T, alphas[i])
       
        Ess = np.dot(Ks_energy[i].T, alphas[i])
        Et  = np.dot(Kt_energy[i].T, alphas[i])

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())
        print("TRAINING ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Et - E)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), Ft.flatten())
        print("TRAINING FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
               (np.mean(np.abs(Ft.flatten() - F.flatten())), sigma, slope, intercept, r_value ))
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), Ft5.flatten())
        print("TRAINING FORCE_5  MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
               (np.mean(np.abs(Ft5.flatten() - F.flatten())), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), Ess.flatten())
        print("TEST     ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ess - Es)), sigma, slope, intercept, r_value ))
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), Fss.flatten())
        print("TEST     FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Fss.flatten() - Fs.flatten())), sigma, slope, intercept, r_value ))
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), Fss5.flatten())
        print("TEST     FORCE_5  MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Fss5.flatten() - Fs.flatten())), sigma, slope, intercept, r_value ))

def test_derivative():

    Xall, Fall, Eall, Dall, dXall, dXall5 = csv_to_molecular_reps(CSV_FILE,
                                        force_key=FORCE_KEY, energy_key=ENERGY_KEY)

    Eall = np.array(Eall)
    Fall = np.array(Fall)

    X  = Xall[:TRAINING]
    dX = dXall[:TRAINING]
    F  = Fall[:TRAINING]
    E  = Eall[:TRAINING]
    D  = Dall[:TRAINING]
    
    Xs = Xall[-TEST:]
    dXs = dXall[-TEST:]
    Fs = Fall[-TEST:]
    Es = Eall[-TEST:]
    Ds = Dall[-TEST:]
    
   
    K  = get_local_symmetric_kernels(X, **KERNEL_ARGS)
    Ks = get_local_kernels(Xs, X, **KERNEL_ARGS)

    Kt_force = get_local_gradient_kernels(X, dX,  dx=DX, **KERNEL_ARGS)
    Ks_force = get_local_gradient_kernels(X, dXs, dx=DX, **KERNEL_ARGS)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)
    
    #Y = np.array(F.flatten())
    Y = np.array(E)
    # Y = np.concatenate((E, Y))

    for i, sigma in enumerate(SIGMAS):

        C = deepcopy(K[i])
        for j in range(K.shape[2]):
            C[j,j] += LLAMBDA_ENERGY

        alpha = cho_solve(C, Y)
        
        Fss = np.dot(Ks_force[i].T, alpha)
        Ft  = np.dot(Kt_force[i].T,  alpha)

        Ess = np.dot(Ks[i], alpha)
        Et  = np.dot(K[i], alpha)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E.flatten(), Et.flatten())
        
        print("TRAINING ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Et - E)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), Ft.flatten())
        print("TRAINING FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                 (np.mean(np.abs(Ft.flatten() - F.flatten())), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), Ess.flatten())
        print("TEST     ENERGY   MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Ess - Es)), sigma, slope, intercept, r_value ))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), Fss.flatten())
        print("TEST     FORCE    MAE = %10.4f     sigma = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
                (np.mean(np.abs(Fss.flatten() - Fs.flatten())), sigma, slope, intercept, r_value ))

if __name__ == "__main__":

    t = time.time() 
    test_gp_derivative()
    print("Elapse:", time.time() - t)
    t = time.time() 
    test_gdml_derivative()
    print("Elapse:", time.time() - t)
    t = time.time() 
    test_general_derivative()
    print("Elapse:", time.time() - t)
    t = time.time() 
    test_derivative()
    print("Elapse:", time.time() - t)
