#!.env/bin/python -u
import os
import sys
import glob
import time
import h5py
import random
import argparse
import datetime

import numpy as np
import numpy.linalg as nla
import matplotlib.pyplot as plt

from functools import reduce, partial
from multiprocessing import Pool
from experiment_utils import normalize,denormalize

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2 # 0.1
TEST_RATIO = 0 # 0.1

ORD = "F"
STATES = [
        'Vx', 'Vy', 'Vz', 'density', 'pressure'
    ]
MACHINE_ALIAS = "LH"

__all__ = [
    'get_args',
    'initialize_wandb',
    'read_snapshot',
]

def read_snapshot(timestep,simulation,states,data_loc):
    return np.load(data_loc+f'sim{simulation:03d}_ts{timestep:02d}.npy')[:,:,:,states][...,None]

def read_simulations(simulation_idx, states, timesteps = None, path_to_data ="./", file_name=""):
    sim = []
    # print(path_to_data,file_name, simulation_idx, timesteps)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as f:
        for state in states:
            # print(state, f[state].shape)
            if timesteps is None:
                ## Read all timesteps
                sim.append(f[state][simulation_idx,:,...].transpose(1,2,3,0)[...,None])
            else:
                ## Read only the specified timestep(s)
                sim.append(f[state][simulation_idx,timesteps,...][...,None])
    
    return np.concatenate(sim,axis=-1)[...,None]

def load_simulation(simulation_idx, data_loc, num_timesteps = 1, states=list(range(len(STATES))), sim_shape=None):
    if sim_shape is None:
        timestep = 0
        sim_shape = list(np.load(data_loc+f'sim{simulation_idx:03d}_ts{timestep:02d}.npy')[:,:,:,states][...,None].shape)
        sim_shape[-1] = num_timesteps
    sim = np.zeros(sim_shape)
    for timestep in range(num_timesteps):
        sim[...,timestep] = read_snapshot(timestep,simulation_idx,states,data_loc).squeeze(-1)
    return sim

def compute_energy_spectrum(data):
    """
    Compute the energy spectrum of the flow from the velocity field.

    Parameters:
    data : numpy array
        Simulation data with shape (nx, ny, nz, variables, time_steps).
        Variables should include the velocity components (u, v, w) in the first three indices.

    Returns:
    spectrum : numpy array
        Energy spectrum for each time step.
    """
    nx, ny, nz, nvars, nsteps = data.shape
    assert nvars >= 3, "Data should contain at least three variables (u, v, w)."

    # Extract velocity components
    u = data[..., 0, :]
    v = data[..., 1, :]
    w = data[..., 2, :]

    # Initialize the spectrum array
    spectrum = np.zeros((nx, nsteps))
    max_k = np.sqrt((nx/2)**2 + (ny/2)**2 + (nz/2)**2)
    k_bins = np.linspace(0, max_k, nx//2)
    spectrum = np.zeros((len(k_bins) - 1, nsteps))
    for t in range(nsteps):
        # Compute the Fourier transform of each velocity component
        u_hat = np.fft.fftn(u[..., t])
        v_hat = np.fft.fftn(v[..., t])
        w_hat = np.fft.fftn(w[..., t])

        # Compute the energy spectrum
        energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

        # Compute the wavenumber magnitude
        kx = np.fft.fftfreq(nx).reshape(nx, 1, 1)
        ky = np.fft.fftfreq(ny).reshape(1, ny, 1)
        kz = np.fft.fftfreq(nz).reshape(1, 1, nz)
        k_squared = kx**2 + ky**2 + kz**2
        k_magnitude = np.sqrt(k_squared)
        # print(k_magnitude)

        # Bin the energy into shells based on wavenumber magnitude
        
        # k_bins = np.arange(1, nx // 2)
        # print(k_bins)
        energy_spectrum, _ = np.histogram(k_magnitude.ravel(), bins=k_bins, weights=energy.ravel())
        print(len(energy_spectrum),len(k_bins))
        # Store the spectrum
        spectrum[:len(energy_spectrum), t] = energy_spectrum

    return spectrum

def compute_energy_spectrum_new(data, L=1.0):
    """
    Compute the energy spectrum of the flow from the velocity field.

    Parameters:
    data : numpy array
        Simulation data with shape (nx, ny, nz, variables, time_steps).
        Variables should include the velocity components (u, v, w) in the first three indices.
    L : float
        Physical size of the simulation domain.

    Returns:
    avg_spectrum : numpy array
        Averaged energy spectrum across all time steps.
    k : numpy array
        Wavenumbers corresponding to the energy spectrum.
    """
    nx, ny, nz, nvars, nsteps = data.shape
    assert nvars >= 3, "Data should contain at least three variables (u, v, w)."

    # Extract velocity components
    u = data[..., 0, :]
    v = data[..., 1, :]
    w = data[..., 2, :]

    # Define wavenumber grid
    dim = nx  # assuming cubic grid
    k_end = int(dim / 2)
    rx = np.array(range(dim)) - dim / 2 + 1
    rx = np.roll(rx, int(dim / 2) + 1)

    X, Y, Z = np.meshgrid(rx, rx, rx)
    r = np.sqrt(X**2 + Y**2 + Z**2)

    dx = 2 * np.pi / L
    k = (np.array(range(k_end)) + 1) * dx

    # Initialize spectrum and bins
    bins = np.zeros((k.shape[0] + 1))
    for N in range(k_end):
        if N == 0:
            bins[N] = 0
        else:
            bins[N] = (k[N] + k[N - 1]) / 2
    bins[-1] = k[-1]

    spectrum_time = np.zeros((k.shape[0], nsteps))

    # Compute the spectrum for each time step
    for t in range(nsteps):
        # Compute the Fourier transform of each velocity component
        u_hat = np.fft.fftn(u[..., t])
        v_hat = np.fft.fftn(v[..., t])
        w_hat = np.fft.fftn(w[..., t])

        # Compute energy in spectral space
        muu = np.abs(u_hat)**2
        mvv = np.abs(v_hat)**2
        mww = np.abs(w_hat)**2

        # Digitize the wavenumbers into bins
        inds = np.digitize(r * dx, bins)
        spectrum = np.zeros((k.shape[0]))
        bin_counter = np.zeros((k.shape[0]))

        # Calculate the energy spectrum
        for N in range(k_end):
            spectrum[N] = (
                np.sum(muu[inds == N + 1])
                + np.sum(mvv[inds == N + 1])
                + np.sum(mww[inds == N + 1])
            )
            bin_counter[N] = np.count_nonzero(inds == N + 1)

        # Normalize the spectrum
        spectrum = spectrum * 2 * np.pi * (k**2) / (bin_counter * dx**3)
        spectrum_time[:, t] = spectrum

    # Average the spectrum over all time steps
    # avg_spectrum = np.mean(spectrum_time, axis=1)

    return spectrum_time, k


def plot_energy_spectrum(spectrum, timestep=None):
    """Plot the energy spectrum for a specific time step."""
    plt.figure(figsize=(8, 6))
    if timestep is None:
        plt.loglog(np.mean(spectrum, axis=1))
        plt.title(f'Time Averaged Energy Spectrum')
    else:
        plt.loglog(spectrum[:, timestep])
        plt.title(f'Energy Spectrum at Time Step {timestep}')
    plt.xlabel('Wavenumber')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.savefig('spectrum.png')

def plot_energy_spectrum_new(spectrum, k, timestep=None, label =''):
    """Plot the averaged energy spectrum over all time steps."""
    plt.figure(figsize=(8, 6))
    if timestep is None:
        avg_spectrum = np.mean(spectrum, axis=1)
        print(avg_spectrum.shape,spectrum.shape,k.shape)
        plt.loglog(k,avg_spectrum,label=label)
        plt.title(f'Time Averaged Energy Spectrum')
    else:
        plt.loglog(k, spectrum[:, timestep],label=label)
        # plt.loglog(spectrum[:, timestep])
        plt.title(f'Energy Spectrum at Time Step {timestep}')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    plt.savefig('spectrum.png')
    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='This script compares the energy spectrum of tensor network compressed NS simulations with turbulent ICs from the PDEBench dataset')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, help='epsilon value', default=0.1)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('--states', dest='states', nargs='+', type=int, help='Determines the states that will be compressed', default=list(range(len(STATES))))
    # parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    parser.add_argument('-t', '--type', dest='type', type=str, help='Type of simulation data', default="Rand")
    parser.add_argument('-c', '--combine', dest='combine', action='store_true', help='Combine timesteps of the simulation', default=False)
    parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Use extracted numpy files to read data' , default=False)
    parser.add_argument('-m', '--method', dest='method', type=str, help='Method of compression' , default=None)
    parser.add_argument('-M', '--mach_number', dest='M', help='Mach number for the simulations', default=None)
    parser.add_argument('-p', '--print', action='store_true', default=False, help='Prints the energy spectrum to a .npy file.')
    parser.add_argument('-N', '--normalization', dest='normalization', type=str, help='Normalization method used', default='none')
    return parser.parse_args()

def energy_spectrum_ht(args):
    import htucker as ht
    sys.path.insert(0,'/home/doruk/TT-ICE/')
    import DaMAT as dmt

    np.set_printoptions(linewidth=100)
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    num_states = len(args.states)
    data_loc = args.data_location+PATH_SEP+f'{args.type}'+PATH_SEP+f'{args.M}'+PATH_SEP
    compression_loc = '/nfs/turbo/coe-goroda/doruk_temp/'
    print(data_loc)
    num_simulations = len(glob.glob(data_loc+'sim*_ts00.npy'))
    num_timesteps = len(glob.glob(data_loc+'sim000_ts*.npy'))
    grid_shape = list(np.load(data_loc+'sim000_ts00.npy').shape[:-1])
    print(num_simulations,num_timesteps,grid_shape)
    # assert np.prod(grid_shape) == np.prod(args.reshaping), "Reshaping does not match the shape of the data"

    epsilon_str = "".join(f"{args.epsilon:0.2f}".split("."))
    compression_name = f'PDEBench3_{args.normalization}_e{epsilon_str}.hto'

    train_simulations = random.sample(range(0, num_simulations), int(num_simulations*TRAIN_RATIO))
    val_simulations = random.sample(list(set(range(0, num_simulations))-set(train_simulations)), int(num_simulations*VAL_RATIO))

    dataset = ht.HTucker.load(compression_name,directory=compression_loc)
    
    # anan = read_snapshot(0,train_simulations[0],STATES,data_loc)
    # print(anan.shape)

    # if TEST_RATIO == 0:
    #     test_simulations =[]
    # else:
    #     test_simulations = list(set(range(0, num_simulations))-set(train_simulations)-set(val_simulations))

    # sim = read_snapshot(1,train_simulations[0],args.states,data_loc)
    # print(sim.shape)
    # with Pool() as p:
    #     sim  = p.map(partial(read_snapshot,data_loc=data_loc,simulation=1,states=args.states),list(range(num_timesteps)))
    # sim = np.concatenate(sim,axis=-1)
    # print(sim.shape)
    print(dataset.root.shape)
    rel_errs = []
    print(len(dataset.normalizing_constants))
    print()
    selected_sim = 6
    test_sim_idx = 89

    # for sim_idx in range(selected_sim,selected_sim+1):
    # # for sim_idx in range(len(train_simulations)):
    #     print(sim_idx)
    #     sim = load_simulation(train_simulations[sim_idx],data_loc,num_timesteps,states=args.states)
    #     for state_idx in range(len(args.states)):
    #         print(sim[...,state_idx,:].min(),sim[...,state_idx,:].max())
    #     root_slice = dataset.root.core[...,sim_idx]
    #     # rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    #     rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    #     print(sim.shape,rec_sim.shape)
    #     # print(rec_sim.shape)
    #     for state_idx in range(len(args.states)):
    #         # print(state_idx, dataset.normalizing_constants[sim_idx])
    #         normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
    #         rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    #     # print(dataset.normalizing_constants[sim_idx])
    #     sim_norm = nla.norm(sim)
    #     rel_errs.append(nla.norm(sim-rec_sim)/sim_norm)
    #     # print(sim.shape,rec_sim.shape)
    # print(rel_errs)

    # sim_idx = np.argmin(rel_errs)
    sim_idx = selected_sim
    print(sim_idx)

    sim = load_simulation(train_simulations[sim_idx],data_loc,num_timesteps,states=args.states)
    root_slice = dataset.root.core[...,sim_idx]
    rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    # print(sim.shape,rec_sim.shape)
    # print(rec_sim.shape)
    for state_idx in range(rec_sim.shape[-2]):
        normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
        rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    # print(dataset.normalizing_constants[sim_idx])
    sim_norm = nla.norm(sim)
    rel_errs.append(nla.norm(sim-rec_sim)/sim_norm)
    print(f"HT approximation error: {round(nla.norm(sim-rec_sim)/sim_norm,4)}")
    # print(sim.shape,rec_sim.shape)


    rec_spec, k_rec = compute_energy_spectrum_new(rec_sim, L=2*np.pi)
    spec, k = compute_energy_spectrum_new(sim, L=2*np.pi)
    ta_rec_spec = np.mean(rec_spec,axis=1)
    ta_spec = np.mean(spec,axis=1)
    print("Time averaged original spectrum")
    print(ta_spec)
    print(f"Time averaged reconstructed spectrum (HT) {epsilon_str}")
    print(ta_rec_spec)
    

    plt.figure(0,figsize=(8, 6))
    # print(avg_spectrum.shape,spectrum.shape,k.shape)
    plt.semilogy(k_rec,ta_rec_spec,label='HT')
    plt.semilogy(k,ta_spec,label='original')
    plt.title(f'Time Averaged Energy Spectrum w/ {args.epsilon*100}% Rel.Err. ({round(np.prod(sim.shape)/np.prod(dataset.root.shape[:-1]),3)}x compression)')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    # plt.legend()
    # plt.savefig(f'spectrum_{epsilon_str}.png')
    # plt.close(0)
    
    
    test_sim = load_simulation(val_simulations[test_sim_idx],data_loc,num_timesteps,states=args.states)
    root_slice = dataset.project(test_sim.reshape(args.reshaping+[num_timesteps],order=ORD))
    rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    # print(sim.shape,rec_sim.shape)
    # print(rec_sim.shape)
    for state_idx in range(rec_sim.shape[-2]):
        normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
        rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    # print(dataset.normalizing_constants[sim_idx])
    sim_norm = nla.norm(test_sim)
    # rel_errs.append(nla.norm(test_sim-rec_sim)/sim_norm)
    # print(nla.norm(test_sim-rec_sim)/sim_norm)
    print(f"HT approximation error: {round(nla.norm(test_sim-rec_sim)/sim_norm,4)}")
    # print(sim.shape,rec_sim.shape)


    rec_spec, k_rec = compute_energy_spectrum_new(rec_sim, L=2*np.pi)
    test_spec, k = compute_energy_spectrum_new(test_sim, L=2*np.pi)
    ta_rec_spec = np.mean(rec_spec,axis=1)
    ta_test_spec = np.mean(test_spec,axis=1)

    print("Time averaged original spectrum (unseen)")
    print(ta_test_spec)
    print(f"Time averaged reconstructed spectrum (unseen) (HT) {epsilon_str}")
    print(ta_rec_spec)

    plt.figure(1,figsize=(8, 6))
    # print(avg_spectrum.shape,spectrum.shape,k.shape)
    # plt.semilogy(k_rec,np.ones_like(k_rec),label='reconstructed')
    plt.semilogy(k_rec,ta_rec_spec,label='HT')
    plt.semilogy(k,ta_test_spec,label='original')
    plt.title(f'Time Averaged Energy Spectrum w/ {args.epsilon*100}% Rel.Err. ({round(np.prod(sim.shape)/np.prod(dataset.root.shape[:-1]),3)}x compression)')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'spectrum_test_{epsilon_str}.png')
    # plt.close(1)

    



    sim = load_simulation(train_simulations[sim_idx],data_loc,num_timesteps,states=args.states)
    root_slice = dataset.project(sim.reshape(args.reshaping+[num_timesteps],order=ORD))
    rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    # print(sim.shape,rec_sim.shape)
    # print(rec_sim.shape)
    for state_idx in range(rec_sim.shape[-2]):
        normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
        rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    # print(dataset.normalizing_constants[sim_idx])
    sim_norm = nla.norm(sim)
    rel_errs.append(nla.norm(sim-rec_sim)/sim_norm)
    print(nla.norm(sim-rec_sim)/sim_norm)
    # print(sim.shape,rec_sim.shape)


    rec_spec, k_rec = compute_energy_spectrum_new(rec_sim, L=2*np.pi)
    spec, k = compute_energy_spectrum_new(sim, L=2*np.pi)
    ta_rec_spec = np.mean(rec_spec,axis=1)
    ta_spec = np.mean(spec,axis=1)

    plt.figure(2,figsize=(8, 6))
    # print(avg_spectrum.shape,spectrum.shape,k.shape)
    # plt.semilogy(k_rec,np.ones_like(k_rec),label='reconstructed')
    plt.semilogy(k_rec,ta_rec_spec,label='HT')
    plt.semilogy(k,ta_spec,label='original')
    plt.title(f'Time Averaged Energy Spectrum w/ {args.epsilon*100}% Rel.Err. ({round(np.prod(sim.shape)/np.prod(dataset.root.shape[:-1]),3)}x compression)')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'spectrum_{epsilon_str}_projection.png')
    # plt.close(2)

    
    dataset = dmt.ttObject.loadData(compression_loc+f'PDEBench3_{args.normalization}_e{epsilon_str}.ttc')

    sim = load_simulation(train_simulations[sim_idx],data_loc,num_timesteps,states=args.states)
    root_slice = dataset.ttCores[-1][:,sim_idx,:]
    rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    # print(sim.shape,rec_sim.shape)
    # print(rec_sim.shape)
    for state_idx in range(rec_sim.shape[-2]):
        normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
        rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    # print(dataset.normalizing_constants[sim_idx])
    sim_norm = nla.norm(sim)
    # rel_errs.append(nla.norm(sim-rec_sim)/sim_norm)
    print(f'TT rec err: {round(nla.norm(sim-rec_sim)/sim_norm,5)}')
    # print(sim.shape,rec_sim.shape)


    rec_spec, k_rec = compute_energy_spectrum_new(rec_sim, L=2*np.pi)
    spec, k = compute_energy_spectrum_new(sim, L=2*np.pi)
    ta_rec_spec = np.mean(rec_spec,axis=1)
    ta_spec = np.mean(spec,axis=1)

    print("Time averaged original spectrum")
    print(ta_spec)
    print(f"Time averaged reconstructed spectrum (TT) {epsilon_str}")
    print(ta_rec_spec)
    

    plt.figure(0,figsize=(8, 6))
    # print(avg_spectrum.shape,spectrum.shape,k.shape)
    plt.semilogy(k_rec,ta_rec_spec,label=fr'TT (${round(dataset.compressionRatio,3)}\times$ compression)')
    # plt.semilogy(k,ta_spec,label='original')
    plt.title(f'Time Averaged Energy Spectrum w/ {args.epsilon*100}% Rel.Err. ({round(np.prod(sim.shape)/np.prod(dataset.ttRanks[-1]),3)}x compression)')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'spectrum_{epsilon_str}.png')
    plt.close(0)



    test_sim = load_simulation(val_simulations[test_sim_idx],data_loc,num_timesteps,states=args.states)
    root_slice = dataset.projectTensor(test_sim.reshape(args.reshaping+[num_timesteps],order=ORD))
    rec_sim = dataset.reconstruct(root_slice).reshape(grid_shape+[num_states,num_timesteps],order=ORD)
    # print(sim.shape,rec_sim.shape)
    # print(rec_sim.shape)
    for state_idx in range(rec_sim.shape[-2]):
        normalizing_constant1, normalizing_constant2 = dataset.normalizing_constants[0][state_idx]
        rec_sim[...,state_idx,:] = denormalize(rec_sim[...,state_idx,:], args.normalization, normalizing_constant1,normalizing_constant2)
    # print(dataset.normalizing_constants[sim_idx])
    sim_norm = nla.norm(test_sim)
    # rel_errs.append(nla.norm(test_sim-rec_sim)/sim_norm)
    # print(nla.norm(test_sim-rec_sim)/sim_norm)
    print(f'TT test err: {round(nla.norm(test_sim-rec_sim)/sim_norm,5)}')
    # print(sim.shape,rec_sim.shape)


    rec_spec, k_rec = compute_energy_spectrum_new(rec_sim, L=2*np.pi)
    test_spec, k = compute_energy_spectrum_new(test_sim, L=2*np.pi)
    ta_rec_spec = np.mean(rec_spec,axis=1)
    ta_test_spec = np.mean(test_spec,axis=1)

    print("Time averaged original spectrum (unseen)")
    print(ta_test_spec)
    print(f"Time averaged reconstructed spectrum (unseen) (TT) {epsilon_str}")
    print(ta_rec_spec)

    plt.figure(1,figsize=(8, 6))
    # print(avg_spectrum.shape,spectrum.shape,k.shape)
    # plt.semilogy(k_rec,np.ones_like(k_rec),label='reconstructed')
    plt.semilogy(k_rec,ta_rec_spec,label='TT')
    # plt.semilogy(k,ta_test_spec,label='original')
    plt.title(f'Time Averaged Energy Spectrum w/ {args.epsilon*100}% Rel.Err. ({round(np.prod(sim.shape)/np.prod(dataset.ttRanks[-1]),3)}x compression)')
    # plt.loglog(k, spectrum)
    plt.xlabel('Wavenumber')
    plt.ylabel('Average Energy')
    # plt.title('Averaged Energy Spectrum Over All Time Steps')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'spectrum_test_{epsilon_str}.png')
    plt.close(1)


    # print(sim.shape)
    # print(sim.min(), sim.max())

    # spectrum = compute_energy_spectrum(sim)
    # print(spectrum.min(), spectrum.max())
    # print()
    # print(spectrum.shape)
    # plot_energy_spectrum(spectrum)


    # spectrum,k = compute_energy_spectrum_new(sim, L=2*np.pi)
    # print(spectrum.min(), spectrum.max())
    # print()
    # print(spectrum.shape,k)
    # # print(spectrum)
    # plot_energy_spectrum_new(spectrum,k)


    # train_batch = []
    # # for sim_idx, simulation in enumerate(train_simulations[:args.batch_size]):
    # normalizing_constants_batch = []
    # for simulation in train_simulations[:args.batch_size]:
    #     normalizing_constants_sim = []
    #     with Pool() as p:
    #         sim  = p.map(partial(read_snapshot,data_loc=data_loc,simulation=simulation,states=args.states),list(range(num_timesteps)))
    #     sim = np.concatenate(sim,axis=-1)
    #     for state_idx in range(num_states):
    #         sim[...,state_idx,:], normalizing_constant1, normalizing_constant2 = normalize(sim[...,state_idx,:],method = args.normalization)
    #         normalizing_constants_sim.append(
    #         np.array([normalizing_constant1,normalizing_constant2])[..., None]
    #     )
    #     # train_batch.append(np.concatenate(sim,axis=-1).transpose(0,1,2,4,3)[...,None])
    #     train_batch.append(sim[...,None])
    #     normalizing_constants_batch.append(np.concatenate(normalizing_constants_sim,axis=-1).T)
    # train_batch = np.concatenate(train_batch,axis=-1)
    # print(train_batch.shape)

    
    
    # print(len(train_simulations),len(val_simulations))


def energy_spectrum_tt(args):
    raise NotImplementedError



if __name__ == '__main__':
    args = get_args()
    
    if args.type in  ["rand", "random", "randomized", "Rand", "RAND", "Random"]:
        args.type = "Rand"
    elif args.type in ["turb", "turbulence", "turbulent", "Turb", "TURB", "Turbulence", "Turbulent"]:
        args.type = "Turb"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if (args.type == "Rand") and (args.M is None):
        print(f'No M is given. Using default M=1.0')
        args.M = "1.0"
    elif (args.type == "Turb") and (args.M is None):
        args.M = "1.0"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass
    
    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for PDEBench")
        if args.type == "Rand":
            # args.reshaping = [8,4,4,8,4,4,8,4,4]
            args.reshaping = [8,4,4,8,4,4,8,4,4]
        elif args.type == "Turb":
            # args.reshaping = [8,8,8,8,8,8]
            args.reshaping = [8,8,8,8,8,8]
    args.reshaping.extend([sum(args.states)])
    print("Reshaping used: ", args.reshaping)


    print(args)

    if args.method is None:
        raise ValueError('Please specify the compression type.')
    elif args.method == 'ht':
        energy_spectrum_ht(args)
    elif args.method == 'tt':
        energy_spectrum_tt(args)
    else:
        raise ValueError(f'Method {args.method} is not defined!')


