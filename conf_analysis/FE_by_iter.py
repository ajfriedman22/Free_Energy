import numpy as np
from alchemlyb.parsing.gmx import extract_u_nk
import pandas as pd
from alchemlyb.estimators import MBAR
from alchemlyb import concat
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_convergence
from alchemlyb.preprocessing.subsampling import statistical_inefficiency
from alchemlyb.postprocessors.units import to_kcalmol
import argparse
import mdtraj as md

def convergence(name, lambda_list):
    data_list = [statistical_inefficiency(extract_u_nk(xvg, T=temp)) for xvg in lambda_list]
    est = MBAR()
            
    forward, forward_error, backward, backward_error = [], [], [], []
    num_points = 10
    for i in range(1, num_points+1):
        # Do the forward
        slice = int(len(data_list[0])/num_points*i)
        data = concat([data[:slice] for data in data_list])
        estimate = est.fit(data)
        #Add estimates
        forward.append(get_estimate(estimate.delta_f_))
        forward_error.append(get_estimate(estimate.d_delta_f_))
        # Do the backward
        data = concat([data[-slice:] for data in data_list])
        estimate = est.fit(data)
        #Add estimates
        backward.append(get_estimate(estimate.delta_f_))
        backward_error.append(get_estimate(estimate.d_delta_f_))
    df = pd.DataFrame({'Forward': forward, 'Forward_Error': forward_error, 'Backward': backward, 'Backward Error': backward_error})
    df.attrs['temperature'] = temp
    df.attrs['energy_unit'] = 'kT'
    ax = plot_convergence(df, units='kcal/mol')
    ax.figure.savefig(f'Convergence_{name}.png')

def get_estimate(df):
    #Change labels to fit syntax for estimates dataframe
    l = np.linspace(0, 1, num=len(df.index))
    df.index = l
    df.columns = l
    est = np.round(df.loc[0, 1], decimals = 3)
    return est

def get_MBAR(lambda_list, name):
    #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([extract_u_nk(xvg, T=300) for xvg in lambda_list])

    #MBAR FE Estimates
    mbar = MBAR(relative_tolerance=1e-04).fit(u_nk)

    #Convert Units to Kcal/mol
    mbar_kcal = to_kcalmol(mbar.delta_f_, T=temp)
    dmbar_kcal = to_kcalmol(mbar.d_delta_f_, T=temp)

    #Get estimates from df
    MBAR_est = get_estimate(mbar_kcal)
    MBAR_est_err = get_estimate(dmbar_kcal)

    #Plot MBAR Overlap Matrix
    ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
    ax.figure.savefig(f'Overlap_{name}.png', bbox_inches='tight', pad_inches=0.0)

    #Forward and Backward Convergence for Coulomb MBAR
    #convergence(name, lambda_list)

    return MBAR_est, MBAR_est_err

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR Divided by Pocket for MT-REXEE')
parser.add_argument('-ln', required=True, nargs='+', type = str, help='Names of Ligand for FE output')
parser.add_argument('-s', required=True, nargs='+', type = str, help='Simulation numbers for the ligands')
parser.add_argument('-f', required=False, default='./', help='base file path storing simulations')
parser.add_argument('-i', required=False, type = int, default = 1, help='Total Number of Iterations')
parser.add_argument('-t', required=False, type = float, default = 300, help='Temperature Simulations Run (K)')

#Save imported arguments to variables
args = parser.parse_args()
lig_names = args.ln
sim_num = args.s
n_iter = args.i
file_path = args.f
temp = args.t

# Perform analysis for all ligand names listed
output_df = pd.DataFrame()
for s in range(len(lig_names)):
    #Determine which iterations are in which pocket
    pocketA, pocketB = [],[]
    track_iter_B = []
    for i in range(n_iter):
        traj = md.load(f'{file_path}/sim_{sim_num[s]}/iteration_{i}/traj.trr', top=f'{file_path}/{lig_names[s]}.gro')
        traj.remove_solvent()

        #Select atom pairs
        c4 = traj.topology.select('name C4')
        if len(c4) == 0:
            c4 = traj.topology.select('name DC4')
        check1 = traj.topology.select('resid 271 and name CD')
        check2 = traj.topology.select('resid 299 and name CB')

        dist = md.compute_distances(traj, [[c4[0], check1[0]], [c4[0], check2[0]]])

        pocketB_both = 0
        for t in range(traj.n_frames):
            if dist[t,0] < 1.0 and dist[t,1] < 0.85:
                pocketB_both += 1

        per = pocketB_both/traj.n_frames
        if per > 0.90:
            pocketB.append(f'{file_path}/sim_{s}/iteration_{i}/dhdl.xvg')
            track_iter_B.append(i)
        else:
            pocketA.append(f'{file_path}/sim_{s}/iteration_{i}/dhdl.xvg')
    np.savetxt(f'pockeB_{lig_names[s]}.txt', np.array(track_iter_B, dtype=float))

    pA_est, pA_err = get_MBAR(pocketA, f'{lig_names[s]}_pocketA')
    pB_est, pB_err = get_MBAR(pocketB, f'{lig_names[s]}_pocketB')
    
    df = pd.DataFrame({'Ligand': lig_names[s], 'Free Energy Estimate': [pA_est, pB_est], 'Free Energy Error': [pA_err, pB_err], 'Pocket': ['A', 'B']})
    output_df - pd.concat([output_df, df])
output_df.to_csv('FE_estimate.csv')
