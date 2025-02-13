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

def get_estimate(df):
    #Change labels to fit syntax for estimates dataframe
    l = np.linspace(0, 1, num=len(df.index))
    df.index = l
    df.columns = l
    est = np.round(df.loc[0, 1], decimals = 3)
    return est

def get_MBAR(u_nk, name):
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

    return MBAR_est, MBAR_est_err

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR Divided by Pocket for MT-REXEE')
parser.add_argument('-ln', required=True, nargs='+', type = str, help='Names of Ligand for FE output')
parser.add_argument('-s', required=True, nargs='+', type = str, help='Simulation numbers for the ligands')
parser.add_argument('-f', required=False, default='./', help='base file path storing simulations')
parser.add_argument('-i', required=False, type = int, default = 1, help='Total Number of Iterations')
parser.add_argument('-t', required=False, type = float, default = 300, help='Temperature Simulations Run (K)')
parser.add_argument('-dt', required=False, type = float, default = 2, help='Time Step in fs')

#Save imported arguments to variables
args = parser.parse_args()
lig_names = args.ln
sim_num = args.s
n_iter = args.i
file_path = args.f
temp = args.t
time_step = args.dt

# Perform analysis for all ligand names listed
output_df = pd.DataFrame()
drop_df = pd.DataFrame()
for s in range(len(lig_names)):
    #Determine which iterations are in which pocket
    pocketA, pocketB = [],[]
    track_iter_B = []
    track_frame_drop = []
    u_nk_B = pd.DataFrame()
    u_nk_A = pd.DataFrame()
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

        B_frame, A_frame = [],[]

        for t in range(traj.n_frames):
            if dist[t,0] < 1.0 and dist[t,1] < 0.85:
                B_frame.append(t)
            elif dist[t,0] > 1.0 and dist[t,1] > 0.85:
                A_frame.append(t)
        #Obtain u_nk reduced potentials
        full_u_nk = concat([extract_u_nk(xvg, T=300) for xvg in f'{file_path}/sim_{sim_num[s]}/iteration_{i}/dhdl.xvg'])
        freq_dhdl = full_u_nk['time'].values[1]

        #Extract frames in each pocket
        B_time, A_time = [], []
        num_frame_drop = 0
        for t in range(traj.n_frames-1):
            if t in B_frame and t+1 in B_frame:
                for x in np.arange(t*time_step, (t+1)*time_step, freq_dhdl):
                    B_time.append(x)
            elif t in A_frame and t+1 in A_frame:
                for x in np.arange(t*time_step, (t+1)*time_step, freq_dhdl):
                    A_time.append(x)
            else:
                num_frame_drop += time_step/freq_dhdl
        track_frame_drop.append(num_frame_drop)

        
        if len(B_time) != 0:
            u_nk_sele_B = full_u_nk[full_u_nk['time'].isin(B_time)]
            u_nk_B = pd.concat([u_nk_B, u_nk_sele_B])
        if len(A_time) != 0:
            u_nk_sele_A = full_u_nk[full_u_nk['time'].isin(A_time)]
            u_nk_A = pd.concat([u_nk_B, u_nk_sele_B])
    pA_est, pA_err = get_MBAR(u_nk_A, f'{lig_names[s]}_pocketA')
    pB_est, pB_err = get_MBAR(u_nk_B, f'{lig_names[s]}_pocketB')
    
    df = pd.DataFrame({'Ligand': lig_names[s], 'Free Energy Estimate': [pA_est, pB_est], 'Free Energy Error': [pA_err, pB_err], 'Pocket': ['A', 'B']})
    output_df - pd.concat([output_df, df])

    df = pd.DataFrame({'Ligand': lig_names[s], '# of Dropped Frames': track_frame_drop})
    drop_df = pd.concat([drop_df, df])
output_df.to_csv('FE_estimate.csv')
drop_df.to_csv('Frames_dropped_FE_by_frame.csv')

