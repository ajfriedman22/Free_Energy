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
from tqdm import tqdm

def get_estimate(df):
    #Change labels to fit syntax for estimates dataframe
    l = np.linspace(0, 1, num=len(df.index))
    df.index = l
    df.columns = l
    est = np.round(df.loc[0, 1], decimals = 3)
    return est

def get_MBAR(u_nk, name, num_points):
    #MBAR FE Estimates
    try:
        mbar = MBAR(relative_tolerance=1e-04).fit(u_nk)
    except:
        return None, None, None, None

    #Convert Units to Kcal/mol
    mbar_kcal = to_kcalmol(mbar.delta_f_, T=temp)
    dmbar_kcal = to_kcalmol(mbar.d_delta_f_, T=temp)

    #Get estimates from df
    MBAR_est = get_estimate(mbar_kcal)
    MBAR_est_err = get_estimate(dmbar_kcal)

    #Plot MBAR Overlap Matrix
    ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
    ax.figure.savefig(f'Overlap_{name}.png', bbox_inches='tight', pad_inches=0.0)
    
    #Get the estimates over time
    forward, forward_error = [], []
    start=0
    for i in range(1, num_points+1):
        end = int((len(u_nk.index)/num_points)*i)
        try:
            # Do the forward
            data = u_nk.iloc[:end]
            estimate = MBAR(relative_tolerance=1e-04).fit(data)
            #Add estimates
            forward.append(get_estimate(estimate.delta_f_))
            forward_error.append(get_estimate(estimate.d_delta_f_))

        except:
            forward.append(None)
            forward_error.append(None)
        start=end

    return MBAR_est, MBAR_est_err, forward, forward_error

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
time_df = pd.DataFrame()
for s in range(len(lig_names)):
    #Determine which iterations are in which pocket
    pocketA, pocketB = [],[]
    track_iter_B = []
    track_frame_drop = []
    u_nk_B = pd.DataFrame()
    u_nk_A = pd.DataFrame()
    for i in tqdm(range(n_iter)):
        traj = md.load(f'{file_path}/sim_{sim_num[s]}/iteration_{i}/traj.trr', top=f'{file_path}/{lig_names[s]}.gro')
        traj.remove_solvent()
        
        if i == 0:
            # Determine trajectory length
            traj_length = int(time_step*n_iter*traj.n_frames/1000)
            num_points = int(traj_length/2)-1

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
        full_u_nk = extract_u_nk(f'{file_path}/sim_{sim_num[s]}/iteration_{i}/dhdl.xvg', T=300)
        freq_dhdl = float(full_u_nk.index.get_level_values('time').values[1])

        #Extract frames in each pocket
        B_time, A_time = [], []
        num_frame_drop = 0
        for t in range(traj.n_frames-1):
            if t in B_frame and t+1 in B_frame:
                for x in np.arange(t*time_step, (t+1)*time_step, freq_dhdl):
                    B_time.append(np.round(x, 1))
            elif t in A_frame and t+1 in A_frame:
                for x in np.arange(t*time_step, (t+1)*time_step, freq_dhdl):
                    A_time.append(np.round(x, 1))
            elif t in B_frame:
                for x in np.arange((t-(time_step/4))*time_step, (t+(time_step/4))*time_step, freq_dhdl):
                    B_time.append(np.round(x, 1))
            else:
                num_frame_drop += time_step/freq_dhdl
        
        track_frame_drop.append(num_frame_drop)
        if len(B_time) != 0:
            u_nk_sele_B = full_u_nk[full_u_nk.index.get_level_values('time').isin(B_time)]
            u_nk_B = pd.concat([u_nk_B, u_nk_sele_B])
        if len(A_time) != 0:
            u_nk_sele_A = full_u_nk[full_u_nk.index.get_level_values('time').isin(A_time)]
            u_nk_A = pd.concat([u_nk_A, u_nk_sele_A])
    
    pA_est, pA_err, pA_est_time, pA_err_time = get_MBAR(u_nk_A, f'{lig_names[s]}_pocketA', num_points)
    pB_est, pB_err, pB_est_time, pB_err_time = get_MBAR(u_nk_B, f'{lig_names[s]}_pocketB', num_points)
    
    df = pd.DataFrame({'Ligand': lig_names[s], 'Free Energy Estimate': [pA_est, pB_est], 'Free Energy Error': [pA_err, pB_err], 'Pocket': ['A', 'B']})
    output_df = pd.concat([output_df, df])

    df = pd.DataFrame({'Ligand': lig_names[s], '# of Dropped Frames': track_frame_drop})
    drop_df = pd.concat([drop_df, df])

    df1 = pd.DataFrame({'Ligand': lig_name[s],  'Free Energy Estimate': pA_est_time, 'Free Energy Error': pA_err_time, 'Pocket': 'A', 'Time': np.arange(2, int(traj_length/(num_points+1))*(num_points+1), step=2)})
    df2 = pd.DataFrame({'Ligand': lig_name[s],  'Free Energy Estimate': pB_est_time, 'Free Energy Error': pB_err_time, 'Pocket': 'B', 'Time': np.arange(2, int(traj_length/(num_points+1))*(num_points+1), step=2)})
    time_df = pd.concat([time_df, df1, df2])

output_df.to_csv('FE_estimate_by_frame.csv')
drop_df.to_csv('Frames_dropped_FE_by_frame.csv')
time_df.to_csv('FE_estimate_by_frame_over_time.csv')
