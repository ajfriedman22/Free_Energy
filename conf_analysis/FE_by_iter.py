import numpy as np
from alchemlyb.parsing.gmx import extract_u_nk
import pandas as pd
from alchemlyb.estimators import MBAR
from alchemlyb import concat
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_convergence
#from alchemlyb.preprocessing.subsampling import statistical_inefficiency
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

def get_MBAR(lambda_list, name, num_points):
    #Obtain u_nk reduced potentials and remove uncorrelated samples
    try:
        u_nk = concat([extract_u_nk(xvg, T=300) for xvg in lambda_list])
    except:
        return None, None, None, None

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

#Save imported arguments to variables
args = parser.parse_args()
lig_names = args.ln
sim_num = args.s
n_iter = args.i
file_path = args.f
temp = args.t

#Import pocket occupancy per iter
pocket_occupy = pd.read_csv('../Pockets/Pocket_occupancy_per_iter.csv', index_col=0)

# Perform analysis for all ligand names listed
output_df = pd.DataFrame()
time_df = pd.DataFrame()
for s in tqdm(range(len(lig_names))):
    #Determine which iterations are in which pocket
    pocketA, pocketB = [],[]
    track_iter_B = []
    per_list = pocket_occupy[pocket_occupy['Ligands']==lig_names[s]]['Percent Pocket B both'].values
    for i, per in enumerate(per_list):
        if i == 0:
            #Determine trajectory length
            input_file = open(f'{file_path}/sim_{s}/iteration_{i}/dhdl.xvg').readlines()
            final_time_step = 100
            traj_length = int(n_iter * final_time_step/1000)
            num_points = int(traj_length / 2)-1
        if per > 75:
            pocketB.append(f'{file_path}/sim_{s}/iteration_{i}/dhdl.xvg')
            track_iter_B.append(i)
        else:
            pocketA.append(f'{file_path}/sim_{s}/iteration_{i}/dhdl.xvg')
    np.savetxt(f'pockeB_{lig_names[s]}.txt', np.array(track_iter_B, dtype=float))

    pA_est, pA_err, pA_est_time, pA_err_time = get_MBAR(pocketA, f'{lig_names[s]}_pocketA', num_points)
    pB_est, pB_err, pB_est_time, pB_err_time = get_MBAR(pocketB, f'{lig_names[s]}_pocketB', num_points)
    df = pd.DataFrame({'Ligand': lig_names[s], 'Free Energy Estimate': [pA_est, pB_est], 'Free Energy Error': [pA_err, pB_err], 'Pocket': ['A', 'B']})
    output_df = pd.concat([output_df, df])

    df1 = pd.DataFrame({'Ligand': lig_names[s],  'Free Energy Estimate': pA_est_time, 'Free Energy Error': pA_err_time, 'Pocket': 'A', 'Time': np.arange(2, int(traj_length/(num_points+1))*(num_points+1), step=2)})
    df2 = pd.DataFrame({'Ligand': lig_names[s],  'Free Energy Estimate': pB_est_time, 'Free Energy Error': pB_err_time, 'Pocket': 'B', 'Time': np.arange(2, int(traj_length/(num_points+1))*(num_points+1), step=2)})
    time_df = pd.concat([time_df, df1, df2])

output_df.to_csv('FE_estimate.csv')
time_df.to_csv('FE_estimes_over_time.csv')
