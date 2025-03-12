import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Compute the Relative Occupancy of Both Pockets')
parser.add_argument('-ln', required=True, nargs='+', type = str, help='Names of Ligand for FE output')
parser.add_argument('-s', required=True, nargs='+', type = str, help='Simulation numbers for the ligands')
parser.add_argument('-g', required=True, nargs='+', type = str, help='Names of Gro Files')
parser.add_argument('-i', required=False, type = int, default = 1, help='Total Number of Iterations')
parser.add_argument('-f', required=False, default='./', help='base file path storing simulations')

#Save imported arguments to variables
args = parser.parse_args()
lig_names = args.ln
gro_list = args.g
sim_num = args.s
num_iter = args.i
dir_path = args.f

if not os.path.exists('Pocket_occupancy_per_iter.csv'):
    per = []
    iter_df = pd.DataFrame()
    for g in tqdm(range(len(gro_list))):
        per_iter = []
        tot_pocketB = 0
        both_count = 0 
        for i in range(num_iter):
            traj = md.load(f'{dir_path}/sim_{sim_num[g]}/iteration_{i}/traj.trr', top=gro_list[g])
            traj.remove_solvent()

            #Select atom pairs
            c4 = traj.topology.select('name C4')
            if len(c4) == 0:
                c4 = traj.topology.select('name DC4')
            check1 = traj.topology.select('resid 271 and name CD')
            check2 = traj.topology.select('resid 299 and name CB')

            dist = md.compute_distances(traj, [[c4[0], check1[0]], [c4[0], check2[0]]])

            pocketB = 0
            for t in range(traj.n_frames):
                if dist[t,0] < 1.0 and dist[t,1] < 0.85:
                    pocketB += 1
            tot_pocketB += pocketB

            per_iter.append(100*pocketB/traj.n_frames)
        per.append(100*tot_pocketB/(traj.n_frames*num_iter))
        df = pd.DataFrame({'Ligands': lig_names[g], 'Percent Pocket B both': per_iter})
        iter_df = pd.concat([iter_df, df])

    iter_df.to_csv('Pocket_occupancy_per_iter.csv')
else:
    iter_df = pd.read_csv('Pocket_occupancy_per_iter.csv', index_col=0)

df_time = pd.DataFrame()
sample_both_pockets, all_per = np.zeros(len(lig_names)), np.zeros(len(lig_names))
for l, lig in enumerate(lig_names):
    per_array = iter_df[iter_df['Ligands'] == lig]['Percent Pocket B both'].to_numpy()
    # Get average every 2 ns
    traj = md.load(f'{dir_path}/sim_{sim_num[l]}/iteration_0/traj.trr', top=gro_list[l])
    traj_length = int(num_iter*traj.n_frames*.002)
    num_points = int(traj_length/2)-1
    time_avg = np.zeros(num_points)
    for i in range(1, num_points):
        t = int(i * len(per_array)/(num_points+1))
#        x = int(t - len(per_array)/(num_points+1))
#        y = int(t + len(per_array)/(num_points+1))
#        time_avg[i] = np.sum(per_array[x:y])/len(per_array[x:y])
        time_avg[i] = np.sum(per_array[:t])/len(per_array[:t])

    # Count which iterations have frames in both pockets
    both_count = 0
    for per in per_array:
        if per < 90 and per > 10:
            both_count += 1
    sample_both_pockets[l] = both_count
    all_per[l] = np.sum(per_array)/len(per_array)
    df = pd.DataFrame({'Ligand': lig_names[l], 'Time': np.arange(2, int(traj_length/(num_points+1))*(num_points+1), step=2),'Percent Occupancy Pocket B': time_avg})
    df_time = pd.concat([df_time, df])

df = pd.DataFrame({'Ligands': lig_names, '# of iterations which sample both pockets': sample_both_pockets})
df.to_csv('sample_both_pockes.csv')

df = pd.DataFrame({'Ligands': lig_names, 'Percent Pocket B': all_per})
df.to_csv('Pocket_occupancy.csv')

df_time.to_csv('Pocket_occupancy_over_time.csv')
