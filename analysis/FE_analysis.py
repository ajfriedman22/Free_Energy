from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
import pandas as pd
from alchemlyb.estimators import TI, MBAR, BAR, AutoMBAR
from alchemlyb import concat
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_ti_dhdl, plot_convergence
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.preprocessing.subsampling import statistical_inefficiency, slicing
import argparse

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR, BAR, and TI with all dHdl files name dhdl#.xvg with # = lambda state')
parser.add_argument('-n', required=True, type = int, help='Number of Lambda States')
parser.add_argument('-t', required=False, type = bool, default = False, help='Should TI estimate be completed?')
parser.add_argument('-m', required=False, type = bool, default = False, help='Should MBAR estimate be completed?')
parser.add_argument('-r', required=True, type = int, help='Total Run Time(ps)')
parser.add_argument('-e', required=True, type = int, help='Time at Which equilibrium was reached (ps)')

#Save imported arguments to variables
args = parser.parse_args()
n_state = args.n
TI_check = args.t
MBAR_check = args.m
run_time = args.r
eq_time = args.e


#load dH/dl files
lambda_list = []
for i in range(n_state):
    lambda_list.append('dhdl' + str(i) + '.xvg')
print(lambda_list)

#Set Output File
output = open('FE_analysis.txt', 'w')
output.write('---------------------------------------------\n')

if TI_check == True:
    #Load Data and Remove uncorrelated Samples
    dHdl = concat([statistical_inefficiency(extract_dHdl(xvg, T=300), lower=eq_time, upper=run_time, step=2000) for xvg in lambda_list])

    #TI Free Energy Estimate
    ti = TI().fit(dHdl)

    #Free energy difference for each lambda window
    output.write('TI:\n')
    output.write(str(ti.delta_f_) + '\n')
    
if MBAR_check == True:
   #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([statistical_inefficiency(extract_u_nk(xvg, T=300), lower=eq_time, upper=run_time, step=2000) for xvg in lambda_list])

    #MBAR FE Estimates
    mbar = AutoMBAR(relative_tolerance=1e-04).fit(u_nk)

    #Free energy difference for each lambda window
    output.write('MBAR:\n')
    output.write(str(mbar.delta_f_) + '\n')

#Error on delta f
output.write('---------------------------------------------\n')
if TI_check == True:
    output.write('TI Free Energy Error\n')
    output.write(str(ti.d_delta_f_) + '\n')
if MBAR_check == True:
    output.write('MBAR Free Energy Error\n')
    output.write(str(mbar.d_delta_f_) + '\n')

if MBAR_check == True:
    #Plot MBAR Overlap Matrix
    ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
    ax.figure.savefig('Overlap_MBAR.png', bbox_inches='tight', pad_inches=0.0)

if TI_check == True:
    #Plot dhdl of the TI
    ax = plot_ti_dhdl([ti])
    ax.figure.savefig('dhdl_TI.png')

if MBAR_check == True:
    #Forward and Backward Convergence for Coulomb MBAR
    data_list = [extract_u_nk(xvg, T=300) for xvg in lambda_list]
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = 10
    for i in range(1, num_points+1):
        # Do the forward
        slice = int(len(data_list[0])/num_points*i)
        u_nk = concat([data[:slice] for data in data_list])
        estimate = AutoMBAR().fit(u_nk)
        forward.append(estimate.delta_f_.iloc[0,-1])
        forward_error.append(estimate.d_delta_f_.iloc[0,-1])
        # Do the backward
        u_nk = concat([data[-slice:] for data in data_list])
        estimate = AutoMBAR().fit(u_nk)
        backward.append(estimate.delta_f_.iloc[0,-1])
        backward_error.append(estimate.d_delta_f_.iloc[0,-1])
    
    ax = plot_convergence(forward, forward_error, backward, backward_error)
    ax.figure.savefig('Convergence_MBAR.png')

if TI_check == True:
    #Forward and Backward Convergence for Coulomb TI
    data_list = [extract_dHdl(xvg, T=300) for xvg in lambda_list]
    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = 10
    for i in range(1, num_points+1):
        # Do the forward
        slice = int(len(data_list[0])/num_points*i)
        dHdl = concat([data[:slice] for data in data_list])
        estimate = TI().fit(dHdl)
        forward.append(estimate.delta_f_.iloc[0,-1])
        forward_error.append(estimate.d_delta_f_.iloc[0,-1])
        # Do the backward
        dHdl = concat([data[-slice:] for data in data_list])
        estimate = TI().fit(dHdl)
        backward.append(estimate.delta_f_.iloc[0,-1])
        backward_error.append(estimate.d_delta_f_.iloc[0,-1])
    
    ax = plot_convergence(forward, forward_error, backward, backward_error)
    ax.figure.savefig('Convergence_TI.png')
