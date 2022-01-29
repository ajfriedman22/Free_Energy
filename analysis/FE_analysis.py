import numpy as np
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
import pandas as pd
from alchemlyb.estimators import TI, MBAR, BAR, AutoMBAR
from alchemlyb import concat
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_ti_dhdl, plot_convergence
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.preprocessing.subsampling import statistical_inefficiency, equilibrium_detection
import argparse

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR, BAR, and TI with all dHdl files name dhdl#.xvg with # = lambda state')
parser.add_argument('-n', required=True, type = int, help='Number of Lambda States')
parser.add_argument('-t', required=False, type = bool, default = True, help='Should TI estimate be completed?')
parser.add_argument('-m', required=False, type = bool, default = True, help='Should MBAR estimate be completed?')
parser.add_argument('-a', required=False, type = bool, default = True, help='Should TI, MBAR, and BAR estimates be completed?')
parser.add_argument('-r', required=True, type = int, help='Total Run Time(ps)')
parser.add_argument('-e', required=True, type = int, help='Time at Which equilibrium was reached (ps)')

#Save imported arguments to variables
args = parser.parse_args()
n_state = args.n
TI_check = args.t
MBAR_check = args.m
ALL_check = args.a
run_time = args.r
eq_time = args.e


#load dH/dl files
lambda_list = []
for i in range(n_state):
    lambda_list.append('dhdl' + str(i) + '.xvg')
print(lambda_list)

#Set Output File
output = open('FE_analysis.txt', 'w')
output_all = open('FE_analysis_all.txt', 'w')

if TI_check == True or ALL_check == True:
    #Load Data and Remove uncorrelated Samples
    dHdl = concat([statistical_inefficiency(extract_dHdl(xvg, T=300), lower=eq_time, upper=run_time, step=2000) for xvg in lambda_list])

    #TI Free Energy Estimate
    ti = TI().fit(dHdl)

    #Free energy difference for each lambda window
    output_all.write('TI:\n')
    output_all.write(str(ti.delta_f_) + '\n')
    
    #Change labels to fit syntax for estimates dataframe
    j = n_state - 1
    l = np.linspace(0, 1, num=n_state)
    df_ti = ti.delta_f_
    df_ti.index = l
    df_ti.columns = l
    TI_est = np.round(df_ti.loc[0, 1], decimals = 3)
    
    #Change labels for error estimates
    df_ti_err = ti.d_delta_f_
    df_ti_err.index = l
    df_ti_err.columns = l
    TI_est_err = np.round(df_ti_err.loc[0, 1], decimals = 3)

    #Output Free Energy Differnece
    output.write('TI Estimate: ' + str(TI_est) + ' +/- ' + str(TI_est_err) + '\n')

if MBAR_check == True or ALL_check == True:
   #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([statistical_inefficiency(extract_u_nk(xvg, T=300), lower=eq_time, upper=run_time, step=2000) for xvg in lambda_list])

    #MBAR FE Estimates
    mbar = AutoMBAR(relative_tolerance=1e-04).fit(u_nk)
    bar = BAR().fit(u_nk)

    #Free energy difference for each lambda window
    output_all.write('MBAR:\n')
    output_all.write(str(mbar.delta_f_) + '\n')

    #Change labels to fit syntax for estimates dataframe
    j = n_state - 1
    l = np.linspace(0, 1, num=n_state)
    df_mbar = mbar.delta_f_
    df_mbar.index = l
    df_mbar.columns = l
    MBAR_est = np.round(df_mbar.loc[0, 1], decimals = 3)
    
    #Change labels for error estimates
    df_mbar_err = mbar.d_delta_f_
    df_mbar_err.index = l
    df_mbar_err.columns = l
    MBAR_est_err = np.round(df_mbar_err.loc[0, 1], decimals = 3)

    #Output Free Energy Differnece
    output.write('MBAR Estimate: ' + str(MBAR_est) + ' +/- ' + str(MBAR_est_err) + '\n')

#Compare different Estimators
if ALL_check == True:
    estimators = [ti, mbar, bar,]
    fig = plot_dF_state(estimators, orientation='portrait')
    fig.savefig('dF_state.png', bbox_inches='tight')

#Error and Analysis
output.write('---------------------------------------------\n')
if TI_check == True or ALL_check == True:
    #Output TI Error for all lambdas
    output_all.write('TI Free Energy Error\n')
    output_all.write(str(ti.d_delta_f_) + '\n')
    #Plot dhdl of the TI
    ax = plot_ti_dhdl([ti])
    ax.figure.savefig('dhdl_TI.png')
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

if MBAR_check == True or ALL_check == True:
    #Output MBAR error for all lambdas
    output_all.write('MBAR Free Energy Error\n')
    output_all.write(str(mbar.d_delta_f_) + '\n')
    #Plot MBAR Overlap Matrix
    ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
    ax.figure.savefig('Overlap_MBAR.png', bbox_inches='tight', pad_inches=0.0)

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


