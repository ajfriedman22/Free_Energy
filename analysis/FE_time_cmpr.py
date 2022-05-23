from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
import pandas as pd
from alchemlyb.estimators import TI, MBAR, BAR, AutoMBAR
from alchemlyb import concat
from alchemlyb.preprocessing.subsampling import statistical_inefficiency, slicing
import numpy as np
import matplotlib.pyplot as plt
import argparse

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR, BAR, and TI with all dHdl files name dhdl#.xvg with # = lambda state')
parser.add_argument('-n', required=True, type = int, help='Number of Lambda States')
parser.add_argument('-r', required=True, type = int, help='Total Run Time(ps)')
parser.add_argument('-e', required=True, type = int, help='Time at Which equilibrium was reached (ps)')

#Save imported arguments to variables
args = parser.parse_args()
n_state = args.n
run_time = args.r
eq_time = args.e


#load dH/dl files
lambda_list = []
for i in range(n_state):
    lambda_list.append('dhdl' + str(i) + '.xvg')

#Time List
time_value = np.linspace(eq_time*2, run_time, num = 25)

#Declare array for TI and MBAR 
TI_est = np.zeros(25)
TI_est_err = np.zeros(25)
MBAR_est = np.zeros(25)
MBAR_est_err = np.zeros(25)
BAR_est = np.zeros(25)
BAR_est_err = np.zeros(25)


#Loop over time values
for i in range(len(time_value)):
    t = time_value[i]
    #Load Data and Remove uncorrelated Samples
    dHdl = concat([statistical_inefficiency(extract_dHdl(xvg, T=300), lower=eq_time, upper=t, step=2000) for xvg in lambda_list])

    #TI Free Energy Estimate
    ti = TI().fit(dHdl)

    #Change labels to fit syntax for dataframe
    j = n_state - 1
    l = np.linspace(0, 1, num=n_state)
    df = ti.delta_f_
    df.index = l
    df.columns = l
    est = df.loc[0, 1]
    
    #Change labels for error estimates
    df_err = ti.d_delta_f_
    df_err.index = l
    df_err.columns = l
    est_err = df_err.loc[0, 1]

    #Save FE Difference
    TI_est[i] = est
    TI_est_err[i] = est_err

    #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([statistical_inefficiency(extract_u_nk(xvg, T=300), lower=eq_time, upper=t, step=2000) for xvg in lambda_list])

    #MBAR FE Estimates
    mbar = AutoMBAR(relative_tolerance=1e-04).fit(u_nk)

    #Change labels to fit syntax for dataframe
    df = mbar.delta_f_
    df.index = l
    df.columns = l
    est = df.loc[0, 1]
    
    #Change labels for error estimates
    df_err = mbar.d_delta_f_
    df_err.index = l
    df_err.columns = l
    est_err = df_err.loc[0, 1]

    #Save FE Difference
    MBAR_est[i] = est
    MBAR_est_err[i] = est_err
    
    #BAR FE Estimates
    bar = BAR().fit(u_nk)
    
    #Change labels to fit syntax for dataframe
    df = bar.delta_f_
    df.index = l
    df.columns = l
    est = df.loc[0, 1]
    
    #Change labels for error estimates
    df_err = bar.d_delta_f_
    df_err.index = l
    df_err.columns = l
    est_err = df_err.loc[0, 1]

    #Save FE Difference
    BAR_est[i] = est
    BAR_est_err[i] = est_err

#Convert time to ns for plot
time_ns = np.zeros(len(time_value))
for i in range(len(time_value)):
    time_ns[i] = time_value[i]/1000

#Plot comparison
fig = plt.figure()
plt.plot(time_ns, TI_est, color = 'red')
plt.fill_between(time_ns, TI_est - TI_est_err, TI_est + TI_est_err, color = 'red', alpha = 0.2)
plt.scatter(time_ns, TI_est, color = 'red', Label = 'TI')
plt.plot(time_ns, MBAR_est, color = 'blue')
plt.fill_between(time_ns, MBAR_est - MBAR_est_err, MBAR_est + MBAR_est_err, color = 'blue', alpha = 0.2)
plt.scatter(time_ns, MBAR_est, color = 'blue', Label = 'MBAR')
plt.plot(time_ns, BAR_est, color = 'gray')
plt.fill_between(time_ns, BAR_est - BAR_est_err, BAR_est + BAR_est_err, color = 'gray', alpha = 0.2)
plt.scatter(time_ns, BAR_est, color = 'gray', Label = 'BAR')
plt.legend(loc='best')
plt.xlabel('Trajectory Run Time (ns)')
plt.ylabel('Free Energy Estimate (kJ/mol)')
fig.savefig('Time_comparison.png')

