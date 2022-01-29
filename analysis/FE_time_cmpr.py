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
print(lambda_list)

#Time List
time_value = np.linspace(eq_time*2, run_time, num = 25)

#Declare array for TI and MBAR 
TI_est = np.zeros(25)
MBAR_est = np.zeros(25)

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
    
    #Save FE Difference
    TI_est[i] = est

    #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([statistical_inefficiency(extract_u_nk(xvg, T=300), lower=eq_time, upper=t, step=2000) for xvg in lambda_list])

    #MBAR FE Estimates
    mbar = AutoMBAR(relative_tolerance=1e-04).fit(u_nk)

    #Change labels to fit syntax for dataframe
    j = n_state - 1
    l = np.linspace(0, 1, num=n_state)
    df = mbar.delta_f_
    df.index = l
    df.columns = l
    est = df.loc[0, 1]
    
    #Save FE Difference
    MBAR_est[i] = est

#Plot comparison
fig = plt.figure()
plt.plot(time_value, TI_est, color = 'red')
plt.scatter(time_value, TI_est, color = 'red', Label = 'TI')
plt.plot(time_value, TI_est, color = 'blue')
plt.scatter(time_value, TI_est, color = 'blue', Label = 'MBAR')
plt.legend(loc='best')
plt.xlabel('Trajectory Run Time (ps)')
plt.ylabel('Free Energy Estimate (kJ/mol)')
fig.savefig('Time_comparison.png')

