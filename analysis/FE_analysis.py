import numpy as np
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
import pandas as pd
from alchemlyb.estimators import TI, MBAR, BAR
from alchemlyb import concat
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_ti_dhdl, plot_convergence
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.preprocessing.subsampling import statistical_inefficiency, equilibrium_detection
from alchemlyb.postprocessors.units import to_kcalmol
import argparse

def get_estimate(df):
    #Change labels to fit syntax for estimates dataframe
    l = np.linspace(0, 1, num=len(df.index))
    df.index = l
    df.columns = l
    est = np.round(df.loc[0, 1], decimals = 3)
    return est

def convergence(estimator):
    if estimator == 'TI':
        data_list = [statistical_inefficiency(extract_dHdl(xvg, T=temp), lower=eq_time, upper=run_time) for xvg in lambda_list]
        est = TI()
    else:
        data_list = [statistical_inefficiency(extract_u_nk(xvg, T=temp), lower=eq_time, upper=run_time) for xvg in lambda_list]
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
    ax.figure.savefig(f'Convergence_{estimator}.png')

#Import necessary arguments
parser = argparse.ArgumentParser(description = 'Free Energy Analysis using MBAR, BAR, and TI')
parser.add_argument('-f', required=False, type = str, default = 'dhdl.xvg', help='File path for xvg file ex: output/dhdl0.xvg and output/dhdl1.xvg--> output/dhdl')
parser.add_argument('-n', required=True, type = int, help='Number of Replicates')
parser.add_argument('-s', required=False, type = str, default = 'all', help='Which Estimators should be used? (TI or MBAR or BAR or all)')
parser.add_argument('-t', required=False, type = float, default = 300, help='Temperature Simulations Run (K)')
parser.add_argument('-r', required=True, type = float, help='Total Run Time(ps)')
parser.add_argument('-e', required=True, type = float, help='Time at Which equilibrium was reached (ps)')

#Save imported arguments to variables
args = parser.parse_args()
n_rep = args.n
run_time = args.r
eq_time = args.e
file_path = args.f
estimator = args.s
temp = args.t

#Prepare paths for dH/dl files
file_path = file_path.split('.')[0]
if n_rep > 1:
    lambda_list = []
    for i in range(n_rep):
        lambda_list.append(f'{file_path}{i}.xvg')
else:
    lambda_list = [f'{file_path}.xvg']

#Empty arrays for estimates
estimators, estimates, errors = [],[],[]
if estimator.lower() == 'ti' or estimator.lower() == 'all':
    #Load dHdl
    dHdl = concat([statistical_inefficiency(extract_dHdl(xvg, T=300), lower=eq_time, upper=run_time) for xvg in lambda_list])

    #TI Free Energy Estimate
    ti = TI().fit(dHdl)

    #Convert Units to Kcal/mol
    ti_kcal = to_kcalmol(ti.delta_f_, T=temp)
    dti_kcal = to_kcalmol(ti.d_delta_f_, T=temp)
    ti_all = pd.concat([ti_kcal, dti_kcal])
    ti_all.to_csv('TI.csv')

    #Get estimates from df
    TI_est = get_estimate(ti_kcal)
    TI_est_err = get_estimate(dti_kcal)
    
    estimators.append(TI)
    estimates.append(TI_est)
    errors.append(TI_est_err)

if estimator.lower() == 'mbar' or estimator.lower() == 'bar' or estimator.lower() == 'all':
   #Obtain u_nk reduced potentials and remove uncorrelated samples
    u_nk = concat([statistical_inefficiency(extract_u_nk(xvg, T=300), lower=eq_time, upper=run_time) for xvg in lambda_list])

    if estimator.lower() == 'mbar' or estimator.lower() == 'all':
        #MBAR FE Estimates
        mbar = MBAR(relative_tolerance=1e-04).fit(u_nk)

        #Convert Units to Kcal/mol
        mbar_kcal = to_kcalmol(mbar.delta_f_, T=temp)
        dmbar_kcal = to_kcalmol(mbar.d_delta_f_, T=temp)
        mbar_all = pd.concat([mbar_kcal, dmbar_kcal])
        mbar_all.to_csv('MBAR.csv')

        #Get estimates from df
        MBAR_est = get_estimate(mbar_kcal)
        MBAR_est_err = get_estimate(dmbar_kcal)

        #Add to array
        estimators.append('MBAR')
        estimates.append(MBAR_est)
        errors.append(MBAR_est_err)

    if estimator.lower() == 'bar' or estimator.lower() == 'all':
        #BAR FE Estimates
        bar = BAR().fit(u_nk)

        #Convert Units to Kcal/mol
        bar_kcal = to_kcalmol(bar.delta_f_, T=temp)
        dbar_kcal = to_kcalmol(bar.d_delta_f_, T=temp)
        bar_all = pd.concat([bar_kcal, dbar_kcal])
        bar_all.to_csv('BAR.csv')

        #Convert Units to Kcal/mol
        bar_kcal = to_kcalmol(bar.delta_f_, T=temp)
        dbar_kcal = to_kcalmol(bar.d_delta_f_, T=temp)
        bar_all = pd.concat([bar_kcal, dbar_kcal])
        bar_all.to_csv('BAR.csv')

        #Get estimates from df
        BAR_est = get_estimate(bar_kcal)
        BAR_est_err = get_estimate(dbar_kcal)

        #Add to array
        estimators.append('BAR')
        estimates.append(BAR_est)
        errors.append(BAR_est_err)

#Compare different Estimators
if estimator.lower() == 'all':
    ests = [ti, mbar, bar,]
    fig = plot_dF_state(ests, orientation='portrait')
    fig.savefig('dF_state.png', bbox_inches='tight')

#Error and Analysis
if estimator.lower() == 'ti' or estimator.lower() == 'all':
    #Plot dhdl of the TI
    ax = plot_ti_dhdl([ti])
    ax.figure.savefig('dhdl_TI.png')
    #Forward and Backward Convergence for TI
    convergence('TI')

if estimator.lower() == 'mbar' or estimator.lower() == 'all':
    #Plot MBAR Overlap Matrix
    ax = plot_mbar_overlap_matrix(mbar.overlap_matrix)
    ax.figure.savefig('Overlap_MBAR.png', bbox_inches='tight', pad_inches=0.0)

    #Forward and Backward Convergence for Coulomb MBAR
    convergence('MBAR')

print(errors)
df_est = pd.DataFrame({'Estimator': estimators, 'FE Estimate (kcal/mol)': estimates, 'Error (kcal/mol)': errors})
df_est.to_csv('FE_estimates.csv')
