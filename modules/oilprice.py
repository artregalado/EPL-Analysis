# This document creates the Oil Price Uncertainty Modlule.
# Contains the following class definitions
# A class to calibrate an Ornstein–Uhlenbeck process  for oil price based on Smith 2010 procedure. 
# See https://www.academia.edu/29626021/On_the_Simulation_and_Estimation_of_the_Mean-Reverting_Ornstein-Uhlenbeck_Process_Especially_as_Applied_to_Commodities_Markets_and_Modelling 
# A class that takes the calibrated parameters and simulates oil price paths. 


# The simulation procedure of oil prices follows a variation of Salam's 2022 paper (specifically, the appendix formula of the mean reversion process). And also by adjusting the code based on my previous matlab.

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime

def ou_calibration(price_series, deltat):
    
    """
    This function calibrates an Ornstein - Uhlenbeck process based on oil price data.
    
    Parameters
    ----------
    price_series : numpy.array
        Oil price series in levels.
    deltat : int
        Time differential between periods of the data price series. For example, for annual prices the deltat is 1.
          
    Returns:
        reversion_speed: Speed of mean reversion of price
        equilibrium_price: estimated long-term price of the series
        volatility: estimated volatility if the price
        resutls: results from fitting the model
    """
    
    y = price_series[1:]
    X = price_series[0:-1]
    X = sm.add_constant(X)
    
    results = sm.OLS(y, X).fit()
    
    betacero = results.params[0]
    betaone = results.params[1]
    resid = results.resid
    
    reversion_speed = -np.log(betaone)/deltat
    equilibrium_price = betacero/(1-betaone) 
    volatility = np.std(resid) * np.sqrt( 2*reversion_speed/(1-betaone**2))
    
    return reversion_speed, equilibrium_price, volatility, results

def simulate_mean_reversion_price_paths(number_periods,
                                        number_simulations,
                                        deltat, reversion_speed,
                                        equilibrium_price,
                                        volatility,
                                        initial_price,
                                        minimum_price):

    
    """
    This function simulates mean reversion price paths based on calinrated Ornstein - Uhlenbeck process. 
    
    Parameters
    ----------     
    reversion_speed: Speed of mean reversion of price
    equilibrium_price: estimated long-term price of the series
    volatility: estimated volatility if the price
    initial_price: initial assumed oil price for the simulations
    minimum_price: indicate a minimum price for simulations.
          
    Returns:
    ----------
    numpy array of simulated mean reversion price paths with shape (number_simulations, number_periods)
    """
    
    # Set reversion speed manually to value in literature
    # Becasue the procedure has a bias due to OLS, refer to Salam's paper
    reversion_speed = .5
    periods_simulated = math.floor(number_periods/deltat)
    exp_minus_speed_deltat = np.exp(-reversion_speed*deltat)

    # Simulate the random error terms. For the mean reversion process is a Weiner Proccess 
    # Be sure to handle the case where reversion speed is zero, meaning no mean reversion
    # Weiner Process
    if reversion_speed == 0: # No mean reversion case
        np.random.seed(42)
        error = np.random.normal(size = [number_simulations, periods_simulated])
        dWt = np.sqrt(deltat) * error
    else:
        np.random.seed(42)
        error = np.random.normal(size = [number_simulations, periods_simulated])
        dWt = np.sqrt((1-np.exp(-2*reversion_speed*deltat)) / (2*reversion_speed)) * error


    # Container for results and simulation process
    simulated_price_path = np.full([number_simulations, number_periods],
                                fill_value=initial_price)


    for i in np.arange(0, number_simulations):
        for t in np.arange(1, periods_simulated):
            S_t = simulated_price_path[i][t-1]*exp_minus_speed_deltat + equilibrium_price*(1-exp_minus_speed_deltat) + volatility*dWt[i][t]
            simulated_price_path[i][t] = S_t
            
    simulated_price_path[simulated_price_path < minimum_price] = minimum_price
            
    return simulated_price_path


def calibrate_and_simulate_mean_reversion_price_paths(price_series, deltat,
                                                      number_periods, number_simulations,
                                                      initial_price, minimum_price):
    
    """Function wrapper for ou_calibration and simulate_mean_reversion_price_paths functions

    Returns:
        numpy ndarray: simulated mean reversion price paths
        results: results of O-U calibration
        """
    
    [reversion_speed, equilibrium_price, volatility, ols_results] = ou_calibration(price_series, deltat)
    
    calibration_results = {"reversion_speed": reversion_speed,
                           "equilibrium_price": equilibrium_price,
                           "volatility":volatility}
    
    simulated_price_paths = simulate_mean_reversion_price_paths(number_periods,
                                                                number_simulations,
                                                                deltat, reversion_speed,
                                                                equilibrium_price,
                                                                volatility,
                                                                initial_price,
                                                                minimum_price)
    
    
    # Print results and procedure so the user can track the stages and get the results
    
    print(f"Date of analysis: {datetime.now()}")
    print("Beginning oil pricecalibrationa and simulation procedure for mean reversion process.")
    print("Calibrating mean mean reversion process...")
    print(f"Generating {simulated_price_paths.shape[0]} mean reversion price paths over {simulated_price_paths.shape[1]} years...")
    print("-"*80)
    print(ols_results.summary())
    print("")
    print("-"*80)
    print("Calibration results:")
    print(f"Equilibrium Price: {np.round(calibration_results['equilibrium_price'],3)}")
    print(f"Volatility: {np.round(calibration_results['volatility'],3)}")  
    print(f"Reversion Speed: {np.round(calibration_results['reversion_speed'],3)}")    
    
    
    return simulated_price_paths, ols_results, calibration_results


# # Example of the procedure with sample prices
# # From Salam's paper

# prices = pd.read_csv('sample_price_data.txt',
#                      header=None, usecols=[0]).values

# [simulated_price_paths, ols_results, calibration_results] = calibrate_and_simulate_mean_reversion_price_paths(
#     price_series=prices,
#     deltat=1,
#     number_periods=30,
#     number_simulations=10000,
#     initial_price=60,
#     minimum_price=20)
# %%
