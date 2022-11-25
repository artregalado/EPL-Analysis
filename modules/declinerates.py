# This document creates the exponential decline moduel for decline rate uncertainty analysis
# Simulates production paths for all designed oil fields for the PhD Research.
# %% Header and imports
# This script retrieves production data from sanctioned and probable fields
# in the OGA Stewardship Database (ACREEF). It also estimates exponential decline rates
# for the fields and derives an optimal distribution to be used in Monte Carlo modelling.

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.optimize
import scipy.stats as st
from distfit import distfit



# %% Required aux functions
def get_exponential_decline_rates(production_profiles):
    def exponential_decay(x, q0, beta, c):
        # Parameters
        # x = time component, used x to simplify naming with scipy
        # q0 = initial rate of production at decline
        # beta = decline rate parameter
        # c = constant

        return q0 * np.exp(-beta * x) + c

    # Container and estimation
    decline_rates = []
    parameters = []

    for field in production_profiles.keys():
        profile = np.array(production_profiles[field])

        # Data to fit
        # Get only from max point and then the decline
        id_max = np.argmax(profile)
        y = profile[id_max:]
        x = np.arange(len(y))

        # Initial Guess
        p0 = [np.max(y),  # Max production, begin of decline
              0.2,  # Decline rate
              0]  # Constant

        # popt and pcov (see scipy documentation) is used by convention, the first are
        # parameters, the second covariance

        popt, pcov = scipy.optimize.curve_fit(exponential_decay, x, y, p0,
                                              bounds=(0, np.inf),
                                              maxfev=3000)

        # Store beta parameters, second parameter of popt
        decline_rates.append(popt[1])
        parameters.append(popt)

        # Plot if needed
        # plt.plot(x, y, 'b-', label='data')
        # plt.plot(x, exponential_decay(x, *popt), 'r-')
        # plt.show()

    decline_rates = np.array(decline_rates)
    parameters = np.array(parameters)
    # Clean workspace and keep only the production profiles and decline rates

    del field, id_max, p0, pcov, popt, profile, x, y

    return decline_rates, parameters


# %% Hyperbolic decline rate estimations

def get_hyperbolic_decline_rates(production_profiles):
    def hyperbolic_decay(x, q0, beta, gamma):
        # Parameters
        # x = time component, used x to simplify naming with scipy
        # q0 = initial rate of production at decline
        # beta = decline rate parameter
        # gamma = production rate fraction at which decline occurs
        # c = constant
        return q0 / ((1 + gamma * beta * x) ** (1 / gamma))

    # Container and estimation
    decline_rates = []
    parameters = []

    for field in production_profiles.keys():
        profile = np.array(production_profiles[field])

        # Data to fit
        # Get only from max point and then the decline
        id_max = np.argmax(profile)
        y = profile[id_max:]
        x = np.arange(len(y))

        # Initial Guess for hyperbolic

        p0 = [np.max(y),  # Max production, begin of decline
              0.2,  # Decline rate
              0.5,  # Gamma
              ]

        # popt and pcov (see scipy documentation) is used by convention, the first are
        # parameters, the second covariance

        popt, pcov = scipy.optimize.curve_fit(hyperbolic_decay, x, y, p0,
                                              bounds=(0, np.inf),
                                              maxfev=3000)

        # Store beta parameters, second parameter of popt
        decline_rates.append(popt[1])
        parameters.append(popt)

        # Plot if needed
        # plt.plot(x, y, 'b-', label='data')
        # plt.plot(x, hyperbolic_decay(x, *popt), 'r-')
        # plt.show()

    decline_rates = np.array(decline_rates)
    parameters = np.array(parameters)
    del field, id_max, p0, pcov, popt, profile, x, y

    return decline_rates, parameters


# %% Logistic declines

def get_logistic_decline_rates(production_profiles):
    # Define function and calibrate from fields
    def logistic_growth(x, a, b, K):
        return K / (1 + a * np.exp(-b * x))

    # Container and estimation
    decline_rates = []
    parameters = []

    for field in production_profiles.keys():
        # Get profile and transform to total cumulative
        profile = production_profiles[field]
        profile = profile * 365  # Change from bbl/d to barrels per year
        profile = profile.cumsum()

        bounds = (0, [np.inf, 6, np.inf])

        x = np.arange(len(profile))
        y = profile.values

        # Curve fit and get parameters
        popt, popv = scipy.optimize.curve_fit(logistic_growth, x, profile,
                                              bounds=bounds, maxfev=1000000,
                                              method='trf')

        # Store parameters, beta is second parameter
        decline_rates.append(popt[1])
        parameters.append(popt)

        # Plot if visual inspection required
        # plt.plot(x, y, 'b-', label='data')
        # plt.plot(x, logistic_growth(x, *popt), 'r-')
        # plt.show()

    decline_rates = np.array(decline_rates)
    parameters = np.array(parameters)

    del field, bounds, popv, popt, profile, x, y
    return decline_rates, parameters


# %% Initial production estimates for Monte Carlo Simulations

def calculate_initial_production(upper_bound, lower_bound,
                                 K, a, logistic_parameters,
                                 simulate_logistic_profile,
                                 num_years):
    # Filter by size of K
    mask = np.logical_and(np.greater(logistic_parameters[:, 2], lower_bound),
                          np.less(logistic_parameters[:, 2], upper_bound))

    filtered_log_params = logistic_parameters[mask, :]
    beta = np.mean(filtered_log_params[:, 1])

    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K,
                                                                 num_years)

    # Transform to barrels per day.
    x_logistic = annualized_rate / 365
    plt.plot(x_logistic)
    plt.show()

    # Get initial production that will then be used to simulate the exponential and
    # logistic profiles. I decided to go this route as setting the arbitrary initial values
    # could not be as good as actually using the logistic growth as starting point
    idx = np.argmax(x_logistic)
    initial_production = x_logistic[0:idx + 1]

    print('Total field production in MMbbl')
    print(np.sum(x_logistic * 365) / 1000000)

    # Output is in barrels per day, and return the full profile in annualized
    return initial_production, x_logistic


# %% Read and clean stewardship data
print('Reading and cleaning OGA Stewardship database...')

master = pd.read_excel('data/copy_oga_stewardship_2021_confidential.xlsx',
                       usecols='A:AH', skiprows=1, engine='openpyxl')
# Notes about the data set
# Columns P-S refer to CAPEX df.columns[15:19] (last index not inclusive)
# Columns T-Y refer to OPEX df.columns[19:25] (last index not inclusive)
# Columns Z-AD refer to DECOMMX df.columns[25:30] (last index not inclusive)

# Essential cleaning

# Variable names
master.columns = [column.strip() for column in master.columns]
master.columns = [column.replace('\n', ' ') for column in master.columns]
master.columns = [column.replace(' ', '_') for column in master.columns]
master.columns = [column.lower() for column in master.columns]

# %% Cleaning and filtering
print('Filtering desired fields prior to decline rate estimation...')

# Filters and desired columns
df = master.copy()
mask = df['category_(ii)'].isin(['SANCTIONED'])
df = df[mask]

df = df[['year', 'field_name', 'crude_oil_sales_(bbl/d)']]
df.columns = ['year', 'field', 'bbl/d']

# Drop NA and zeros, that is the way to clean and keep only fields which report
# oil production

df['bbl/d'] = df['bbl/d'].replace({0: np.nan})
df = df.dropna()

# Procedure to keep only fields with more than 5 years of production. This is to make sure
# I have enough data points to undertake decline rate analysis. Need to group by year and
# sum production because it is possible that there are different activities for the same
# field so they need to be accounted for.One example is "Schiehallion Wells and
# Schiehallion Base".

all_fields = df.field.unique()
print(f'Sanctioned oil fields in database {str(len(all_fields))}')

fields_to_exclude = []

for field in all_fields:
    subset = df.loc[df['field'] == field]
    producing_years = subset.groupby('year')['bbl/d'].sum()
    if len(producing_years) <= 5:
        fields_to_exclude.append(field)

mask = df['field'].isin(fields_to_exclude)

# Dataset to obtain production profiles
df = df[~mask]

# %% Make dictionary with production profiles

production_profiles = {}

for field in df.field.unique():
    subset = df.copy()
    subset = subset.loc[subset['field'] == field]
    subset['year'] = pd.to_datetime(subset['year'], format='%Y')
    subset = subset.sort_values('year', ascending=True)
    production = subset.groupby('year')['bbl/d'].sum()
    production_profiles[field] = production

# # Manual check for inconsistencies and remove
# for field, profile in production_profiles.items():
#     profile.plot()
#     plt.title(field)
#     plt.show()
#     time.sleep(2)

remove_visual = ['CYRUS', 'FOINAVEN', 'ORLANDO', 'COOK', 'PICT', 'LEVEN', 'NETHAN',
                 'STARLING', 'HAWKINS', 'BRAE-CENTRAL [PART OF BRAE]']

for field in remove_visual:
    if field in production_profiles:
        del production_profiles[field]

# Clean variables that are not used an keep only objects used
del field, fields_to_exclude, producing_years, mask
del production, remove_visual, subset, all_fields

print(
    f'Final sample of production profiles used for estimation {str(len(production_profiles))}')

# %% Estimate decline rate parameters from fields
print('Calibrating decline rates from UKCS fields...')

print('Exponetial model...')
exponential_decline_rates, exponential_parameters = get_exponential_decline_rates(
    production_profiles)

print('Hyperbolic model...')
hyperbolic_decline_rates, hyperbolic_parameters = get_hyperbolic_decline_rates(
    production_profiles)

print('Logistic model...')
logistic_decline_rates, logistic_parameters = get_logistic_decline_rates(
    production_profiles)

print('Calibration finished...')

# %% Create function to Simulate logistic profile and test

# Test by recovering mean of estimated parameters. I wanted to simulate, say,
# a 50 million barrel field I would only have to set K to be 50,000,000
a = np.mean(logistic_parameters[:, 0])
beta = np.mean(logistic_parameters[:, 1])
K = np.mean(logistic_parameters[:, 2])

num_years = 20

def simulate_logistic_profile(a, beta, K, num_years):
    # From Kemp and Kasim 2005 we can get the marginal annualized rate of production
    # following formula 2 in their paper.

    # We calculate D1t with the logistic growth formula (1) in paper
    D1t = K / (1 + a * np.exp(-beta * np.arange(num_years)))

    # Then apply formula (2) to get annualized rate of production decline
    # rate = (-1*(a/K)) - D1t*(K-D1t)

    # From Sympy differentiation
    # I differentiated (1) using Sympy because it appears that Kemp and Kasim
    # Formula differentiatied is wrong or is meant as something else
    x = np.arange(num_years)
    num = K*a*beta*np.exp(-beta*x)
    denom = np.power(a*np.exp(-beta*x) + 1, 2)
    rate = num/denom

    return D1t, rate

cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)

# Transform to barrels per day.
x_logistic = annualized_rate/365
plt.plot(x_logistic)
plt.show()


# Get initial production that will then be used to simulate the exponential and
# logistic profiles. I decided to go this route as setting the arbitrary initial values
# could not be as good as actually using the logistic growth as starting point
idx = np.argmax(x_logistic)
initial_production = x_logistic[0:idx+1]

# %% Create function to simulate exponential profile and test

num_years = 20
beta = np.mean(exponential_decline_rates)

def simulate_exponential_profile(initial_production, beta, num_years):
    # Initial production must be in barrels per day
    max_prod = np.max(initial_production)
    years_forecast = np.arange(num_years)
    simulation = max_prod * np.exp(-beta * years_forecast)
    simulated_profile = np.append(initial_production[0:], simulation[1:])

    # Simulation is in barrels per day

    return simulated_profile


x_exponential = simulate_exponential_profile(initial_production, beta, num_years)
plt.plot(x_exponential)
plt.show()

# %% Create function to Simulate hyperbolic profile and test
num_years = 20

# Test by recovering mean of estimated parameters
beta = np.mean(hyperbolic_parameters[:, 1])
gamma = np.mean(hyperbolic_parameters[:, 2])


def simulate_hyperbolic_profile(initial_production, beta, gamma, num_years):
    # Initial production must be in barrels per day
    simulation = np.max(initial_production) / (
            (1 + beta * gamma * np.arange(num_years)) ** (1 / gamma))
    simulated_profile = np.append(initial_production[0:], simulation[1:])

    # Simulation is in barrels per day

    return simulated_profile


x_hyperbolic = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
plt.plot(x_hyperbolic)
plt.show()

# %% Plot comparison of different simulations for validation.
plt.plot(x_exponential, label='exponential')
plt.plot(x_hyperbolic, label='hyperbolic')
plt.plot(x_logistic, label='logistic')
#plt.title('Example of simulated profiles at mean parameters by model')
plt.ylabel('barrels per day')
plt.xlabel('Years from beginning of production')
plt.legend()
plt.savefig("plots/simulated_profiles_at_mean_parameters.png")
plt.show()
# %% Clean workspace of unnecessary variables

del K, a, annualized_rate, beta, gamma, idx, cumulative_rate
del num_years, x_exponential, x_logistic, x_hyperbolic, initial_production

# %% Begin Monte Carlo Simulations
#############################
#############################
# This section begins the Monte Carlo simulations that will then be analysed in
# Matlab financial model.

# Set initial values for the different field sizes using the logistic procedure

##########################
# 10 MMbbl Initial values

# Inputs
upper_bound = 20000000 # Size in million barrels
lower_bound = 0
K = 18000000 # K is the parameter for field size
num_years = 20
a = 1

# We allow the a parameter to vary as that decides the amount of years to reach peak
# production, and it needs to change between field sizes for more realistic profiles.

initial_10m, full_profile_10 = calculate_initial_production(upper_bound, lower_bound,
                                                K, a, logistic_parameters,
                                                simulate_logistic_profile, num_years)

##########################
# 25 MMbbl Initial values
# Inputs
upper_bound = 35000000 # Size in million barrels, upper bound of field sizes in sample
lower_bound = 5000000
K = 35000000 # K is the parameter for field size
num_years = 20
a = 1.5

# We allow the a parameter to vary as that decides the amount of years to reach peak
# production, and it needs to change between field sizes for more realistic profiles.

initial_25m, full_profile_25 = calculate_initial_production(upper_bound, lower_bound,
                                                K, a, logistic_parameters,
                                                simulate_logistic_profile, num_years)

##########################
# 50 MMbbl Initial values
# Inputs
upper_bound = 65000000 # Size in million barrels, upper bound of field sizes in sample
lower_bound = 50000000
K = 75000000 # K is the parameter for field size
num_years = 20
a = 2

# We allow the a parameter to vary as that decides the amount of years to reach peak
# production, and it needs to change between field sizes for more realistic profiles.

initial_50m, full_profile_50 = calculate_initial_production(upper_bound, lower_bound,
                                                K, a, logistic_parameters,
                                                simulate_logistic_profile, num_years)


##########################
# 100 MMbbl Initial values
# Inputs
upper_bound = 200000000 # Size in million barrels, upper bound of field sizes in sample
lower_bound = 75000000
K = 138000000 # K is the parameter for field size
num_years = 20
a = 2.4

# We allow the a parameter to vary as that decides the amount of years to reach peak
# production, and it needs to change between field sizes for more realistic profiles.

initial_100m, full_profile_100 = calculate_initial_production(upper_bound, lower_bound,
                                                K, a, logistic_parameters,
                                                simulate_logistic_profile, num_years)

##########################
# 150 MMbbl Initial values
# Inputs
upper_bound = 1000000000 # Size in million barrels, upper bound of field sizes in sample
lower_bound = 100000000
K = 215000000 # K is the parameter for field size
num_years = 20
a = 2.4

# We allow the a parameter to vary as that decides the amount of years to reach peak
# production, and it needs to change between field sizes for more realistic profiles.

initial_150m, full_profile_150 = calculate_initial_production(upper_bound, lower_bound,
                                                K, a, logistic_parameters,
                                                simulate_logistic_profile, num_years)


# %% Exponential Model
print('Monte Carlo Simulations procedure to simulate exponential production profiles...')
print('Fitting probability distributions...')

# Fit distributions and find optimal one
dist = distfit()
dist.fit_transform(exponential_decline_rates)
# fig = dist.plot()
# f = fig[0]
# f.show()
# f.savefig('plots/distribution_fit_decline_rates.svg')

optimal_dist = dist.model['name']
print('Best fitting distribution for decline rates:', optimal_dist)

# Simulate exponential production profiles for the five oil fields
print('Simulating exponential production profiles...')

# Test with optimal distribution
# Find parameters of the distribution given the population decline rates
# Then generate random decline rates

a, loc, scale = st.gamma.fit(exponential_decline_rates)
sim_decline_rates = st.gamma.rvs(a, loc, scale,
                                 size=10000, random_state=64)

# Now simulate production profiles based on the random decline rates for each of the
# fields Production levels are taken from the matlab model assumptions files The
# original assumption is in barrels/y, just transform to bbl/d and then back

num_years = 20


# 10 MMbbl field
initial_production = list(initial_10m)
simulated_profiles_10mmbl = np.empty((0, 20))

for rate in sim_decline_rates:
    x = simulate_exponential_profile(initial_production, rate, num_years)
    x = x[0:21]
    x = np.reshape(x, (1, 20))
    simulated_profiles_10mmbl = np.append(simulated_profiles_10mmbl, x, axis=0)
    simulated_profiles_10mmbl = simulated_profiles_10mmbl

simulated_profiles_10mmbl = simulated_profiles_10mmbl * 365


# 25 MMbbl field
initial_production = list(initial_25m)
simulated_profiles_25mmbl = np.empty((0, 20))

for rate in sim_decline_rates:
    x = simulate_exponential_profile(initial_production, rate, num_years)
    x = x[0:20]
    x = np.reshape(x, (1, 20))
    simulated_profiles_25mmbl = np.append(simulated_profiles_25mmbl, x, axis=0)
    simulated_profiles_25mmbl = simulated_profiles_25mmbl

simulated_profiles_25mmbl = simulated_profiles_25mmbl * 365


# 50 MMbbl field
initial_production = list(initial_50m)
simulated_profiles_50mmbl = np.empty((0, 21))

for rate in sim_decline_rates:
    x = simulate_exponential_profile(initial_production, rate, num_years)
    x = x[0:21]
    x = np.reshape(x, (1, 21))
    simulated_profiles_50mmbl = np.append(simulated_profiles_50mmbl, x, axis=0)
    simulated_profiles_50mmbl = simulated_profiles_50mmbl

simulated_profiles_50mmbl = simulated_profiles_50mmbl * 365



# # 100 MMbbl field
initial_production = list(initial_100m)
simulated_profiles_100mmbl = np.empty((0, 21))

for rate in sim_decline_rates:
    x = simulate_exponential_profile(initial_production, rate, num_years)
    x = x[0:21]
    x = np.reshape(x, (1, 21))
    simulated_profiles_100mmbl = np.append(simulated_profiles_100mmbl, x, axis=0)
    simulated_profiles_100mmbl = simulated_profiles_100mmbl

simulated_profiles_100mmbl = simulated_profiles_100mmbl * 365



# 150 MMbbl field
initial_production = list(initial_150m)

simulated_profiles_150mmbl = np.empty((0, 21))

for rate in sim_decline_rates:
    x = simulate_exponential_profile(initial_production, rate, num_years)
    x = x[0:21]
    x = np.reshape(x, (1, 21))
    simulated_profiles_150mmbl = np.append(simulated_profiles_150mmbl, x, axis=0)
    simulated_profiles_150mmbl = simulated_profiles_150mmbl

simulated_profiles_150mmbl = simulated_profiles_150mmbl * 365

# Save to python objects
exponential_decline_profiles = {
    'simulated_exponential_rates': sim_decline_rates,
    'mmbbl_10': simulated_profiles_10mmbl,
    'mmbbl_25': simulated_profiles_25mmbl,
    'mmbbl_50': simulated_profiles_50mmbl,
    'mmbbl_100': simulated_profiles_100mmbl,
    'mmbbl_150': simulated_profiles_150mmbl
}


# Export to matlab
from scipy.io import savemat

decline_rates_simulations = {'sim_decline_rates': sim_decline_rates}
mmbl10 = {'mmbbl_10': simulated_profiles_10mmbl}
mmbl25 = {'mmbbl_25': simulated_profiles_25mmbl}
mmbl50 = {'mmbbl_50': simulated_profiles_50mmbl}
mmbl100 = {'mmbbl_100': simulated_profiles_100mmbl}
mmbl150 = {'mmbbl_150': simulated_profiles_150mmbl}

savemat('data/simulated_exponential_decline_rates.mat', decline_rates_simulations)
savemat('data/simulated_exponential_profiles_10mmbbl.mat', mmbl10)
savemat('data/simulated_exponential_profiles_25mmbbl.mat', mmbl25)
savemat('data/simulated_exponential_profiles_50mmbbl.mat', mmbl50)
savemat('data/simulated_exponential_profiles_100mmbbl.mat', mmbl100)
savemat('data/simulated_exponential_profiles_150mmbbl.mat', mmbl150)

print('Exporting to matlab data file...')
print('Finished exponential simulations...')
print('....')
print('....')

# Clean workspace to avoid arros in simulations
del K, a, decline_rates_simulations, dist, initial_production, loc
del upper_bound, lower_bound, num_years, optimal_dist, mmbl10
del mmbl25, mmbl50, mmbl100, mmbl150, rate, scale
del sim_decline_rates, simulated_profiles_10mmbl, simulated_profiles_25mmbl
del simulated_profiles_50mmbl, simulated_profiles_100mmbl, simulated_profiles_150mmbl
del x

# %% Hyperbolic model

# Calibration and distribution for parameters
# For Hyperbolic I simulate Beta and gamma and the constant will be set at mean values

print('Monte Carlo Simulations procedure to simulate hyperbolic production profiles...')
print('Fitting probability distributions...')

# Fit distributions for beta and gamma
dist_beta = distfit()
dist_beta.fit_transform(hyperbolic_parameters[:, 1])
optimal_beta = dist_beta.model['name']

dist_gamma = distfit()
dist_gamma.fit_transform(hyperbolic_parameters[:, 2])
optimal_gama = dist_gamma.model['name']

print('Best fitting distribution for parameters beta and gamma of hyperbolic model')
print('Beta parameter: ', optimal_beta)
print('Gamma parameter ', optimal_gama)

# Simulate parameters

# First fit distribution to find parameters and then generate the random
# Beta
c, loc, scale = st.pareto.fit(hyperbolic_parameters[:, 1])
sim_beta= st.pareto.rvs(c, loc, scale, size=10000, random_state=64)

# Gamma
c, loc, scale = st.gamma.fit(hyperbolic_parameters[:, 2])
sim_gamma= st.gamma.rvs(c, loc, scale, size=10000, random_state=64)

# Make simulations into tuple to then unzip in loop for the simulations
sim_hyperbolic_parameters = list(zip(sim_beta, sim_gamma))
num_years = 20

print('Simulating hyperbolic production profiles...')

# Simulate for each field size
# 10 MMbbl field
initial_production = list(initial_10m)
simulated_profiles_10mmbl = np.empty((0, 20))

for beta, gamma in sim_hyperbolic_parameters:

    x = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
    x = x[0:21]
    x = np.reshape(x, (1, 20))
    simulated_profiles_10mmbl = np.append(simulated_profiles_10mmbl, x, axis=0)
    simulated_profiles_10mmbl = simulated_profiles_10mmbl

simulated_profiles_10mmbl = simulated_profiles_10mmbl * 365

# 25 MMbbl field
initial_production = list(initial_25m)
simulated_profiles_25mmbl = np.empty((0, 20))

for beta, gamma in sim_hyperbolic_parameters:

    x = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
    x = x[0:20]
    x = np.reshape(x, (1, 20))
    simulated_profiles_25mmbl = np.append(simulated_profiles_25mmbl, x, axis=0)
    simulated_profiles_25mmbl = simulated_profiles_25mmbl

simulated_profiles_25mmbl = simulated_profiles_25mmbl * 365


# 50 MMbbl field
initial_production = list(initial_50m)
simulated_profiles_50mmbl = np.empty((0, 20))

for beta, gamma in sim_hyperbolic_parameters:

    x = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
    x = x[0:20]
    x = np.reshape(x, (1, 20))
    simulated_profiles_50mmbl = np.append(simulated_profiles_50mmbl, x, axis=0)
    simulated_profiles_50mmbl = simulated_profiles_50mmbl

simulated_profiles_50mmbl = simulated_profiles_50mmbl * 365



# 100 MMbbl field
initial_production = list(initial_100m)
simulated_profiles_100mmbl = np.empty((0, 20))

for beta, gamma in sim_hyperbolic_parameters:

    x = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
    x = x[0:20]
    x = np.reshape(x, (1, 20))
    simulated_profiles_100mmbl = np.append(simulated_profiles_100mmbl, x, axis=0)
    simulated_profiles_100mmbl = simulated_profiles_100mmbl

simulated_profiles_100mmbl = simulated_profiles_100mmbl * 365


# 150 MMbbl field
initial_production = list(initial_150m)
simulated_profiles_150mmbl = np.empty((0, 20))

for beta, gamma in sim_hyperbolic_parameters:

    x = simulate_hyperbolic_profile(initial_production, beta, gamma, num_years)
    x = x[0:20]
    x = np.reshape(x, (1, 20))
    simulated_profiles_150mmbl = np.append(simulated_profiles_150mmbl, x, axis=0)
    simulated_profiles_150mmbl = simulated_profiles_150mmbl

simulated_profiles_150mmbl = simulated_profiles_150mmbl * 365


# Save to python objects
hyperbolic_decline_profiles = {
    'simulated_hyperbolic_rates': hyperbolic_decline_rates,
    'mmbbl_10': simulated_profiles_10mmbl,
    'mmbbl_25': simulated_profiles_25mmbl,
    'mmbbl_50': simulated_profiles_50mmbl,
    'mmbbl_100': simulated_profiles_100mmbl,
    'mmbbl_150': simulated_profiles_150mmbl
}

# Export to matlab
from scipy.io import savemat

decline_rates_simulations = {'sim_decline_rates': hyperbolic_decline_rates}
mmbl10 = {'mmbbl_10': simulated_profiles_10mmbl}
mmbl25 = {'mmbbl_25': simulated_profiles_25mmbl}
mmbl50 = {'mmbbl_50': simulated_profiles_50mmbl}
mmbl100 = {'mmbbl_100': simulated_profiles_100mmbl}
mmbl150 = {'mmbbl_150': simulated_profiles_150mmbl}

savemat('data/simulated_hyperbolic_decline_rates.mat', decline_rates_simulations)
savemat('data/simulated_hyperbolic_profiles_10mmbbl.mat', mmbl10)
savemat('data/simulated_hyperbolic_profiles_25mmbbl.mat', mmbl25)
savemat('data/simulated_hyperbolic_profiles_50mmbbl.mat', mmbl50)
savemat('data/simulated_hyperbolic_profiles_100mmbbl.mat', mmbl100)
savemat('data/simulated_hyperbolic_profiles_150mmbbl.mat', mmbl150)

print('Exporting to matlab data file...')
print('Finished hyperbolic simulations...')
print('....')
print('....')

# Clean workspace to avoid across in simulations
del dist_beta, dist_gamma, initial_production
del c, loc, scale, beta, gamma, num_years, optimal_beta, optimal_gama
del mmbl10, mmbl25, mmbl50, mmbl100, mmbl150
del sim_beta, sim_gamma, simulated_profiles_10mmbl, simulated_profiles_25mmbl
del simulated_profiles_50mmbl, simulated_profiles_100mmbl, simulated_profiles_150mmbl
del x, sim_hyperbolic_parameters

# %% Logistic model

## To ensure we have sore of same initial values for all fields among
# different simulation models, I am setting a and K manually according to the ranges
# set in the initial profile calculations. Because those values change the shape of when
# production is growing. For the time when production is declining beta comes handy.

print('Monte Carlo Simulations procedure to simulate logistic production profiles...')
print('Fitting probability distributions...')

# Fit distributions for beta and gamma
dist_beta = distfit()
dist_beta.fit_transform(logistic_parameters[:, 1])
optimal_beta = dist_beta.model['name']

print('Best fitting distribution for parameters beta of logistic model')
print('Beta parameter: ', optimal_beta)

# Simulate parameters

# First fit distribution to find parameters and then generate the random
# Beta
a, b, loc, scale = st.beta.fit(logistic_parameters[:, 1])
sim_beta = st.beta.rvs(a=b, b=b, loc=loc, scale=scale, size=10000, random_state=64)
num_years = 20


# Simulate for each field size
# 10 MMbbl field
a = 1
K = 14000000
num_years = 20

simulated_profiles_10mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_10mmbl = np.append(simulated_profiles_10mmbl, x_logistic, axis=0)


# 25 MMbbl field
a = 1.5
K = 30000000
num_years = 20

simulated_profiles_25mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_25mmbl = np.append(simulated_profiles_25mmbl, x_logistic, axis=0)


# 50 MMbbl field
a = 2
K = 60000000
num_years = 20

simulated_profiles_50mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_50mmbl = np.append(simulated_profiles_50mmbl, x_logistic, axis=0)


# 100 MMbbl field
a = 2.4
K = 120000000
num_years = 20

simulated_profiles_100mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_100mmbl = np.append(simulated_profiles_100mmbl, x_logistic, axis=0)


# 150 MMbbl field
a = 2.4
K = 180000000
num_years = 20

simulated_profiles_150mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_150mmbl = np.append(simulated_profiles_150mmbl, x_logistic, axis=0)


# Save to python objects
logistic_decline_profiles = {
    'simulated_logistic_rates': sim_beta,
    'mmbbl_10': simulated_profiles_10mmbl,
    'mmbbl_25': simulated_profiles_25mmbl,
    'mmbbl_50': simulated_profiles_50mmbl,
    'mmbbl_100': simulated_profiles_100mmbl,
    'mmbbl_150': simulated_profiles_150mmbl
}

# Export to matlab
from scipy.io import savemat

decline_rates_simulations = {'sim_decline_rates': sim_beta}
mmbl10 = {'mmbbl_10': simulated_profiles_10mmbl}
mmbl25 = {'mmbbl_25': simulated_profiles_25mmbl}
mmbl50 = {'mmbbl_50': simulated_profiles_50mmbl}
mmbl100 = {'mmbbl_100': simulated_profiles_100mmbl}
mmbl150 = {'mmbbl_150': simulated_profiles_150mmbl}

savemat('data/simulated_logistic_decline_rates.mat', decline_rates_simulations)
savemat('data/simulated_logistic_profiles_10mmbbl.mat', mmbl10)
savemat('data/simulated_logistic_profiles_25mmbbl.mat', mmbl25)
savemat('data/simulated_logistic_profiles_50mmbbl.mat', mmbl50)
savemat('data/simulated_logistic_profiles_100mmbbl.mat', mmbl100)
savemat('data/simulated_logistic_profiles_150mmbbl.mat', mmbl150)

print('Exporting to matlab data file...')
print('Finished logistic simulations...')
print('....')
print('....')

print('Finished decline rate simulations procedure')

# Clean workspace to avoid across in simulations
del dist_beta
del a, K, b, loc, scale, beta, num_years, optimal_beta
del x_logistic, cumulative_rate, annualized_rate
del mmbl10, mmbl25, mmbl50, mmbl100, mmbl150
del sim_beta, simulated_profiles_10mmbl, simulated_profiles_25mmbl
del simulated_profiles_50mmbl, simulated_profiles_100mmbl, simulated_profiles_150mmbl

#%%
# Fix logistic decline simulations. 
# This fix was introduced because simulations for logistic model were incorrect.
# Get parameters to simulate decline rate (beta) of logistic function. 

a, b, loc, scale = st.beta.fit(logistic_parameters[:, 1])
sim_beta = st.beta.rvs(a=b, b=b, loc=loc, scale=scale, size=10000, random_state=64)


# The value of alpha is set manually based on sample profiles from the UKCS. This is because larger fields will tend to have larger values of alpha compared to smaller ones. 
# I am only interested in the simulations for the large fields as the smaller ones are unlikely to show logistic delcine based on analysis by Kemp and mine. 


# Profiles for the 100 MMmmbl Field
a = 15
K = 105000000
num_years = 20
simulated_profiles_100mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_100mmbl = np.append(simulated_profiles_100mmbl, x_logistic, axis=0)

for field_profile in simulated_profiles_100mmbl:
    plt.plot(field_profile)
title=f"Sample of simulated profiles for Logsitic model 100 MMbbl"
plt.title(title)
plt.show()   


print(np.max(np.sum(simulated_profiles_100mmbl, axis=1)/1000000))
print(np.min(np.sum(simulated_profiles_100mmbl, axis=1)/1000000))

# Profiles for the 150 MMmmbl Field
a = 20
K = 160000000
num_years = 20
simulated_profiles_150mmbl = np.empty((0, 20))

for beta in sim_beta:
    cumulative_rate, annualized_rate = simulate_logistic_profile(a, beta, K, num_years)
    # Do not Transform output into barrels per day.
    x_logistic = annualized_rate
    x_logistic = np.reshape(x_logistic, (1, 20))
    simulated_profiles_150mmbl = np.append(simulated_profiles_150mmbl, x_logistic, axis=0)

for field_profile in simulated_profiles_150mmbl:
    plt.plot(field_profile)
title=f"Sample of simulated profiles for Logsitic model 150 MMbbl"
plt.title(title)
plt.show()   


print(np.max(np.sum(simulated_profiles_150mmbl, axis=1)/1000000))
print(np.min(np.sum(simulated_profiles_150mmbl, axis=1)/1000000))

# End of fix


# %% Store results and profiles to use other python analysis. 
exponential_decline_profiles
hyperbolic_decline_profiles
logistic_decline_profiles = {'mmbbl_100':simulated_profiles_100mmbl,
                             'mmbbl_150':simulated_profiles_150mmbl}