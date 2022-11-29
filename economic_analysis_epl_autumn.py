import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Analysis example
from modules.modelling_assumptions import field_10m_assum
from modules.modelling_assumptions import field_50m_assum
from modules.modelling_assumptions import field_100m_assum
from modules.modelling_assumptions import market_assumptions
from modules.modelling_assumptions import tax_assumptions
from modules.modelling_assumptions import tax_assumptions_autumn_epl
from modules.modelling_assumptions import tax_assumptions_delayed_autumn_epl
from economic_analysis_class import calculate_economic_results
from economic_analysis_class import calculate_cashflow_results

# %% Get results for analysis and plotting

# Important assumptions that need to be set out in the paper.
# The original EPL began May 2022 and was to finish Dec 2025, the new EPL extends until March 2028
# For simplicty and because the model runs in years I assum orginal EPL lasts 4 years (2022-2025) assuming it
# includes all of 2022.
# For the new EPL it is 6 years, only two are added (2022-2027) this is to avid adjustments to first quarters of 2022
# and 2028; the net impact will not be meaningful as we are assume 5 more months in 2022 and 3 less of 2028.
# See EPL factsheet for more details
# https://www.gov.uk/government/publications/changes-to-the-energy-oil-and-gas-profits-levy/energy-oil-and-gas-profits-levy

field_assumptions = field_10m_assum
results_small_field = calculate_economic_results(field_assumptions=field_assumptions,
                                                 market_assumptions=market_assumptions,
                                                 tax_assumptions=tax_assumptions,
                                                 tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                 tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

field_assumptions = field_50m_assum
results_medium_field = calculate_economic_results(field_assumptions=field_assumptions,
                                                  market_assumptions=market_assumptions,
                                                  tax_assumptions=tax_assumptions,
                                                  tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                  tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

field_assumptions = field_100m_assum
results_large_field = calculate_economic_results(field_assumptions=field_assumptions,
                                                 market_assumptions=market_assumptions,
                                                 tax_assumptions=tax_assumptions,
                                                 tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                 tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

field_assumptions = field_10m_assum
cashflow_small_field = calculate_cashflow_results(field_assumptions=field_assumptions,
                                                  market_assumptions=market_assumptions,
                                                  tax_assumptions=tax_assumptions,
                                                  tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                  tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

field_assumptions = field_50m_assum
cashflow_medium_field = calculate_cashflow_results(field_assumptions=field_assumptions,
                                                   market_assumptions=market_assumptions,
                                                   tax_assumptions=tax_assumptions,
                                                   tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                   tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

field_assumptions = field_100m_assum
cashflow_large_field = calculate_cashflow_results(field_assumptions=field_assumptions,
                                                  market_assumptions=market_assumptions,
                                                  tax_assumptions=tax_assumptions,
                                                  tax_assumptions_autumn_epl=tax_assumptions_autumn_epl,
                                                  tax_assumptions_delayed_autumn_epl=tax_assumptions_delayed_autumn_epl)

# %%

# plt.plot(pre_tax_cashflow['ncf'], label='pre-tax', c='grey')
# plt.plot(post_tax_cashflow['post_tax_ncf'], label="other", c='r')
# plt.plot(post_tax_cashflow2['post_tax_ncf'], label='other no EPL', c='r', ls='--')
# plt.plot(post_tax_cashflow3['post_tax_ncf'], label='no other', c='b')
# plt.plot(post_tax_cashflow4['post_tax_ncf'], label='no other no EPL', c='b', ls='--')
# plt.legend()
# plt.show()
