import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Analysis example
from modules.modelling_assumptions import field_10m_assum
from modules.modelling_assumptions import field_50m_assum
from modules.modelling_assumptions import field_100m_assum
from modules.modelling_assumptions import market_assumptions
from modules.modelling_assumptions import tax_assumptions
from economic_analysis_class import EconomicAnalysis, EconomicResultsGenerator
from modules.pre_tax_calculations import PreTaxSystem
from modules.tax_system_other_income import TaxSystemOtherIncome
from modules.tax_system_no_other_income import TaxSystemWithoutOtherIncome

# %% Get results for analysis and plotting

# Important assumptions that need to be set out in the paper.
# The original EPL began May 2022 and was to finish Dec 2025, the new EPL extends until March 2028
# For simplicty and because the model runs in years I assum orginal EPL lasts 4 years (2022-2025) assuming it
# includes all of 2022.
# For the new EPL it is 6 years, only two are added (2022-2027) this is to avid adjustments to first quarters of 2022
# and 2028; the net impact will not be meaningful as we are assume 5 more months in 2022 and 3 less of 2028.
# See EPL factsheet for more details
# https://www.gov.uk/government/publications/changes-to-the-energy-oil-and-gas-profits-levy/energy-oil-and-gas-profits-levy

# Case where with permanent system and summer EPL
small_no_epl = EconomicAnalysis(field_10m_assum, market_assumptions, tax_assumptions, epl_case=False)
small_epl_summer = EconomicAnalysis(field_10m_assum, market_assumptions, tax_assumptions, epl_case=True)

# Case of new EPL
market_assumptions['epl_years'] = 6
small_epl_autumn = EconomicAnalysis(field_10m_assum, market_assumptions, tax_assumptions, epl_case=True)

# Case of new EPL and field starting 2019
market_assumptions['epl_beginning_year'] = 3
small_epl_autumn_2019 = EconomicAnalysis(field_10m_assum, market_assumptions, tax_assumptions, epl_case=True)
market_assumptions['epl_years'] = 4  # Return to base case of summer
market_assumptions['epl_beginning_year'] = 0  # Return to base case of summer

# %%

print(small_no_epl.get_post_tax_economics_other_income())
print(small_no_epl.get_post_tax_cashflow_other_income())
print(small_no_epl.get_pre_tax_cashflow())

# %%

EconomicResultsGenerator(field_10m_assum, market_assumptions, tax_assumptions)


# %%

# plt.plot(pre_tax_cashflow['ncf'], label='pre-tax', c='grey')
# plt.plot(post_tax_cashflow['post_tax_ncf'], label="other", c='r')
# plt.plot(post_tax_cashflow2['post_tax_ncf'], label='other no EPL', c='r', ls='--')
# plt.plot(post_tax_cashflow3['post_tax_ncf'], label='no other', c='b')
# plt.plot(post_tax_cashflow4['post_tax_ncf'], label='no other no EPL', c='b', ls='--')
# plt.legend()
# plt.show()
