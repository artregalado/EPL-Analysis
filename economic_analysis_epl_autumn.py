import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Analysis example
from modules.modelling_assumptions import field_10m_assum
from modules.modelling_assumptions import field_50m_assum
from modules.modelling_assumptions import field_100m_assum
from modules.modelling_assumptions import market_assumptions
from modules.modelling_assumptions import tax_assumptions
from modules.pre_tax_calculations import PreTaxSystem
from modules.tax_system_other_income import TaxSystemOtherIncome
from modules.tax_system_no_other_income import TaxSystemWithoutOtherIncome

pre_tax_model = PreTaxSystem(field_50m_assum, market_assumptions)
print(pre_tax_model.create_production_series())
pre_tax_cashflow = pre_tax_model.create_pre_tax_cashflow()
pre_tax_economics = pre_tax_model.calculate_pre_tax_economics()
print(pre_tax_economics)
pre_tax_model.plot_pre_tax_cashflow()
plt.show()

tax_system = TaxSystemOtherIncome(pre_tax_cashflow, tax_assumptions, market_assumptions)
post_tax_cashflow = tax_system.calculate_post_tax_cashflow(include_epl=True)
post_tax_economics = tax_system.calculate_post_tax_economics(epl_case=True)
print(post_tax_economics)

tax_system2 = TaxSystemOtherIncome(pre_tax_cashflow, tax_assumptions, market_assumptions)
post_tax_cashflow2 = tax_system2.calculate_post_tax_cashflow(include_epl=False)
post_tax_economics2 = tax_system2.calculate_post_tax_economics(epl_case=False)
print(post_tax_economics2)

tax_system3 = TaxSystemWithoutOtherIncome(pre_tax_cashflow, tax_assumptions, market_assumptions)
post_tax_cashflow3 = tax_system3.calculate_post_tax_cashflow(include_epl=True)
post_tax_economics3 = tax_system3.calculate_post_tax_economics(epl_case=True)
print(post_tax_economics3)

tax_system4 = TaxSystemWithoutOtherIncome(pre_tax_cashflow, tax_assumptions, market_assumptions)
post_tax_cashflow4 = tax_system4.calculate_post_tax_cashflow(include_epl=False)
post_tax_economics4 = tax_system4.calculate_post_tax_economics(epl_case=False)
print(post_tax_economics4)

# %%
plt.plot(pre_tax_cashflow['ncf'], label='pre-tax', c='grey')
plt.plot(post_tax_cashflow['post_tax_ncf'], label="other", c='r')
plt.plot(post_tax_cashflow2['post_tax_ncf'], label='other no EPL', c='r', ls='--')
plt.plot(post_tax_cashflow3['post_tax_ncf'], label='no other', c='b')
plt.plot(post_tax_cashflow4['post_tax_ncf'], label='no other no EPL', c='b', ls='--')
plt.legend()
plt.show()