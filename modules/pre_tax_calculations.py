import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .aux_functions import compound_value
from .aux_functions import discount_stream

class PreTaxSystem:
    def __init__(self, oil_field_assumptions, market_assumptions):
        self.oil_field_assumptions = oil_field_assumptions
        self.market_assumptions = market_assumptions

    def plot_production_profile_mmboey(self):
        profile = np.array(self.oil_field_assumptions['production_profile'])
        fig, ax = plt.subplots()
        ax.plot(profile / 1000000)
        plt.title(
            f"Production profile for {self.oil_field_assumptions['reserves'] / 1000000} MMboe field")
        plt.xlabel("Project year")
        plt.ylabel("MMboe/y")
        return ax

    def create_production_series(self):
        assum = self.oil_field_assumptions
        production_series = pd.Series(data=assum['production_profile'],
                                      index=range(assum['production_starts'],
                                                  assum['production_starts'] + len(
                                                      assum['production_profile'])
                                                  ),
                                      name="production")
        return production_series

    def create_devex_series(self):
        assum = self.oil_field_assumptions
        total_devex = assum['devex'] * assum['reserves']
        devex_schedule = total_devex * np.array(assum['development_schedule'])
        devex_series = pd.Series(data=devex_schedule, name="devex")
        # No index as devex assumed to always begin at time zero
        return devex_series

    def create_opex_series(self):
        assum = self.oil_field_assumptions
        devex_series = self.create_devex_series()
        production_series = self.create_production_series()

        opex_initial = devex_series.cumsum() * assum['opex']
        opex_repeated = np.repeat(np.max(opex_initial), max(production_series.index) + 1)
        np.put(opex_repeated, list(opex_initial.index), opex_initial)
        opex_series = pd.Series(opex_repeated, name="opex")
        return opex_series

    def create_production_and_costs_table(self):
        production_series = self.create_production_series()
        devex_series = self.create_devex_series()
        opex_series = self.create_opex_series()
        df = devex_series.to_frame().join(production_series, how="outer").join(opex_series, how="outer")

        # Tidy data table by setting opex costs only when there is production and NaN values to cero
        df.fillna(0, inplace=True)
        df.loc[df['production'] == 0, 'opex'] = 0
        return df

    # Group of functions designed to create a pre-tax cashflow

    def create_deterministic_oil_price_path(self):
        production_costs_table = self.create_production_and_costs_table()
        oil_in_mod = compound_value(initial_value=self.market_assumptions['oil_price'],
                                    interest_rate=self.market_assumptions['cpi'],
                                    n_years=max(production_costs_table.index) + 1)
        return oil_in_mod

    def create_opex_escalated(self):
        production_costs_table = self.create_production_and_costs_table()
        opex_escalated = []
        for time, value in production_costs_table.opex.items():
            new = value * ((1 + self.market_assumptions['spi']) ** float(time))
            opex_escalated.append(new)
        opex_escalated = pd.Series(opex_escalated, name='opex')
        return opex_escalated

    def create_devex_escalated(self):
        production_costs_table = self.create_production_and_costs_table()
        devex_escalated = []
        for time, value in production_costs_table.devex.items():
            new = value * ((1 + self.market_assumptions['spi']) ** float(time))
            devex_escalated.append(new)
        devex_escalated = pd.Series(devex_escalated, name='devex')
        return devex_escalated

    def create_revenues_path(self):
        production_costs_table = self.create_production_and_costs_table()
        oil_price_path = self.create_deterministic_oil_price_path()
        revenues = oil_price_path * production_costs_table['production']
        revenues.rename("revenues", inplace=True)
        return revenues

    def calculate_cop_index_and_year_pre_tax(self):
        production_costs_table = self.create_production_and_costs_table()
        revenues = self.create_revenues_path()
        opex = self.create_opex_escalated()
        flag = revenues < opex
        cop_index = flag.idxmax()
        cop_year = cop_index + 1
        return cop_index, cop_year

    def create_decommx_escalated(self):
        cop_index, cop_year = self.calculate_cop_index_and_year_pre_tax()
        total_devex = self.oil_field_assumptions['devex'] * self.oil_field_assumptions['reserves']
        total_decommx = total_devex * self.oil_field_assumptions['decommx']
        decomm_schedule = total_decommx * np.array(self.oil_field_assumptions['decomm_schedule'])
        index = np.arange(cop_index + 1, cop_index + 1 + len(decomm_schedule))
        decommx_series = pd.Series(data=decomm_schedule, name="decommx", index=index)
        decommx_escalated = []
        for time, value in decommx_series.items():
            new = value * ((1 + self.market_assumptions['spi']) ** float(time))
            decommx_escalated.append(new)
        decommx_escalated = pd.Series(data=decommx_escalated, name='decommx', index=index)

        return decommx_escalated, max(index)

    def create_pre_tax_cashflow(self):
        revenues = self.create_revenues_path()
        opex = self.create_opex_escalated()
        devex = self.create_devex_escalated()
        cop_index, cop_year = self.calculate_cop_index_and_year_pre_tax()
        decommx, max_cashflow_index = self.create_decommx_escalated()

        cashflow = revenues.to_frame().join(devex, how='outer').join(opex, how='outer').join(decommx,
                                                                                             how='outer')
        # Set cash flow in millions and tidy up. Notably, limiting the cash flow to the decommissioning times
        # and making sure that during decomm there is no prodution, devex and opex, onlhy decommx.
        cashflow.fillna(0, inplace=True)
        cashflow = cashflow / 1000000
        cashflow = cashflow.iloc[0:max_cashflow_index + 1]
        cashflow.loc[cop_index + 1:, ['revenues', 'devex', 'opex']] = 0
        cashflow['ncf'] = cashflow['revenues'] - cashflow['opex'] - cashflow['devex'] - cashflow[
            'decommx']
        return cashflow

    def calculate_pre_tax_economics(self):
        pre_tax_cashflow = self.create_pre_tax_cashflow()
        net_cashflow = pre_tax_cashflow['ncf'].values
        deflated_cashflow = discount_stream(net_cashflow, discount_rate=self.market_assumptions['cpi'])
        discounted_cashflow = discount_stream(deflated_cashflow,
                                              discount_rate=self.market_assumptions['discount_factor'])
        real_pre_tax_npv = discounted_cashflow.sum()

        cop_flag = pre_tax_cashflow['revenues'] < pre_tax_cashflow['opex']
        cop_index = cop_flag.idxmax()

        economics = {'real_pre_tax_npv': real_pre_tax_npv,
                     'cop_date': cop_index}

        return economics

    def plot_pre_tax_cashflow(self):
        pre_tax_cashflow = self.create_pre_tax_cashflow()
        # Create title including size of field and economics
        economics = self.calculate_pre_tax_economics()
        npv = np.round(economics['real_pre_tax_npv'], 2)
        cop_date = economics['cop_date']
        reserves = int(self.oil_field_assumptions['reserves'] / 1000000)
        title = f"Pre-tax cashflow of {reserves} MMboe field. \n NPV = ${npv} million; COP date = year {cop_date}."

        plt.plot(pre_tax_cashflow.ncf, label='Net Cash Flow', ls='--')
        plt.plot(pre_tax_cashflow.revenues, label='Revenues')
        plt.plot(pre_tax_cashflow.opex, label='Opex')
        plt.plot(pre_tax_cashflow.devex, label='Devex')
        plt.plot(pre_tax_cashflow.decommx, label='Decommx')

        plt.title(title)
        plt.xlabel("Project year")
        plt.ylabel("Million USD")
        plt.legend()

    def is_pre_tax_npv_negative(self):
        economics = self.calculate_pre_tax_economics()
        npv = economics['real_pre_tax_npv']
        flag = npv <= 0
        return flag

    def is_pre_tax_npvi_hurdle_passed(self):

        economics = self.calculate_pre_tax_economics()
        devex_series = self.create_devex_escalated()
        npvi_hurdle = self.market_assumptions['npvi_ratio']

        deflated_cashflow = discount_stream(devex_series.values, discount_rate=self.market_assumptions['spi'])
        discounted_cashflow = discount_stream(deflated_cashflow,
                                              discount_rate=self.market_assumptions['discount_factor'])
        devex = discounted_cashflow.sum()
        npv = economics['real_pre_tax_npv']
        flag = (npv / devex) < npvi_hurdle
        return flag
