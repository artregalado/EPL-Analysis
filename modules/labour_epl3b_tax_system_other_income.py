import numpy as np
import pandas as pd
from .aux_functions import discount_stream

# Tax calculations
# For UKCS overview of tax system see:
# https://www.nstauthority.co.uk/exploration-production/taxation/overview/
# For Energy Profits Levy see
# https://www.gov.uk/government/publications/autumn-statement-2022-energy-taxes-factsheet/energy-taxes-factsheet


class TaxSystemOtherIncome:
    def __init__(self, pre_tax_cashflow, tax_assumptions, market_assumptions):
        self.pre_tax_cashflow = pre_tax_cashflow
        self.tax_assumptions = tax_assumptions
        self.market_assumptions = market_assumptions

    def get_ncf(self):
        ncf = np.array(self.pre_tax_cashflow.ncf)
        return ncf

    def get_devex(self):
        devex = np.array(self.pre_tax_cashflow.devex)
        return devex

    def get_opex(self):
        opex = np.array(self.pre_tax_cashflow.opex)
        return opex

    def get_decommx(self):
        decommx = np.array(self.pre_tax_cashflow.decommx)
        return decommx

    def calculate_corporate_tax(self):
        # Calculate tax saved for CT from first year capital allowance
        # Calculate taxable income for CT, only payable when there are profits, and then
        # the corporate tax to be paid.

        profits = self.get_ncf()
        devex = self.get_devex()

        ct_saved = devex * self.tax_assumptions['ct_rate']

        ct_paid = []
        for profit in np.nditer(profits):
            if profit > 0:
                paid = profit * self.tax_assumptions['ct_rate']
                ct_paid.append(paid)
            else:
                paid = 0
                ct_paid.append(paid)
        ct_paid = np.array(ct_paid)
        return ct_paid, ct_saved

    def calculate_supplementary_charge(self):
        # Calculate tax saved for SC from first year capital allowance Calculate taxable income
        # for SC. Investment allowance for supplementary charge might reduce the value of taxable
        # income for SC to cero. Recall the IA can only be claimed when there are profits.

        profits = self.get_ncf()
        devex = self.get_devex()

        sc_saved = devex * self.tax_assumptions['sc_rate']

        # Get the bag of IA for SC available to offset against profits and then calculate sc paid
        ia_bag = devex.sum() * self.tax_assumptions['ia_for_sc_rate']
        taxable_income_for_sc = []

        for profit in np.nditer(profits):
            # If there are no profits, IA does not apply as no SC payable
            if profit <= 0:
                taxable_income_for_sc.append(0)

            # If there are profits, then check how much is in the bag, and reduce by the amount of profits
            # Taxable income will be cero if there is IA available on the bag, or positive if the bag
            # doesn't cover or there is no more bag.
            else:
                if ia_bag >= profit:
                    taxable_income_for_sc.append(0)
                    ia_bag -= profit
                else:
                    taxable_income_for_sc.append(np.abs(ia_bag - profit))
                    ia_bag = 0
                    # The code for the else condition is a trick. There might be the case that
                    # profits remaining in the bag are not enough to cover the profits. In that
                    # case taxable income for SC will be reduced but not completely. What needs
                    # to be paid is the difference between what's in the bag and profits. Because
                    # the result will be negative as ia_bag < profits, then we use the absolute
                    # value to return it to positive. We then set the bag to cero and this case
                    # will not happen again. To avoid coding more if or for loops, the code (
                    # np.abs) also ensures that once the bag is cero, taxable income and profits
                    # will be the same as the difference will amount to the profits.

        sc_paid = np.array(taxable_income_for_sc) * self.tax_assumptions['sc_rate']
        return sc_paid, sc_saved

    def calculate_energy_profits_levy(self):

        profits = self.get_ncf()
        devex = self.get_devex()

        epl_saved_from_ia = devex * self.tax_assumptions['epl_rate'] * self.tax_assumptions['ia_for_epl_rate']
        epl_saved_from_first_year = devex * self.tax_assumptions['epl_rate']
        epl_saved = epl_saved_from_ia + epl_saved_from_first_year
        
        # In case of Labour EPL3b there is no savings from IA or first year
        epl_saved = np.zeros_like(epl_saved)
         

        epl_paid = []
        for profit in np.nditer(profits):
            if profit > 0:
                paid = profit * self.tax_assumptions['epl_rate']
                epl_paid.append(paid)
            else:
                paid = 0
                epl_paid.append(paid)

        # Restrict the payments and saving only to the years where the epl will be active. All
        # years outside of the applicable years are cero. Base case assumptions is 2022 so moving
        # beginning year is like shifting the time of projects to left. So if the
        # epl_beginning_year assumptions is 3, then this shifts 2022 three years to the left to
        # 2019. If the beginning year is 0 then the first year is 2022.

        epl_beginning_year = self.tax_assumptions['epl_beginning_year']
        epl_duration = self.tax_assumptions['epl_beginning_year'] + self.tax_assumptions['epl_years']

        epl_paid_adjusted = []
        for t in np.arange(len(epl_paid)):
            if epl_beginning_year <= t < epl_duration:
                epl_paid_adjusted.append(epl_paid[t])
            else:
                epl_paid_adjusted.append(0)
        epl_paid_adjusted = np.array(epl_paid_adjusted)

        epl_saved_adjusted = []
        for t in np.arange(len(epl_saved)):
            if epl_beginning_year <= t < epl_duration:
                epl_saved_adjusted.append(epl_saved[t])
            else:
                epl_saved_adjusted.append(0)
        epl_saved_adjusted = np.array(epl_saved_adjusted)

        return epl_paid_adjusted, epl_saved_adjusted

    def calculate_decomm_relief(self):
        decommx = self.get_decommx()
        saved_from_decommx = decommx * self.tax_assumptions['decommx_relief_rate']
        return saved_from_decommx

    def calculate_post_tax_cashflow_no_epl(self):
        ct_paid, ct_saved = self.calculate_corporate_tax()
        sc_paid, sc_saved = self.calculate_supplementary_charge()
        decomm_saved = self.calculate_decomm_relief()

        # If EPL-b from labour is approved, this means the EPL capitall allowance is
        # not applicable, and thus there are no savings from EPL. 
        tax_savings = ct_saved + sc_saved + decomm_saved
        tax_paid = ct_paid + sc_paid
        post_tax_ncf = self.pre_tax_cashflow['ncf'] + tax_savings - tax_paid

        post_tax_cashflow = self.pre_tax_cashflow.copy()
        post_tax_cashflow['ct_paid'] = ct_paid
        post_tax_cashflow['sc_paid'] = sc_paid
        post_tax_cashflow['tax_savings'] = tax_savings
        post_tax_cashflow['tax_paid'] = tax_paid
        post_tax_cashflow['post_tax_ncf'] = post_tax_ncf

        return post_tax_cashflow

    def calculate_post_tax_cashflow_with_epl(self):
        ct_paid, ct_saved = self.calculate_corporate_tax()
        sc_paid, sc_saved = self.calculate_supplementary_charge()
        epl_paid, epl_saved = self.calculate_energy_profits_levy()
        decomm_saved = self.calculate_decomm_relief()


        tax_savings = ct_saved + sc_saved + epl_saved +decomm_saved
        tax_paid = ct_paid + sc_paid + epl_paid
        post_tax_ncf = self.pre_tax_cashflow['ncf'] + tax_savings - tax_paid

        post_tax_cashflow = self.pre_tax_cashflow.copy()
        post_tax_cashflow['ct_paid'] = ct_paid
        post_tax_cashflow['sc_paid'] = sc_paid
        post_tax_cashflow['epl_paid'] = epl_paid
        post_tax_cashflow['tax_savings'] = tax_savings
        post_tax_cashflow['tax_paid'] = tax_paid
        post_tax_cashflow['post_tax_ncf'] = post_tax_ncf

        return post_tax_cashflow

    def calculate_post_tax_cashflow(self, include_epl=True):
        if include_epl:
            post_tax_cashflow = self.calculate_post_tax_cashflow_with_epl()
        else:
            post_tax_cashflow = self.calculate_post_tax_cashflow_no_epl()
        return post_tax_cashflow

    def calculate_post_tax_economics(self, epl_case):

        # All values calculated in real terms and discounted for present value

        # Post Tax NPV
        post_tax_cashflow = self.calculate_post_tax_cashflow(include_epl=epl_case)
        net_cashflow = post_tax_cashflow['post_tax_ncf'].values
        deflated_cashflow = discount_stream(net_cashflow, self.market_assumptions['cpi'])
        discounted_cashflow = discount_stream(deflated_cashflow, self.market_assumptions['discount_factor'])
        post_tax_npv = discounted_cashflow.sum()

        # Tax saved
        post_tax_cashflow = self.calculate_post_tax_cashflow(include_epl=epl_case)
        net_cashflow = post_tax_cashflow['tax_savings'].values
        deflated_cashflow = discount_stream(net_cashflow, self.market_assumptions['cpi'])
        discounted_cashflow = discount_stream(deflated_cashflow, self.market_assumptions['discount_factor'])
        tax_saved = discounted_cashflow.sum()

        # Tax paid
        post_tax_cashflow = self.calculate_post_tax_cashflow(include_epl=epl_case)
        net_cashflow = post_tax_cashflow['tax_paid'].values
        deflated_cashflow = discount_stream(net_cashflow, self.market_assumptions['cpi'])
        discounted_cashflow = discount_stream(deflated_cashflow, self.market_assumptions['discount_factor'])
        tax_paid = discounted_cashflow.sum()

        # NPVI/Ratio
        devex = self.get_devex()
        devex_deflated = discount_stream(devex, self.market_assumptions['cpi'])
        devex_discounted = discount_stream(devex_deflated, self.market_assumptions['discount_factor'])
        ratio = post_tax_npv/devex_discounted.sum()

        results = {'real_post_tax_npv': post_tax_npv,
                   'tax_saved_real_pv': tax_saved,
                   'tax_paid_real_pv': tax_paid,
                   'npv/i_ratio': ratio}

        return results




