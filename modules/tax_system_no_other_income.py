import numpy as np
import pandas as pd
from .aux_functions import discount_stream
from .tax_system_other_income import TaxSystemOtherIncome


# Tax calculations
# For UKCS overview of tax system see:
# https://www.nstauthority.co.uk/exploration-production/taxation/overview/
# For Energy Profits Levy see
# https://www.gov.uk/government/publications/autumn-statement-2022-energy-taxes-factsheet/energy-taxes-factsheet

# The difficulty in this tax system is that there is no income for the first year allowance to be offset.
# Therefore the losses are carried forward, and in the case of decommissioning, carried backwards and then
# the money is claimed. The carried back losses increase by a rate called RFES to account for the time value
# of money. RFES is only availble for the first 10 years.

class TaxSystemWithoutOtherIncome(TaxSystemOtherIncome):

    def calculate_taxable_income_with_rfes_for_ct_and_sc(self):
        profits = self.get_ncf()

        # Set bag of losses that will be carried forward.
        # and taxable income for both CT and SC
        # Because RFES only applies for the first 10 years, then restrict the carrying back to 10 years.
        rfes_bag = 0
        taxable_income = []

        for t, profit in enumerate(profits):
            # Increase the amount in bag by the RFES rate each year. In year cero because its cero, the rfes
            #  won't have any effect so the code can be simplified.
            rfes_bag = rfes_bag * (1 + self.tax_assumptions['rfes_rate'])

            # If less than ten years, then RFES applies
            if t < 10:
                # If there are losses add to the bag, use the absolute to change negative values to positive.
                # taxable income would be zero in this case.
                if profit < 0:
                    rfes_bag += np.abs(profit)
                    taxable_income.append(0)

                # If there are profits, check against the bag. If the bag is higher, then reduce the bag by the amount
                # of profits and taxable income would be zero; if the bag is lower than profits, then consume all the bag
                # and taxable income is the remaining amount.
                # Here we are using the same trick as calculate_supplementary_charge() function. See comments
                # there for explanation.
                else:
                    if rfes_bag > profit:
                        rfes_bag -= profit
                        taxable_income.append(0)
                    else:
                        taxable_income.append(np.abs(rfes_bag - profit))
                        rfes_bag = 0

            # If more than ten years, then RFES does not apply. And taxable income = profit
            else:
                if profit < 0:
                    taxable_income.append(0)
                else:
                    taxable_income.append(profit)

        taxable_income = np.array(taxable_income)


        # Now do the calculations for taxable after considering losses carried back from decommissioning.
        flipped_taxable = np.flip(taxable_income)
        decommx = self.get_decommx()

        # Carryback limit is the number of years that decommissioning expenditures are occuring + 3 years (the COP years plus 2 more)
        carryback_limit = (len(decommx[decommx > 0]) + 3) - 1  # -1 used to adjust for python 0 indexing

        # For decommissioning we can assume relief to be at 40% because of the DRD that actually guarantees the rate to be such.
        # The key point is how to assign it to the taxable income flow for CT and SC. Can only be claimed back max 3 years.
        # Source for carry back limit is years https://www.gov.uk/hmrc-internal-manuals/oil-taxation-manual/ot21060

        decomm_relief_bag = decommx.sum() * (self.tax_assumptions['ct_rate'] + self.tax_assumptions['sc_rate'])
        taxable_adjusted_for_decomm_relief = []
        for t, profit in enumerate(flipped_taxable):
            if t <= carryback_limit:
                if profit <= 0:
                    taxable_adjusted_for_decomm_relief.append(0)
                else:
                    if profit < decomm_relief_bag:
                        taxable_adjusted_for_decomm_relief.append(0)
                        decomm_relief_bag -= profit
                    else:
                        taxable_adjusted_for_decomm_relief.append((np.abs(decomm_relief_bag - profit)))
                        decomm_relief_bag = 0
            else:
                taxable_adjusted_for_decomm_relief.append(profit)
        taxable_adjusted_for_decomm_relief = np.array(taxable_adjusted_for_decomm_relief)
        taxable_adjusted_for_decomm_relief = np.flip(taxable_adjusted_for_decomm_relief)

        return taxable_adjusted_for_decomm_relief


    def calculate_corporate_tax(self):
        taxable_income = self.calculate_taxable_income_with_rfes_for_ct_and_sc()
        corporate_tax_paid = taxable_income * self.tax_assumptions['ct_rate']
        return corporate_tax_paid

    def calculate_supplementary_charge_no_other_income(self):
        # For the tax case with other income, the operator must either use the Supplementary charge from RFES or the
        # investment allowance from the IA. Cannot have both because it would be considered double accounting. So it is
        # important to check SC with investment allowance against SC with RFES and keep the one where payment is lower.
        # Can use the function from above to make this choice
        taxable_income = self.calculate_taxable_income_with_rfes_for_ct_and_sc()
        sc__paid_with_rfes = taxable_income * self.tax_assumptions['sc_rate']
        sc_paid_with_ia, _ = self.calculate_supplementary_charge()
        supplementary_charge_paid = np.minimum(sc__paid_with_rfes, sc_paid_with_ia)
        return supplementary_charge_paid

    def calculate_energy_profits_levy(self):
        # Note that decommissioning expenditures are not deducted to calculate the taxable income for EPL.
        profits = self.pre_tax_cashflow['revenues'] - self.pre_tax_cashflow['devex'] - self.pre_tax_cashflow.opex
        profits = np.array(profits)
        devex = self.get_devex()

        epl_saved_from_ia = devex * self.tax_assumptions['epl_rate'] * self.tax_assumptions['ia_for_epl_rate']

        epl_paid = []
        for profit in np.nditer(profits):
            if profit > 0:
                paid = profit * self.tax_assumptions['epl_rate']
                epl_paid.append(paid)
            else:
                paid = 0
                epl_paid.append(paid)
        epl_paid = np.array(epl_paid)

        # Restrict the payments and saving only to the years where the epl will be active. All years outside
        # of the applicable years are cero. Base case assumptions is 2022 so moving beginning year is like shifting the
        # time of projects to left. So if the epl_beginning_year assumptions is 3, then this shifts 2022 three years
        # to the left to 2019.

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
        for t in np.arange(len(epl_saved_from_ia)):
            if epl_beginning_year <= t < epl_duration:
                epl_saved_adjusted.append(epl_saved_from_ia[t])
            else:
                epl_saved_adjusted.append(0)
        epl_saved_adjusted = np.array(epl_saved_adjusted)

        return epl_paid_adjusted, epl_saved_adjusted

    def calculate_post_tax_cashflow_no_epl(self):
        ct_paid = self.calculate_corporate_tax()
        sc_paid = self.calculate_supplementary_charge_no_other_income()

        tax_paid = ct_paid + sc_paid
        post_tax_ncf = self.pre_tax_cashflow['ncf'] - tax_paid

        post_tax_cashflow = self.pre_tax_cashflow.copy()
        post_tax_cashflow['ct_paid'] = ct_paid
        post_tax_cashflow['sc_paid'] = sc_paid
        post_tax_cashflow['tax_paid'] = tax_paid
        post_tax_cashflow['post_tax_ncf'] = post_tax_ncf

        return post_tax_cashflow

    def calculate_post_tax_cashflow_with_epl(self):
        ct_paid = self.calculate_corporate_tax()
        sc_paid = self.calculate_supplementary_charge_no_other_income()
        epl_paid, epl_saved = self.calculate_energy_profits_levy()

        tax_savings = epl_saved
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

    def calculate_post_tax_economics(self, epl_case):

        # All values calculated in real terms and discounted for present value

        # Post Tax NPV
        post_tax_cashflow = self.calculate_post_tax_cashflow(include_epl=epl_case)
        net_cashflow = post_tax_cashflow['post_tax_ncf'].values
        deflated_cashflow = discount_stream(net_cashflow, self.market_assumptions['cpi'])
        discounted_cashflow = discount_stream(deflated_cashflow, self.market_assumptions['discount_factor'])
        post_tax_npv = discounted_cashflow.sum()

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
        ratio = post_tax_npv / devex_discounted.sum()

        results = {'real_post_tax_npv': post_tax_npv,
                   'tax_paid_real_pv': tax_paid,
                   'npv/i_ratio': ratio}

        return results

