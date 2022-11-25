import numpy as np
import pandas as pd
from modules.pre_tax_calculations import PreTaxSystem
from modules.tax_system_other_income import TaxSystemOtherIncome
from modules.tax_system_no_other_income import TaxSystemWithoutOtherIncome


class EconomicAnalysis:
    def __init__(self, field_assumptions, market_assumptions, tax_assumptions, epl_case):
        self.field_assumptions = field_assumptions
        self.market_assumptions = market_assumptions
        self.tax_assumptions = tax_assumptions
        self.epl_case = epl_case

    # Pre tax economics
    def create_pre_tax_model(self):
        model = PreTaxSystem(self.field_assumptions, self.market_assumptions)
        return model

    def get_pre_tax_cashflow(self):
        model = self.create_pre_tax_model()
        pre_tax_cashflow = model.create_pre_tax_cashflow()
        pre_tax_cashflow = np.round(pre_tax_cashflow / self.market_assumptions['exchange_rate'])
        return pre_tax_cashflow

    def get_pre_tax_economics(self):
        model = self.create_pre_tax_model()
        pre_tax_economics = pd.Series(model.calculate_pre_tax_economics())
        pre_tax_economics['real_pre_tax_npv'] = np.round(pre_tax_economics['real_pre_tax_npv']/ self.market_assumptions['exchange_rate'])
        return pre_tax_economics

    # Post tax economics case other income available

    def create_tax_model_other_income(self):
        pre_tax_cashflow = self.get_pre_tax_cashflow()
        model = TaxSystemOtherIncome(pre_tax_cashflow, self.tax_assumptions, self.market_assumptions)
        return model

    def get_post_tax_cashflow_other_income(self):
        model = self.create_tax_model_other_income()
        post_tax_cashflow = model.calculate_post_tax_cashflow(include_epl=self.epl_case)
        post_tax_cashflow = np.round(post_tax_cashflow)
        return post_tax_cashflow

    def get_post_tax_economics_other_income(self):
        model = self.create_tax_model_other_income()
        post_tax_economics = pd.Series(model.calculate_post_tax_economics(epl_case=self.epl_case))
        return post_tax_economics

    def create_tax_model_without_other_income(self):
        pre_tax_cashflow = self.get_pre_tax_cashflow()
        model = TaxSystemWithoutOtherIncome(pre_tax_cashflow, self.tax_assumptions, self.market_assumptions)
        return model

    def get_post_tax_cashflow_no_other_income(self):
        model = self.create_tax_model_without_other_income()
        post_tax_cashflow = model.calculate_post_tax_cashflow(include_epl=self.epl_case)
        post_tax_cashflow = np.round(post_tax_cashflow)
        return post_tax_cashflow

    def get_post_tax_economics_no_other_income(self):
        model = self.create_tax_model_without_other_income()
        post_tax_economics = pd.Series(model.calculate_post_tax_economics(epl_case=self.epl_case))
        return post_tax_economics


def economic_results_generator(field_assumptions, market_assumptions, tax_assumptions):
    # Case where with permanent system and summer EPL
    results_no_epl = EconomicAnalysis(field_assumptions,
                                      market_assumptions,
                                      tax_assumptions, epl_case=False)

    results_epl_summer = EconomicAnalysis(field_assumptions,
                                          market_assumptions,
                                          tax_assumptions, epl_case=True)

    # Case of new EPL
    market_assumptions['epl_years'] = 6
    tax_assumptions['epl_rate'] = 0.35
    tax_assumptions['ia_for_epl_rate'] = 0.29
    results_epl_autumn = EconomicAnalysis(field_assumptions,
                                          market_assumptions,
                                          tax_assumptions, epl_case=True)
    market_assumptions['epl_years'] = 4
    tax_assumptions['epl_rate'] = 0.25
    tax_assumptions['ia_for_epl_rate'] = 0.80

    # Case of new EPL and field starting 2019
    market_assumptions['epl_years'] = 6
    market_assumptions['epl_beginning_year'] = 3
    tax_assumptions['epl_rate'] = 0.35
    tax_assumptions['ia_for_epl_rate'] = 0.29
    results_epl_autumn_2019 = EconomicAnalysis(field_assumptions,
                                               market_assumptions,
                                               tax_assumptions, epl_case=True)
    market_assumptions['epl_years'] = 4
    tax_assumptions['epl_rate'] = 0.25
    market_assumptions['epl_years'] = 4  # Return to base case of summer statement
    market_assumptions['epl_beginning_year'] = 0  # Return to base case of summer statement

    pre_tax_npv = results_no_epl.get_pre_tax_economics().real_pre_tax_npv
    post_tax_no_epl_other = results_no_epl.get_post_tax_economics_other_income().real_post_tax_npv
    post_tax_no_epl_no_other_income = results_no_epl.get_post_tax_economics_no_other_income().real_post_tax_npv

    post_tax_epl_summer_other = results_epl_summer.get_post_tax_economics_other_income().real_post_tax_npv
    post_tax_epl_summer_no_other = results_epl_summer.get_post_tax_economics_no_other_income().real_post_tax_npv

    post_tax_epl_autumn_other = results_epl_autumn.get_post_tax_economics_other_income().real_post_tax_npv
    post_tax_epl_autumn_no_other = results_epl_autumn.get_post_tax_economics_no_other_income().real_post_tax_npv

    post_tax_epl_autumn_other_delayed = results_epl_autumn_2019.get_post_tax_economics_other_income().real_post_tax_npv
    post_tax_epl_autumn_no_other_delayed = results_epl_autumn_2019.get_post_tax_economics_no_other_income().real_post_tax_npv

    economics = pd.Series({
        "Pre tax": pre_tax_npv,
        "Post tax no EPL other income": post_tax_no_epl_other,
        "Post tax no EPL no other income": post_tax_no_epl_no_other_income,
        "Post tax EPL summer other income": post_tax_epl_summer_other,
        "Post tax EPL summer no other income": post_tax_epl_summer_no_other,
        "Post tax EPL autumn other income": post_tax_epl_autumn_other,
        "Post tax EPL autumn no other income": post_tax_epl_autumn_no_other,
        "Post tax EPL autumn 2019 start other income": post_tax_epl_autumn_other_delayed,
        "Post tax EPL autumn 2019 start no other income": post_tax_epl_autumn_no_other_delayed
    }, name="NPV in Million £, Real values (rounded)")
    economics = np.round(economics)

    return economics
