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
        pre_tax_economics['real_pre_tax_npv'] = np.round(
            pre_tax_economics['real_pre_tax_npv'] / self.market_assumptions['exchange_rate'])
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


def calculate_economic_results(field_assumptions, market_assumptions,
                               tax_assumptions, tax_assumptions_delayed_summer_epl,
                               tax_assumptions_autumn_epl, tax_assumptions_delayed_autumn_epl):

    no_epl = EconomicAnalysis(field_assumptions, market_assumptions, tax_assumptions, epl_case=False)

    epl_summer = EconomicAnalysis(field_assumptions, market_assumptions, tax_assumptions, epl_case=True)

    # Case of new EPL
    epl_autumn = EconomicAnalysis(field_assumptions, market_assumptions,
                                  tax_assumptions_autumn_epl, epl_case=True)

    # Case of summer EPL and field starting 2019
    epl_summer_2019 = EconomicAnalysis(field_assumptions, market_assumptions,
                                       tax_assumptions_delayed_summer_epl, epl_case=True)

    # Case of new EPL and field starting 2019
    epl_autumn_2019 = EconomicAnalysis(field_assumptions, market_assumptions,
                                       tax_assumptions_delayed_autumn_epl, epl_case=True)

    pre_tax_npv = no_epl.get_pre_tax_economics().real_pre_tax_npv
    post_tax_no_epl_other = no_epl.get_post_tax_economics_other_income()
    post_tax_no_epl_no_other_income = no_epl.get_post_tax_economics_no_other_income()

    post_tax_epl_summer_other = epl_summer.get_post_tax_economics_other_income()
    post_tax_epl_summer_no_other = epl_summer.get_post_tax_economics_no_other_income()

    post_tax_epl_autumn_other = epl_autumn.get_post_tax_economics_other_income()
    post_tax_epl_autumn_no_other = epl_autumn.get_post_tax_economics_no_other_income()

    post_tax_epl_summer_other_delayed = epl_summer_2019.get_post_tax_economics_other_income()
    post_tax_epl_summer_no_other_delayed = epl_summer_2019.get_post_tax_economics_no_other_income()

    post_tax_epl_autumn_other_delayed = epl_autumn_2019.get_post_tax_economics_other_income()
    post_tax_epl_autumn_no_other_delayed = epl_autumn_2019.get_post_tax_economics_no_other_income()

    npv_results = pd.Series({
        "Pre tax": pre_tax_npv,
        "Post tax no EPL other income": post_tax_no_epl_other['real_post_tax_npv'],
        "Post tax EPL summer other income": post_tax_epl_summer_other['real_post_tax_npv'],
        "Post tax EPL autumn other income": post_tax_epl_autumn_other['real_post_tax_npv'],
        "Post tax EPL summer 2019 start other income": post_tax_epl_summer_other_delayed['real_post_tax_npv'],
        "Post tax EPL autumn 2019 start other income": post_tax_epl_autumn_other_delayed['real_post_tax_npv'],
        "Post tax no EPL no other income": post_tax_no_epl_no_other_income['real_post_tax_npv'],
        "Post tax EPL summer no other income": post_tax_epl_summer_no_other['real_post_tax_npv'],
        "Post tax EPL autumn no other income": post_tax_epl_autumn_no_other['real_post_tax_npv'],
        "Post tax EPL summer 2019 start no other income": post_tax_epl_summer_no_other_delayed['real_post_tax_npv'],
        "Post tax EPL autumn 2019 start no other income": post_tax_epl_autumn_no_other_delayed['real_post_tax_npv']
    }, name="NPV in Million £, Real values (rounded)")
    npv_results = np.round(npv_results)

    tax_paid_results = pd.Series({
        "Post tax no EPL other income": post_tax_no_epl_other['tax_paid_real_pv'],
        "Post tax EPL summer other income": post_tax_epl_summer_other['tax_paid_real_pv'],
        "Post tax EPL autumn other income": post_tax_epl_autumn_other['tax_paid_real_pv'],
        "Post tax EPL summer 2019 start other income": post_tax_epl_summer_other_delayed['tax_paid_real_pv'],
        "Post tax EPL autumn 2019 start other income": post_tax_epl_autumn_other_delayed['tax_paid_real_pv'],
        "Post tax no EPL no other income": post_tax_no_epl_no_other_income['tax_paid_real_pv'],
        "Post tax EPL summer no other income": post_tax_epl_summer_no_other['tax_paid_real_pv'],
        "Post tax EPL autumn no other income": post_tax_epl_autumn_no_other['tax_paid_real_pv'],
        "Post tax EPL summer 2019 start no other income": post_tax_epl_summer_no_other_delayed['tax_paid_real_pv'],
        "Post tax EPL autumn 2019 start no other income": post_tax_epl_autumn_no_other_delayed['tax_paid_real_pv']
    }, name="Tax paid in Million £, Real values (rounded)")
    tax_paid_results = np.round(tax_paid_results)

    npvi_ratio = pd.Series({
        "Post tax no EPL other income": post_tax_no_epl_other['npv/i_ratio'],
        "Post tax EPL summer other income": post_tax_epl_summer_other['npv/i_ratio'],
        "Post tax EPL autumn other income": post_tax_epl_autumn_other['npv/i_ratio'],
        "Post tax EPL summer 2019 start other income": post_tax_epl_summer_other_delayed['npv/i_ratio'],
        "Post tax EPL autumn 2019 start other income": post_tax_epl_autumn_other_delayed['npv/i_ratio'],
        "Post tax no EPL no other income": post_tax_no_epl_no_other_income['npv/i_ratio'],
        "Post tax EPL summer no other income": post_tax_epl_summer_no_other['npv/i_ratio'],
        "Post tax EPL autumn no other income": post_tax_epl_autumn_no_other['npv/i_ratio'],
        "Post tax EPL summer 2019 start no other income": post_tax_epl_summer_no_other_delayed['npv/i_ratio'],
        "Post tax EPL autumn 2019 start no other income": post_tax_epl_autumn_no_other_delayed['npv/i_ratio']
    }, name="Post tax NPV/ Pre tax I Ratios")
    npvi_ratio = np.round(npvi_ratio, 2)

    results = {
        "npv_results": npv_results,
        "tax_paid_results": tax_paid_results,
        "npvi_ratios": npvi_ratio
    }
    return results


def calculate_cashflow_results(field_assumptions, market_assumptions,
                               tax_assumptions, tax_assumptions_delayed_summer_epl,
                               tax_assumptions_autumn_epl, tax_assumptions_delayed_autumn_epl):

    no_epl = EconomicAnalysis(field_assumptions, market_assumptions, tax_assumptions, epl_case=False)
    epl_summer = EconomicAnalysis(field_assumptions, market_assumptions, tax_assumptions, epl_case=True)

    # Case of new EPL
    epl_autumn = EconomicAnalysis(field_assumptions, market_assumptions,
                                  tax_assumptions_autumn_epl, epl_case=True)

    # Case of summer EPL and field starting 2019
    epl_summer_2019 = EconomicAnalysis(field_assumptions, market_assumptions,
                                       tax_assumptions_delayed_summer_epl, epl_case=True)

    # Case of new EPL and field starting 2019
    epl_autumn_2019 = EconomicAnalysis(field_assumptions, market_assumptions,
                                       tax_assumptions_delayed_autumn_epl, epl_case=True)

    pre_tax_npv = no_epl.get_pre_tax_cashflow()
    post_tax_no_epl_other = no_epl.get_post_tax_cashflow_other_income()
    post_tax_no_epl_no_other_income = no_epl.get_post_tax_cashflow_no_other_income()

    post_tax_epl_summer_other = epl_summer.get_post_tax_cashflow_other_income()
    post_tax_epl_summer_no_other = epl_summer.get_post_tax_cashflow_no_other_income()

    post_tax_epl_autumn_other = epl_autumn.get_post_tax_cashflow_other_income()
    post_tax_epl_autumn_no_other = epl_autumn.get_post_tax_cashflow_no_other_income()

    post_tax_epl_summer_other_delayed = epl_summer_2019.get_post_tax_cashflow_other_income()
    post_tax_epl_summer_no_other_delayed = epl_summer_2019.get_post_tax_cashflow_no_other_income()

    post_tax_epl_autumn_other_delayed = epl_autumn_2019.get_post_tax_cashflow_other_income()
    post_tax_epl_autumn_no_other_delayed = epl_autumn_2019.get_post_tax_cashflow_no_other_income()

    cashflows = pd.Series({
        "Pre tax": pre_tax_npv,
        "Post tax no EPL other income": post_tax_no_epl_other,
        "Post tax EPL summer other income": post_tax_epl_summer_other,
        "Post tax EPL autumn other income": post_tax_epl_autumn_other,
        "Post tax EPL summer 2019 start other income": post_tax_epl_summer_other_delayed,
        "Post tax EPL autumn 2019 start other income": post_tax_epl_autumn_other_delayed,
        "Post tax no EPL no other income": post_tax_no_epl_no_other_income,
        "Post tax EPL summer no other income": post_tax_epl_summer_no_other,
        "Post tax EPL autumn no other income": post_tax_epl_autumn_no_other,
        "Post tax EPL summer 2019 start no other income": post_tax_epl_summer_no_other_delayed,
        "Post tax EPL autumn 2019 start no other income": post_tax_epl_autumn_no_other_delayed
    }, name="Cashflow in Million £")

    return cashflows
