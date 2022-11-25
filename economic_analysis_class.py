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
        pre_tax_economics['real_pre_tax_npv'] = np.round(pre_tax_economics['real_pre_tax_npv'])
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
