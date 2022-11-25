## README Section
# This file stores the assumptions for the deterministic economic model.

###############################################################################
## Field assumptions

field_10m_assum = {
    'reserves': 10000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.5, 0.5],
    'production_starts': 2,
    'decomm_schedule': [1],
    'production_profile': [3650000, 2482000, 1687760, 1147676, 780420, 530685, 360866, 245389,
                           166864, 113467, 77158,
                           52467, 35677, 24261, 16497, 11218, 7628, 5187, 3527, 2398],
    'devex': 19,
    'opex': 0.0875,
    'decommx': 0.1
}

field_25m_assum = {
    'reserves': 25000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.3, 0.3, 0.4],
    'production_starts': 2,
    'decomm_schedule': [1],
    'production_profile': [6387500, 4599000, 3311280, 2384122, 1716568, 1235929, 889869, 640705,
                           461308, 332142,
                           239142, 172182, 123971, 89259, 64267, 46272, 33316, 23987, 17271, 12435],
    'devex': 16,
    'opex': 0.0825,
    'decommx': 0.1
}

field_50m_assum = {
    'reserves': 50000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.3, 0.3, 0.4],
    'production_starts': 2,
    'decomm_schedule': [1],
    'production_profile': [7300000, 12775000, 9581250, 7185937, 5389453, 4042089, 3031567, 2273675,
                           1705256, 1278942,
                           959206, 719405, 539553, 404665, 303499, 227624, 170718, 128038, 96028,
                           72021],
    'devex': 13,
    'opex': 0.0775,
    'decommx': 0.1
}

field_100m_assum = {
    'reserves': 100000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.2, 0.3, 0.3, 0.2],
    'production_starts': 3,
    'decomm_schedule': [0.6, 0.4],
    'production_profile': [10950000, 16425000, 16425100, 13140000, 10512000, 8409600, 6727680,
                           5382144, 4305715,
                           3444572, 2755658, 2204526, 1763621, 1410897, 1128717, 902974, 722379,
                           577903, 462323,
                           369858],
    'devex': 10,
    'opex': 0.0725,
    'decommx': 0.1
}

field_150m_assum = {
    'reserves': 150000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.2, 0.2, 0.2, 0.3, 0.1],
    'production_starts': 4,
    'decomm_schedule': [0.60, 0.20, 0.10, .010],
    'production_profile': [16425000, 20075000, 20075000, 18250000, 15695000, 12556000, 10044800,
                           8035840, 6428672,
                           5142938, 4114350, 3291480, 2633184, 2106547, 1685238, 1348190, 1078552,
                           862842, 690273,
                           552219],
    'devex': 7,
    'opex': 0.0675,
    'decommx': 0.1
}

###############################################################################
## Market assumptions
# TODO need to update assumptions based on this OBR fiscal projections

market_assumptions = {
    'oil_price': 60,  # USD / bbl (Nominal Brent Oil Price)
    'gas_price': None,  # Pence / Therm (UK NBC Gas price)
    'gas_to_boe_factor': 1 / 5800,  # Retrieved from the OGA UK Reserves Report
    'discount_factor': 0.10,  # Discount rate for DCF 1 = 100%
    'exchange_rate': 1.2908,  # USD per pound
    'cpi': 0.02,  # Consumer price index 1 = 100%
    'spi': 0.02,  # Services producer index 1 = 100%
    'working_interest': 1,  # Working interest of the company in the field 1 = 100%
    'inflation': 0.02,  # Inflation rate to discount for Real NPV
    'trigger_threshold': 1.5,  # Trigger for decommissioning security payments
    'npvi_ratio': 0.30  # NPV/I ratio hurdle, to analyse capital rationing
}
########################################
# Set tax assumptions
#
# Notes on EPL beginning year:
# beginning year is like shifting the time of projects to left. So if the
# epl_beginning_year assumptions is 3, then this shifts 2022 three years to the left to
# 2019. If the beginning year is 0 then the first year is 2022 which is the base case.
#
# % Case 1: Other income available to offset losses

tax_assumptions = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.25,  # epl = energy profits levy
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2022.
    'epl_years': 4,
    # Number of years the EPL will apply as it is only short term assume 4 years from 2022-2025
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0.80,
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

# assum_tax_case1.trust_income_rate = .45;
#
# % Case 2: No other income available to offset losses
# assum_tax_case2.rfes_rate = .10;
# assum_tax_case2.ia_rate = .625;
# assum_tax_case2.ct_rate = .30;
# assum_tax_case2.sc_rate = .10;
# assum_tax_case2.decommx_relief_rate = .40;
# assum_tax_case2.trust_income_rate = .45;
#
#  Fincancia Security Assumptions
#
# % Letter of credit
# assum_loc.fee = .03; % charge over the total decomm costs 1 = 100%
# assum_loc.interest = .01; % charge over loc fee 1 = 100%
#
# % Surety Bond
# assum_surety.fee = .055; % charge over the total decomm costs 1 = 100%
# assum_surety.interest = .0; % charge over surety bond fee 1 = 100%
#
# % Trust Fund
# assum_trust.interest = .05; % Interest returns for the Trust Fund
