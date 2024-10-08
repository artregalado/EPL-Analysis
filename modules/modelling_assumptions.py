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
    'production_profile': [3650000*.85, 2482000*.85, 1687760*.85, 1147676*.85, 780420*.85, 530685*.85, 360866*.85, 245389*.85,
                           166864*.85, 113467*.85, 77158*.85,
                           52467*.85, 35677*.85, 24261*.85, 16497*.85, 11218*.85, 7628*.85, 5187*.85, 3527*.85, 2398*.85],
    'devex': 19 / 1.3217, # to convert to GBP
    'opex': 0.0875 * 1.05, # add 5% carbon pricing cost
    'decommx': 0.1
}

field_25m_assum = {
    'reserves': 25000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.3, 0.3, 0.4],
    'production_starts': 2,
    'decomm_schedule': [1],
    'production_profile': [6387500*.85, 4599000*.85, 3311280*.85, 2384122*.85, 1716568*.85, 1235929*.85, 889869*.85, 640705*.85,
                           461308*.85, 332142*.85,
                           239142*.85, 172182*.85, 123971*.85, 89259*.85, 64267*.85, 46272*.85, 33316*.85, 23987*.85, 17271*.85, 12435*.85],
    'devex': 16 / 1.3217, # to convert to GBP
    'opex': 0.0825 * 1.05, # add 5% carbon pricing cost
    'decommx': 0.1
}

field_50m_assum = {
    'reserves': 50000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.3, 0.3, 0.4],
    'production_starts': 2,
    'decomm_schedule': [1],
    'production_profile': [7300000*.85, 12775000*.85, 9581250*.85, 7185937*.85, 5389453*.85, 4042089*.85, 3031567*.85, 2273675*.85,
                           1705256*.85, 1278942*.85,
                           959206*.85, 719405*.85, 539553*.85, 404665*.85, 303499*.85, 227624*.85, 170718*.85, 128038*.85, 96028*.85,
                           72021*.85],
    'devex': 13 / 1.3217, # to convert to GBP
    'opex': 0.0775 * 1.05, # add 5% carbon pricing cost
    'decommx': 0.1
}

field_100m_assum = {
    'reserves': 100000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.2, 0.3, 0.3, 0.2],
    'production_starts': 3,
    'decomm_schedule': [0.6, 0.4],
    'production_profile': [10950000*.85, 16425000*.85, 16425100*.85, 13140000*.85, 10512000*.85, 8409600*.85, 6727680*.85,
                           5382144*.85, 4305715*.85,
                           3444572*.85, 2755658*.85, 2204526*.85, 1763621*.85, 1410897*.85, 1128717*.85, 902974*.85, 722379*.85,
                           577903*.85, 462323*.85,
                           369858*.85],
    'devex': 10 / 1.3217, # to convert to GBP
    'opex': 0.0725 * 1.05, # add 5% carbon pricing cost
    'decommx': 0.1
}

field_150m_assum = {
    'reserves': 150000000,
    'oil_ratio': 1,
    'gas_ratio': 0,
    'development_schedule': [0.2, 0.2, 0.2, 0.3, 0.1],
    'production_starts': 4,
    'decomm_schedule': [0.60, 0.20, 0.10, .010],
    'production_profile': [16425000*.85, 20075000*.85, 20075000*.85, 18250000*.85, 15695000*.85, 12556000*.85, 10044800*.85,
                           8035840*.85, 6428672*.85,
                           5142938*.85, 4114350*.85, 3291480*.85, 2633184*.85, 2106547*.85, 1685238*.85, 1348190*.85, 1078552*.85,
                           862842*.85, 690273*.85,
                           552219*.85],
    'devex': 7 / 1.3217, # to convert to GBP
    'opex': 0.0675 * 1.05, # add 5% carbon pricing cost
    'decommx': 0.1
}

###############################################################################
## Market assumptions

market_assumptions = {

    # 70 USD / bbl (Nominal Brent Oil Price), Real 2025 average for 2025-2030 from
    # OBR March 2024 Outlook. See excel file for calcs. Price converted to get value in £
    'oil_price': 60 / 1.3217,  # Conversion to get value in £
    'gas_price': None,  # Pence / Therm (UK NBC Gas price)
    'gas_to_boe_factor': 1 / 5800,  # Retrieved from the OGA UK Reserves Report
    'discount_factor': 0.10,  # Discount rate for DCF 1 = 100%
    'exchange_rate': 1.3217	,  # USD per pound average for September 2024 from Bank of England, retrieve 5 Oct 2024
    'cpi': 0.02,  # Consumer price index 1 = 100% = set to the target of the Bank of England
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

# Tax cases assuming fiscal year to avoid problems with the Levy ending in March and not having monthly modelling

# Base assumptions with EPL as applicable before the July 29, 2024 announced changes by Labour
tax_assumptions = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.35,
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2025.
    'epl_years': 4,
    # Number of years the EPL will apply, if 4 then 2025-2029
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0.29,
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

tax_assumptions_autumn_epl = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.35,  # epl = energy profits levy
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2025.
    'epl_years': 4,
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0.29,
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

# New tax case assumptions for Labour's proposals.

tax_assumptions_labour_epl_a = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.38,
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2025.
    'epl_years': 5, # Propose for levy to end in March 2030
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0, # No investment allowance for EPL
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

tax_assumptions_labour_epl_b = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.38,
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2024.
    'epl_years': 5, # Propose for levy to end in March 2030
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0, # No investment allowance for EPL
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
    # The removal of the capital allowance is done in the modelling with the labour_epl3b_tax_system class
}


# These remaining tax assumptions are legacy and not used in the revised paper of October 2024, are left here for reference
tax_assumptions_delayed_summer_epl = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.25,  # epl = energy profits levy
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2022.
    'epl_years': 5,
    # Number of years the EPL will apply as it is only short term assume 4 years from 2022-2025
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0.80,
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}


tax_assumptions_delayed_autumn_epl = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.35,  # epl = energy profits levy
    'epl_beginning_year': 0,  # 2019 start year
    'epl_years': 5,
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0.29,
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}


tax_assumptions_delayed_labour_epl_a = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.38,  # epl = energy profits levy
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2022.
    'epl_years': 6, # Propose for levy to end in 2029
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0, # No investment allowance for EPL
    # Note that this 80% is allowance on energy levy payable, so you reduce your epl paid
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

tax_assumptions_delayed_labour_epl_b = {
    'ct_rate': 0.30,
    'sc_rate': 0.10,
    'epl_rate': 0.38,  # epl = energy profits levy
    'epl_beginning_year': 0,  # base case cero if I assume fields begin in 2022.
    'epl_years': 6, # Propose for levy to end in 2029
    'ia_for_sc_rate': 0.625,
    'ia_for_epl_rate': 0, # No investment allowance for EPL
    'decommx_relief_rate': .40,
    'rfes_rate': 0.10
}

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
