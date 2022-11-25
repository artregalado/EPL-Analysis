# %%
import numpy as np

# %% Test procedures for functions

ct_rate = .30
sc_rate = .10

decommx = np.array([0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 85, 20])

taxable_income = np.array([0, 0, 0, 405.802734,
                           567.73566792, 420.41323522, 307.43353387, 220.72034159,
                           154.09549356, 102.83233761, 63.31500901, 32.7772367,
                           9.10262793, 0, 0, 0])

flipped_taxable = np.flip(taxable_income)

# Carryback limit is the number of years that decommissioning expenditures are occuring + 3 years (the COP years plus 2 more)
carryback_limit = (len(decommx[decommx > 0]) + 3) - 1  # -1 used to adjust for python 0 indexing

# For decommissioning we can assume relief to be at 40% because of the DRD that actually guarantees the rate to be such.
# The key point is how to assign it to the taxable income flow for CT and SC. Can only be claimed back max 3 years.
# Source for carry back limit is years https://www.gov.uk/hmrc-internal-manuals/oil-taxation-manual/ot21060

decomm_relief_bag = decommx.sum() * (ct_rate + sc_rate)

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

print(taxable_adjusted_for_decomm_relief)
