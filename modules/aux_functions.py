import numpy as np


def compound_value(initial_value, interest_rate, n_years):
    compounded_stream = [initial_value * ((1 + interest_rate) ** t) for t in range(n_years)]
    return np.array(compounded_stream)


def discount_stream(stream, discount_rate):
    stream = np.array(stream)
    discounted_stream = []
    for t, value in enumerate(stream):
        num = float(value)
        denom = (1 + float(discount_rate)) ** float(t)
        discounted_value = num / denom
        discounted_stream.append(discounted_value)
    return np.array(discounted_stream)
