import numpy as np

def calculate_SoC(SoC_prev, charging, consumption):
    return np.max(SoC_prev + charging - consumption, 0)

