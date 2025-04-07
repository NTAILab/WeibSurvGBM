import numpy as np

def safe_log(expression):
    """Avoid log(0) by clamping values."""
    return np.log(np.maximum(expression, 1e-20))