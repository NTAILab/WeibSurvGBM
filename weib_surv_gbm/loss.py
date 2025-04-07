import numpy as np
import xgboost as xgb
from typing import Tuple
from .utils import safe_log

def prepare_data_for_computing(preds: np.ndarray, dtrain: xgb.DMatrix):

    """
    Prepares intermediate variables required for computing the gradient, Hessian, and loss 
    in a parametric survival model based on the Weibull distribution.
    """

    y = dtrain.get_label()
    y = y.reshape(preds.shape)

    time = y[:, 0].astype(np.float64)
    censoring = y[:, 1].astype(np.float64)
    
    N, C = y.shape

    preds = preds.astype(np.float64)
    preds = np.maximum(preds, 10e-20) 

    time_intervals = np.sort(np.unique(time))
    time_indices = np.searchsorted(time_intervals, time, side='right') - 1
    mask = (time_indices == np.max(time_indices))

    lamb = preds[:, 0]
    lamb = np.maximum(lamb, 10e-4)
    
    k = preds[:, 1]
    k = np.clip(k, 10e-4, 10)

    time_left = time_intervals[time_indices[~mask]]
    time_right = time_intervals[time_indices[~mask] + 1]

    time_ratio_left = time_left / lamb[~mask]
    time_ratio_right = time_right / lamb[~mask]

    time_ratio_left_pow_k = time_ratio_left ** k[~mask]
    time_ratio_right_pow_k = time_ratio_right ** k[~mask]

    time_ratio_left_masked = time_intervals[time_indices[mask]] / lamb[mask]
    time_ratio_left_pow_k_masked = time_ratio_left_masked ** k[mask]

    exp_diff = np.exp(-time_ratio_left_pow_k) - np.exp(-time_ratio_right_pow_k)
    exp_diff_replacement = np.maximum(exp_diff, 10e-20)

    return y.shape, censoring, mask, lamb, k, time_ratio_left_pow_k, time_ratio_right_pow_k, exp_diff_replacement, time_ratio_left, time_ratio_right, time_ratio_left_masked, time_ratio_left_pow_k_masked


def surv_grad_hess(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Computes the first- and second-order derivatives (gradient and Hessian) of the 
    survival loss with respect to the predicted parameters of the Weibull distribution.
    """

    shape, censoring, mask, lamb, k, time_ratio_left_pow_k, time_ratio_right_pow_k, exp_diff_replacement, time_ratio_left, time_ratio_right, time_ratio_left_masked, time_ratio_left_pow_k_masked = prepare_data_for_computing(preds, dtrain)
    
    # Gradient

    grad = np.zeros(shape) 

    # Calculate gradients for all but the last interval
    grad[~mask, 0] -= censoring[~mask] * (k[~mask] / lamb[~mask])  * (np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k - 
                                np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k) / exp_diff_replacement

    grad[~mask, 0] -= (1 - censoring[~mask]) * (k[~mask] / lamb[~mask]) * time_ratio_right_pow_k

    grad[~mask, 1] -= censoring[~mask] * (np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k * safe_log(time_ratio_right) 
                                          - np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k * safe_log(time_ratio_left)) / exp_diff_replacement

    grad[~mask, 1] += (1 - censoring[~mask]) * time_ratio_right_pow_k * safe_log(time_ratio_right)

    # Handle the last interval separately
    grad[mask, 0] -= censoring[mask] * (k[mask] / lamb[mask]) * time_ratio_left_pow_k_masked
    grad[mask, 1] += censoring[mask] * time_ratio_left_pow_k_masked * safe_log(time_ratio_left_masked)    
    
    # Hessian

    hess = np.zeros(shape) 
    
    # Calculate hessians for all but the last interval
    u_lambda = k[~mask] * (np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k - np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k)

    v_lambda = lamb[~mask] * exp_diff_replacement

    u_derivative_lambda = (k[~mask] ** 2 / lamb[~mask]) * (np.exp(-time_ratio_left_pow_k) * (time_ratio_left_pow_k ** 2)
                                                            - np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k
                                                            - np.exp(-time_ratio_right_pow_k) * (time_ratio_right_pow_k ** 2) 
                                                            + np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k)
    
    v_derivative_lambda = (np.exp(-time_ratio_left_pow_k) * (1 + k[~mask] * time_ratio_left_pow_k) 
                           - np.exp(-time_ratio_right_pow_k) * (1 + k[~mask] * time_ratio_right_pow_k))

    hess[~mask, 0] -= censoring[~mask] * ((u_derivative_lambda * v_lambda - u_lambda * v_derivative_lambda) / (v_lambda ** 2))
    hess[~mask, 0] += (1 - censoring[~mask]) * k[~mask] * (k[~mask] + 1) * time_ratio_right_pow_k / (lamb[~mask] ** 2) 


    u_k = (np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k * safe_log(time_ratio_right) 
           - np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k * safe_log(time_ratio_left))
    
    v_k = exp_diff_replacement

    u_derivative_k = (np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k * (safe_log(time_ratio_right) ** 2) 
                      - np.exp(-time_ratio_right_pow_k) * (time_ratio_right_pow_k ** 2) * (safe_log(time_ratio_right) ** 2)
                      - np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k * (safe_log(time_ratio_left) ** 2)
                      + np.exp(-time_ratio_left_pow_k) * (time_ratio_left_pow_k ** 2) * (safe_log(time_ratio_left) ** 2))

    v_derivative_k = (- np.exp(-time_ratio_left_pow_k) * time_ratio_left_pow_k * safe_log(time_ratio_left)
                      + np.exp(-time_ratio_right_pow_k) * time_ratio_right_pow_k * safe_log(time_ratio_right))

    hess[~mask, 1] -= censoring[~mask] * ((u_derivative_k * v_k - u_k * v_derivative_k) / (v_k ** 2))
    hess[~mask, 1] += (1 - censoring[~mask]) * time_ratio_right_pow_k * (safe_log(time_ratio_right) ** 2)

    # Handle the last interval separately
    hess[mask, 0] += censoring[mask] * k[mask] * (k[mask] + 1) * time_ratio_left_pow_k_masked / (lamb[mask] ** 2)
    hess[mask, 1] += censoring[mask] * time_ratio_left_pow_k_masked * (safe_log(time_ratio_left_masked) ** 2)

    return grad.ravel(), hess.ravel()

def surv_loss(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:

    """
    Computes the average negative log-likelihood loss for a parametric Weibull 
    survival model on a right-censored dataset.
    """

    shape, censoring, mask, _, _, _, time_ratio_right_pow_k, exp_diff_replacement, _, _, _, time_ratio_left_pow_k_masked = prepare_data_for_computing(preds, dtrain)

    N, _ = shape

    loss = 0

    loss -= np.sum(censoring[~mask] * safe_log(exp_diff_replacement))
    loss -= np.sum((1 - censoring[~mask]) * safe_log(time_ratio_right_pow_k))
    loss -= np.sum(censoring[mask] * safe_log(np.exp(-time_ratio_left_pow_k_masked)))

    loss = loss / N

    return 'SurvivalLoss', loss