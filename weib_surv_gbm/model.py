import numpy as np
import xgboost as xgb
from .loss import surv_grad_hess, surv_loss

from scipy.integrate import trapezoid
from sksurv.metrics import concordance_index_censored

class WeibSurvGBM:

    """
    Gradient Boosting Model for Parametric Survival Analysis.

    This model uses gradient boosting (based on XGBoost) to predict parameters of a parametric
    survival distribution — specifically the Weibull distribution with `lambda` (scale) and `k` (shape)
    parameters. It is designed to work with right-censored survival data and supports custom
    initialization and regularization.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Learning rate for boosting.

    n_estimators : int, default=100
        Number of boosting rounds (trees).

    max_depth : int, default=7
        Maximum tree depth for base learners.

    random_seed : int, default=42
        Random seed for reproducibility.

    lambda_val : float, default=1
        L2 regularization term on weights.

    alpha : float, default=0
        L1 regularization term on weights.

    colsample_bytree : float, default=1
        Subsample ratio of columns when constructing each tree.

    colsample_bylevel : float, default=1
        Subsample ratio of columns for each level.

    colsample_bynode : float, default=1
        Subsample ratio of columns for each split.

    max_leaves : int, default=0
        Maximum number of leaves for tree growth (used when grow_policy="lossguide").

    max_bin : int, default=256
        Number of histogram bins for histogram-based split finding.

    min_child_weight : float, default=1
        Minimum sum of instance weight (hessian) needed in a child.

    subsample : float, default=1
        Subsample ratio of the training instances.

    initial_lambda_by_mean : bool, default=False
        If True, initializes lambda using the mean event time.

    initial_k : float or None, default=None
        If provided, uses this fixed initial value for the shape parameter `k`.

    Attributes
    ----------
    _model : xgb.Booster
        The trained XGBoost model.

    _results : dict
        Training history and logged metrics.

    _times : np.ndarray
        Survival/censoring times, optionally used for later analysis.

    _initial_preds : np.ndarray
        Initial predictions for (lambda, k), if provided.
    """

    def __init__(
            self,
            learning_rate=0.1,
            n_estimators=100,
            max_depth=7,
            random_seed=42,
            lambda_val=1,
            alpha=0,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            max_leaves=0,
            max_bin=256,
            min_child_weight=1,
            subsample=1,
            initial_lambda_by_mean=False,
            initial_k = None,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.initial_lambda_by_mean = initial_lambda_by_mean
        self.initial_k = initial_k

    def _prepare_target(self, y):
        
        """Function for preparing data before training model
            y should have format (delta, time)"""

        delta_str = y.dtype.names[0]
        time_str = y.dtype.names[1]

        delta = y[delta_str]
        time = y[time_str]

        target = np.zeros((2, len(time)))
        target[0] = time

        target[1] = delta
        target = target.transpose()

        return target

    def fit(self, X, y):

        self._model = xgb.Booster()
        self._results = dict()
        self._times = np.array([])

        target = self._prepare_target(y)

        if self.initial_lambda_by_mean:
            self._initial_preds = np.array([np.mean(target[:, 0]), 0.5])
            initial_preds_train = np.tile(self._initial_preds, (len(X), 1))
            d_train = xgb.DMatrix(X, label=target, enable_categorical=True, base_margin=initial_preds_train)
        elif self.initial_k:
            self._initial_preds = np.array([0.5, self.initial_k])
            initial_preds_train = np.tile(self._initial_preds, (len(X), 1))
            d_train = xgb.DMatrix(X, label=target, enable_categorical=True, base_margin=initial_preds_train)
        else:
            d_train = xgb.DMatrix(X, label=target, enable_categorical=True)
        
        time = target[:, 0].astype(np.float64)
        self._times = np.sort(np.unique(time))

        self._model = xgb.train(
            {
                'tree_method': 'hist', 
                'seed': self.random_seed,
                'disable_default_eval_metric': 1,
                'multi_strategy': "multi_output_tree",
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'lambda': self.lambda_val,
                'alpha': self.alpha,
                'nthread': 8,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'colsample_bynode': self.colsample_bynode,
                'min_child_weight': self.min_child_weight,
                'max_leaves': self.max_leaves,
                'max_bin': self.max_bin,
                'subsample': self.subsample,
            },
            dtrain=d_train,
            num_boost_round=self.n_estimators,
            obj=surv_grad_hess,
            custom_metric=surv_loss,
            evals_result=self._results,
            evals=[(d_train, 'd_train')],
        )

        return self
    
    def predict(self, X):

        if self.initial_lambda_by_mean:
            initial_preds_test = np.tile(self._initial_preds, (len(X), 1))
            d_test = xgb.DMatrix(X, enable_categorical=True, base_margin=initial_preds_test)
        elif self.initial_k:
            initial_preds_test = np.tile(self._initial_preds, (len(X), 1))
            d_test = xgb.DMatrix(X, enable_categorical=True, base_margin=initial_preds_test)
        else:
            d_test = xgb.DMatrix(X, enable_categorical=True)

        predicted_params = self._model.predict(d_test)
        predicted_params[:, 0] = np.maximum(predicted_params[:, 0], 10e-4)
        predicted_params[:, 1] = np.clip(predicted_params[:, 1], 10e-4, 10)

        lambdas = predicted_params[:, 0][:, np.newaxis]
        ks = predicted_params[:, 1][:, np.newaxis]

        t_i = np.array(self._times)
        t_next = np.append(t_i[1:], np.inf)

        exp_t_i = np.exp(-(t_i / lambdas) ** ks)
        exp_t_next = np.exp(-(t_next / lambdas) ** ks)

        predicted_proba = exp_t_i - exp_t_next
        predicted_proba[:, -1] = 1 - np.sum(predicted_proba[:, :-1], axis=1)

        return predicted_proba

    def _step_function(self, times, survival_function):

        if isinstance(times, (int, float)):
            times = [times]

        survs = []

        for time in times:
            if time < 0:
                raise ValueError("Time can't have negative value")

            if time < self._times[0]:
                survs.append(1)
            elif time >= self._times[-1]:
                survs.append(survival_function[-1])
            else:
                for i, bound in enumerate(self._times):
                    if time < bound:
                        survs.append(survival_function[i - 1])
                        break

        return survs
    
    def predict_survival_function(self, X):

        predicted_proba = self.predict(X)

        cumulative_proba = np.cumsum(predicted_proba, axis=1)

        cumulative_proba[cumulative_proba > 1.0] = 1.0

        survival_functions = 1 - cumulative_proba

        step_functions = np.array([
            lambda x, sf=sf: self._step_function(x, sf) for sf in survival_functions
        ])
        
        return step_functions
    
    def score(self, X, y):

        delta_str = y.dtype.names[0]
        time_str = y.dtype.names[1]

        delta = y[delta_str]
        time = y[time_str]
        
        predicted_proba = self.predict(X)

        cumulative_proba = np.cumsum(predicted_proba, axis=1)

        cumulative_proba[cumulative_proba > 1.0] = 1.0

        survival_function = 1 - cumulative_proba

        integrated_сum_proba = np.array([trapezoid(survival_function[i], self._times) for i in range(survival_function.shape[0])])
        
        c_index = concordance_index_censored(delta.astype(bool), time, -integrated_сum_proba)

        return c_index[0]
    
    def get_params(self, deep=True):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
    
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)