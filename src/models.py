"""
Statistical modeling for marine litter with confounders.

This module provides:
- GAM (Generalized Additive Models)
- Negative Binomial regression
- Zero-Inflated Negative Binomial (ZINB)
- Cross-validation and diagnostics
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger('marine_litter.models')


def prepare_model_data(
    litter_df: pd.DataFrame,
    effort_df: pd.DataFrame,
    hydro_features_df: pd.DataFrame,
    response_var: str = 'litter_count',
    include_temporal: bool = True,
    precip_features: Optional[List[str]] = None  # NEW PARAMETER
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling by merging and aligning all inputs.

    Parameters
    ----------
    litter_df : pd.DataFrame
        Litter data with MultiIndex (date, port_id).
    effort_df : pd.DataFrame
        Effort data with MultiIndex (date, port_id).
    hydro_features_df : pd.DataFrame
        Hydrology features with MultiIndex (date, port_id).
    response_var : str
        Response variable name.
    include_temporal : bool
        Whether to add temporal features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X is feature matrix and y is response.
    """
    # Merge all data
    data = litter_df[[response_var]].copy()
    data = data.join(effort_df[['effort_rate', 'trawling_hours']], how='left')
    # data = data.join(hydro_features_df, how='left')  # This is wrong!!!!!!!!
    # NEW: Filter precipitation features if specified
    if precip_features is not None:
        hydro_features_filtered = hydro_features_df[precip_features]
        data = data.join(hydro_features_filtered, how='left')
    else:
        data = data.join(hydro_features_df, how='left')

    # Add temporal features if requested
    if include_temporal:
        dates = data.index.get_level_values('date')
        data['year'] = dates.year
        data['month'] = dates.month
        data['day_of_year'] = dates.dayofyear
        data['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
        data['doy_sin'] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
        data['doy_cos'] = np.cos(2 * np.pi * dates.dayofyear / 365.25)

    # Add port dummy variables
    ports = data.index.get_level_values('port_id')
    port_dummies = pd.get_dummies(ports, prefix='port', drop_first=True)
    port_dummies.index = data.index
    data = pd.concat([data, port_dummies], axis=1)

    # Separate X and y
    y = data[response_var]
    X = data.drop(columns=[response_var])

    # Fill NaN in X
    X = X.fillna(0)

    # Ensure all columns are numeric (convert to float)
    X = X.astype(float)

    logger.info(f"Prepared model data: X shape {X.shape}, y shape {y.shape}")

    return X, y


def fit_negative_binomial(
    X: pd.DataFrame,
    y: pd.Series,
    use_offset: bool = True,
    offset_col: str = 'effort_rate'
) -> Tuple:
    """
    Fit Negative Binomial regression model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Response variable (counts).
    use_offset : bool
        Whether to use effort as offset.
    offset_col : str
        Column name for offset variable.

    Returns
    -------
    Tuple
        (model, results) where model is fitted statsmodels model.
    """
    import statsmodels.api as sm

    logger.info("Fitting Negative Binomial model")

    # Prepare data
    X_model = X.copy()

    # Handle offset
    if use_offset and offset_col in X_model.columns:
        offset = np.log(X_model[offset_col] + 0.1)
        X_model = X_model.drop(columns=[offset_col])
    else:
        offset = None

    # Remove highly correlated features
    X_model = X_model.loc[:, X_model.std() > 0]  # Remove constant columns

    # Add intercept
    X_model = sm.add_constant(X_model)

    # Fit model
    if offset is not None:
        model = sm.GLM(y, X_model, family=sm.families.NegativeBinomial(),
                       offset=offset)
    else:
        model = sm.GLM(y, X_model, family=sm.families.NegativeBinomial())

    try:
        results = model.fit()
        logger.info(f"Model converged. AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
    except Exception as e:
        logger.error(f"Model fitting failed: {e}")
        raise

    return model, results


def fit_zinb(
    X: pd.DataFrame,
    y: pd.Series,
    use_offset: bool = True,
    offset_col: str = 'effort_rate'
) -> Tuple:
    """
    Fit Zero-Inflated Negative Binomial model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Response variable (counts).
    use_offset : bool
        Whether to use effort as offset.
    offset_col : str
        Column name for offset.

    Returns
    -------
    Tuple
        (model, results)
    """
    import statsmodels.api as sm

    logger.info("Fitting Zero-Inflated Negative Binomial model")

    # Prepare data
    X_model = X.copy()

    # Handle offset
    if use_offset and offset_col in X_model.columns:
        offset = np.log(X_model[offset_col] + 0.1)
        X_model = X_model.drop(columns=[offset_col])
    else:
        offset = None

    # Remove constant columns
    X_model = X_model.loc[:, X_model.std() > 0]

    # Add intercept
    X_model = sm.add_constant(X_model)

    # Fit ZINB
    try:
        if offset is not None:
            model = sm.ZeroInflatedNegativeBinomialP(
                y, X_model,
                exog_infl=X_model,  # Same covariates for inflation
                offset=offset
            )
        else:
            model = sm.ZeroInflatedNegativeBinomialP(
                y, X_model,
                exog_infl=X_model
            )

        results = model.fit(method='bfgs', maxiter=1000)
        logger.info(f"ZINB converged. AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
    except Exception as e:
        logger.error(f"ZINB fitting failed: {e}")
        raise

    return model, results


def fit_gam(
    X: pd.DataFrame,
    y: pd.Series,
    smooth_features: Optional[List[str]] = None,
    use_offset: bool = True,
    offset_col: str = 'effort_rate',
    n_splines: int = 10,
    lam: Optional[float] = None
) -> Tuple:
    """
    Fit Generalized Additive Model using pygam.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Response variable.
    smooth_features : List[str], optional
        Features to apply smoothing splines. If None, applies to hydro features.
    use_offset : bool
        Whether to use effort as offset.
    offset_col : str
        Column for offset.
    n_splines : int
        Number of spline basis functions.
    lam : float, optional
        Smoothing parameter. If None, uses cross-validation.

    Returns
    -------
    Tuple
        (gam_model, feature_names)
    """
    try:
        from pygam import GAM, s, f, l
    except ImportError:
        logger.error("pygam not installed. Install with: conda install -c conda-forge pygam")
        raise

    logger.info("Fitting GAM with pygam")

    # Validate n_splines (must be > spline_order, which is typically 3)
    if n_splines <= 3:
        logger.warning(f"n_splines={n_splines} is too small (must be > 3). Setting to 5.")
        n_splines = 5

    # Prepare data
    X_model = X.copy()

    # Handle offset
    if use_offset and offset_col in X_model.columns:
        offset = np.log(X_model[offset_col] + 0.1).values
        X_model = X_model.drop(columns=[offset_col])
    else:
        offset = None

    # Identify smooth features
    if smooth_features is None:
        # Default: smooth hydro lag features and temporal features
        smooth_features = [
            col for col in X_model.columns
            if ('lag_' in col or 'doy_' in col or 'roll' in col)
        ]

    # Build formula
    feature_list = []
    for i, col in enumerate(X_model.columns):
        if col in smooth_features:
            feature_list.append(s(i, n_splines=n_splines))
        elif col.startswith('port_') or col in ['month', 'year']:
            feature_list.append(f(i))  # Categorical
        else:
            feature_list.append(s(i, n_splines=5))  # Light smoothing (must be > spline_order=3)

    # Combine terms
    formula = feature_list[0]
    for term in feature_list[1:]:
        formula += term

    # Fit GAM with appropriate distribution for count data
    # Try NegativeBinomial first (if available), fall back to Poisson
    X_array = X_model.values
    y_array = y.values

    # DIAGNOSTIC LOGGING
    logger.info("="*60)
    logger.info("GAM PRE-FLIGHT DIAGNOSTICS:")
    logger.info(f"  Data shape: {X_model.shape}")
    logger.info(f"  Target (y) range: {y.min()} to {y.max()}")
    logger.info(f"  Target zeros: {(y == 0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")

    # Check for problematic values
    if np.any(np.isnan(X_array)):
        logger.warning(f"  ⚠️ NaN values in X: {np.isnan(X_array).sum()}")
    if np.any(np.isinf(X_array)):
        logger.warning(f"  ⚠️ Inf values in X: {np.isinf(X_array).sum()}")
    if np.any(np.isnan(y_array)):
        logger.warning(f"  ⚠️ NaN values in y: {np.isnan(y_array).sum()}")

    # Check predictor scales
    X_ranges = X_model.max() - X_model.min()
    logger.info(f"  Predictor scale range: {X_ranges.min():.2e} to {X_ranges.max():.2e}")
    if X_ranges.max() / X_ranges.min() > 1000:
        logger.warning(f"  ⚠️ Large scale differences (ratio: {X_ranges.max()/X_ranges.min():.1e})")
        logger.warning(f"  Consider standardizing predictors!")

    # Check offset
    if offset is not None:
        logger.info(f"  Offset range: {offset.min():.2e} to {offset.max():.2e}")
        if np.any(offset <= 0):
            logger.warning(f"  ⚠️ Non-positive offset values: {(offset <= 0).sum()}")
        if offset.min() < 0.1:
            logger.warning(f"  ⚠️ Very small offset values (min: {offset.min():.2e})")

    # Check sample size
    if len(y) < 30:
        logger.warning(f"  ⚠️ Small sample size (n={len(y)}) - may cause convergence issues")

    logger.info("="*60)

    # Try to use appropriate distribution for count data
    try:
        # Try newer pyGAM API (0.9+)
        from pygam import PoissonGAM
        logger.info("Using PoissonGAM (suitable for count data with overdispersion via scale parameter)")
        gam = PoissonGAM(formula)
    except ImportError:
        try:
            # Try older pyGAM API with distributions module
            from pygam.distributions import Poisson
            logger.info("Using GAM with Poisson distribution")
            gam = GAM(formula, distribution='poisson')
        except:
            # Fall back to basic GAM (will use Gaussian by default)
            logger.warning("Using GAM with default distribution (Gaussian with log link)")
            logger.warning("For count data, Poisson or Negative Binomial is preferred")
            from pygam import LinearGAM
            gam = LinearGAM(formula)

    try:
        if lam is not None:
            gam.set_params(lam=lam)
            gam.fit(X_array, y_array, weights=offset if offset is not None else None)
        else:
            # Use cross-validation for lambda
            logger.info("Searching for optimal smoothing parameter via CV...")
            gam.gridsearch(X_array, y_array,
                          weights=offset if offset is not None else None,
                          progress=False)

        # Try to report model quality (different pyGAM versions have different attributes)
        try:
            if hasattr(gam, 'statistics_'):
                pseudo_r2 = gam.statistics_['pseudo_r2']['explained_deviance']
                logger.info(f"GAM fitted. Pseudo R²: {pseudo_r2:.3f}")
            elif hasattr(gam, 'pseudo_r2'):
                logger.info(f"GAM fitted. Pseudo R²: {gam.pseudo_r2():.3f}")
            else:
                # Just report it fitted successfully
                logger.info("GAM fitted successfully")
        except:
            logger.info("GAM fitted successfully (could not compute R²)")

    except Exception as e:
        logger.error(f"GAM fitting failed: {e}")
        raise

    return gam, list(X_model.columns)


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_func: callable,
    cv_method: str = 'kfold',
    n_folds: int = 5,
    test_fraction: float = 0.2
) -> Dict[str, float]:
    """
    Cross-validate a model.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Response.
    model_func : callable
        Function that fits model and returns predictions.
    cv_method : str
        'kfold' or 'blocked_time'.
    n_folds : int
        Number of folds for k-fold CV.
    test_fraction : float
        Test set fraction for blocked time CV.

    Returns
    -------
    Dict[str, float]
        Cross-validation metrics (MAE, RMSE, etc.).
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    logger.info(f"Cross-validating with method: {cv_method}")

    y_true_all = []
    y_pred_all = []

    if cv_method == 'kfold':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            y_pred = model_func(X_train, y_train, X_test)

            y_true_all.extend(y_test.values)
            y_pred_all.extend(y_pred)

    elif cv_method == 'blocked_time':
        # Split data temporally
        n_test = int(len(X) * test_fraction)
        X_train = X.iloc[:-n_test]
        X_test = X.iloc[-n_test:]
        y_train = y.iloc[:-n_test]
        y_test = y.iloc[-n_test:]

        y_pred = model_func(X_train, y_train, X_test)

        y_true_all = y_test.values
        y_pred_all = y_pred

    else:
        raise ValueError(f"Unknown CV method: {cv_method}")

    # Compute metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    metrics = {
        'mae': mean_absolute_error(y_true_all, y_pred_all),
        'rmse': np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
        'r2': r2_score(y_true_all, y_pred_all),
        'mape': np.mean(np.abs((y_true_all - y_pred_all) / (y_true_all + 1))) * 100
    }

    logger.info(f"CV Results: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")

    return metrics


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Compute residual diagnostics.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    X : pd.DataFrame, optional
        Feature matrix for additional diagnostics.

    Returns
    -------
    Dict
        Diagnostics including residuals, autocorrelation, etc.
    """
    residuals = y_true - y_pred

    diagnostics = {
        'residuals': residuals,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }

    # Test for autocorrelation (Durbin-Watson)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)
    diagnostics['durbin_watson'] = dw

    # Shapiro-Wilk test for normality
    if len(residuals) < 5000:  # Shapiro-Wilk limit
        _, p_norm = stats.shapiro(residuals)
        diagnostics['normality_p_value'] = p_norm

    logger.info(f"Residual diagnostics: mean={diagnostics['mean_residual']:.3f}, "
                f"DW={dw:.3f}")

    return diagnostics


def extract_feature_importance(
    model: any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from fitted model.

    Parameters
    ----------
    model : any
        Fitted model (statsmodels or pygam).
    feature_names : List[str]
        Feature names.
    top_n : int
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame.
    """
    try:
        # Try statsmodels results
        if hasattr(model, 'params'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.params.values,
                'p_value': model.pvalues.values,
                'abs_coef': np.abs(model.params.values)
            })
        # Try pygam
        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_,
                'abs_coef': np.abs(model.coef_)
            })
        else:
            logger.warning("Cannot extract feature importance from this model type")
            return pd.DataFrame()

        importance = importance.sort_values('abs_coef', ascending=False).head(top_n)

        return importance

    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")
        return pd.DataFrame()
