#!/usr/bin/env python3
"""
VECM-based Granger Causality Implementation
Extends the CointegrationAnalyzer with VECM-specific Wald tests
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.stats.stattools import durbin_watson
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class VECMGrangerCausalityMixin:
    """
    Mixin class to add VECM-based Granger causality testing capabilities
    """
    
    def vecm_granger_causality_test(self, df: pd.DataFrame, 
                                   coint_rank: int = 1, 
                                   k_ar_diff: int = 1,
                                   det_order: int = 0,
                                   alpha: float = 0.05) -> Dict[str, Dict]:
        """
        Perform VECM-based Wald test for Granger causality
        
        This tests the null hypothesis that the lagged differences of one variable
        do not Granger-cause another variable in the VECM framework.
        
        Args:
            df: DataFrame with 2 cointegrated time series
            coint_rank: Number of cointegrating relationships (usually 1 for pairs)
            k_ar_diff: Number of lags in the VECM
            det_order: Deterministic trend order (-1, 0, 1)
            alpha: Significance level for the test
            
        Returns:
            Dictionary with Granger causality test results for both directions
        """
        try:
            if df.shape[1] != 2:
                raise ValueError("VECM Granger causality test requires exactly 2 variables")
            
            # Ensure we have enough observations
            min_obs = max(50, (k_ar_diff + 1) * 10)
            if len(df) < min_obs:
                self.log.warning(f"Insufficient observations for VECM: {len(df)} < {min_obs}")
                return self._empty_vecm_result()
            
            # Fit the VECM model
            vecm_model = VECM(df, k_ar_diff=k_ar_diff, coint_rank=coint_rank, 
                             deterministic=self._get_deterministic_string(det_order))
            
            vecm_result = vecm_model.fit()
            
            # Get variable names
            var_names = df.columns.tolist()
            
            # Perform Wald tests for both directions
            results = {}
            
            # Test 1: Does var1 Granger-cause var2?
            results[f"{var_names[0]}_causes_{var_names[1]}"] = self._perform_vecm_wald_test(
                vecm_result, cause_var=0, effect_var=1, k_ar_diff=k_ar_diff, alpha=alpha
            )
            
            # Test 2: Does var2 Granger-cause var1?
            results[f"{var_names[1]}_causes_{var_names[0]}"] = self._perform_vecm_wald_test(
                vecm_result, cause_var=1, effect_var=0, k_ar_diff=k_ar_diff, alpha=alpha
            )
            
            # Add model diagnostics
            results['model_info'] = self._get_vecm_diagnostics(vecm_result, df)
            
            return results
            
        except Exception as e:
            self.log.error(f"VECM Granger causality test failed: {e}")
            return self._empty_vecm_result()
    
    def _perform_vecm_wald_test(self, vecm_result, cause_var: int, effect_var: int, 
                               k_ar_diff: int, alpha: float) -> Dict:
        """
        Perform Wald test for Granger causality in VECM
        
        Tests H0: γ_ij^(1) = γ_ij^(2) = ... = γ_ij^(k) = 0
        where γ_ij^(l) is the coefficient of the l-th lag of variable j in equation i
        """
        try:
            # Get the coefficient matrix for the lagged differences
            # vecm_result.params contains [alpha, gamma, const/trend] coefficients
            
            # Extract gamma coefficients (lagged differences)
            gamma_params = vecm_result.gamma
            
            # Get covariance matrix of parameters
            param_cov = vecm_result.cov_params()
            
            # Construct restriction matrix for Wald test
            # We want to test if coefficients of cause_var in effect_var equation are zero
            
            # Number of equations (should be 2 for bivariate)
            n_eq = gamma_params.shape[0]
            
            # Number of variables * number of lags
            n_gamma_per_eq = gamma_params.shape[1]
            
            # Create restriction matrix
            restriction_matrix = self._create_restriction_matrix(
                n_eq, n_gamma_per_eq, cause_var, effect_var, k_ar_diff
            )
            
            # Get the relevant parameter vector (flatten gamma matrix)
            gamma_vec = gamma_params.flatten()
            
            # Extract relevant covariance submatrix
            gamma_cov = self._extract_gamma_covariance(param_cov, vecm_result)
            
            # Perform Wald test: W = (Rβ)' * (R * Σ * R')^(-1) * (Rβ)
            restricted_params = restriction_matrix @ gamma_vec
            
            # Calculate the quadratic form
            middle_matrix = restriction_matrix @ gamma_cov @ restriction_matrix.T
            
            # Check if matrix is invertible
            if np.linalg.det(middle_matrix) == 0:
                self.log.warning("Singular covariance matrix in Wald test")
                return self._empty_wald_result()
            
            wald_stat = restricted_params.T @ np.linalg.inv(middle_matrix) @ restricted_params
            
            # Degrees of freedom = number of restrictions
            df = restriction_matrix.shape[0]
            
            # Calculate p-value
            p_value = 1 - chi2.cdf(wald_stat, df)
            
            return {
                'wald_statistic': float(wald_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': df,
                'critical_value': chi2.ppf(1 - alpha, df),
                'significant': p_value < alpha,
                'restricted_coefficients': restricted_params.tolist(),
                'test_type': 'VECM_Wald'
            }
            
        except Exception as e:
            self.log.error(f"VECM Wald test failed: {e}")
            return self._empty_wald_result()
    
    def _create_restriction_matrix(self, n_eq: int, n_gamma_per_eq: int, 
                                  cause_var: int, effect_var: int, k_ar_diff: int) -> np.ndarray:
        """
        Create restriction matrix for Wald test
        
        For a bivariate VECM with k lags, we test if the coefficients of
        cause_var in the effect_var equation are jointly zero.
        """
        try:
            # Total number of gamma parameters
            total_gamma_params = n_eq * n_gamma_per_eq
            
            # Number of variables (should be 2 for bivariate)
            n_vars = n_eq
            
            # Create restriction matrix
            # We want to restrict k_ar_diff coefficients
            R = np.zeros((k_ar_diff, total_gamma_params))
            
            # Fill in the restriction matrix
            for lag in range(k_ar_diff):
                # Position in the flattened parameter vector
                # Structure: [eq0_lag0_var0, eq0_lag0_var1, eq0_lag1_var0, eq0_lag1_var1, 
                #            eq1_lag0_var0, eq1_lag0_var1, eq1_lag1_var0, eq1_lag1_var1]
                
                row_offset = effect_var * n_gamma_per_eq  # Which equation
                col_offset = lag * n_vars + cause_var     # Which coefficient within equation
                
                position = row_offset + col_offset
                
                if position < total_gamma_params:
                    R[lag, position] = 1
            
            return R
            
        except Exception as e:
            self.log.error(f"Failed to create restriction matrix: {e}")
            # Return identity matrix as fallback
            return np.eye(min(k_ar_diff, total_gamma_params))
    
    def _extract_gamma_covariance(self, param_cov: pd.DataFrame, vecm_result) -> np.ndarray:
        """
        Extract the covariance matrix for gamma parameters from full parameter covariance
        """
        try:
            # Get parameter names to identify gamma parameters
            param_names = param_cov.columns.tolist()
            
            # Find gamma parameter indices
            gamma_indices = []
            for i, name in enumerate(param_names):
                if 'L' in name and 'D.' in name:  # Lagged difference terms
                    gamma_indices.append(i)
            
            if not gamma_indices:
                # Fallback: assume gamma parameters are in the middle
                n_alpha = vecm_result.alpha.size
                n_gamma = vecm_result.gamma.size
                gamma_indices = list(range(n_alpha, n_alpha + n_gamma))
            
            # Extract submatrix
            gamma_cov = param_cov.iloc[gamma_indices, gamma_indices].values
            
            return gamma_cov
            
        except Exception as e:
            self.log.error(f"Failed to extract gamma covariance: {e}")
            # Return identity matrix as fallback
            size = vecm_result.gamma.size if hasattr(vecm_result, 'gamma') else 4
            return np.eye(size)
    
    def _get_deterministic_string(self, det_order: int) -> str:
        """Convert deterministic order to string format for VECM"""
        det_map = {
            -1: 'nc',    # No constant
            0: 'co',     # Constant outside cointegration
            1: 'ci',     # Constant inside cointegration
            2: 'lo',     # Linear trend outside
            3: 'li'      # Linear trend inside
        }
        return det_map.get(det_order, 'co')
    
    def _get_vecm_diagnostics(self, vecm_result, df: pd.DataFrame) -> Dict:
        """Get comprehensive VECM model diagnostics"""
        try:
            diagnostics = {
                'aic': vecm_result.aic,
                'bic': vecm_result.bic,
                'hqic': vecm_result.hqic,
                'log_likelihood': vecm_result.llf,
                'n_obs': vecm_result.nobs,
                'coint_rank': vecm_result.coint_rank,
                'k_ar_diff': vecm_result.k_ar_diff
            }
            
            # Add residual diagnostics
            residuals = vecm_result.resid
            
            # Durbin-Watson test for serial correlation
            dw_stats = []
            for i in range(residuals.shape[1]):
                dw_stat = durbin_watson(residuals[:, i])
                dw_stats.append(dw_stat)
            
            diagnostics['durbin_watson'] = dw_stats
            
            # Residual correlation matrix
            residual_corr = np.corrcoef(residuals.T)
            diagnostics['residual_correlation'] = residual_corr.tolist()
            
            # Cointegration vector(s)
            if hasattr(vecm_result, 'beta'):
                diagnostics['cointegration_vectors'] = vecm_result.beta.tolist()
            
            # Adjustment coefficients (alpha)
            if hasattr(vecm_result, 'alpha'):
                diagnostics['adjustment_coefficients'] = vecm_result.alpha.tolist()
            
            return diagnostics
            
        except Exception as e:
            self.log.error(f"Failed to get VECM diagnostics: {e}")
            return {'error': str(e)}
    
    def _empty_vecm_result(self) -> Dict:
        """Return empty VECM result structure"""
        return {
            'var1_causes_var2': self._empty_wald_result(),
            'var2_causes_var1': self._empty_wald_result(),
            'model_info': {'error': 'Failed to fit VECM model'}
        }
    
    def _empty_wald_result(self) -> Dict:
        """Return empty Wald test result"""
        return {
            'wald_statistic': np.nan,
            'p_value': np.nan,
            'degrees_of_freedom': np.nan,
            'critical_value': np.nan,
            'significant': False,
            'restricted_coefficients': [],
            'test_type': 'VECM_Wald'
        }
    
    def vecm_impulse_response_analysis(self, df: pd.DataFrame, 
                                     coint_rank: int = 1, 
                                     k_ar_diff: int = 1,
                                     periods: int = 20) -> Dict:
        """
        Perform impulse response analysis on VECM
        
        Args:
            df: DataFrame with cointegrated series
            coint_rank: Number of cointegrating relationships
            k_ar_diff: Number of lags in VECM
            periods: Number of periods for impulse response
            
        Returns:
            Dictionary with impulse response functions
        """
        try:
            # Fit VECM
            vecm_model = VECM(df, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
            vecm_result = vecm_model.fit()
            
            # Generate impulse responses
            irf = vecm_result.irf(periods=periods)
            
            var_names = df.columns.tolist()
            
            # Extract impulse response functions
            irf_results = {}
            
            for i, impulse_var in enumerate(var_names):
                for j, response_var in enumerate(var_names):
                    key = f"{impulse_var}_to_{response_var}"
                    irf_results[key] = irf.irfs[:, j, i].tolist()
            
            # Add confidence intervals if available
            if hasattr(irf, 'ci'):
                irf_results['confidence_intervals'] = {
                    'lower': irf.ci[:, :, :, 0].tolist(),
                    'upper': irf.ci[:, :, :, 1].tolist()
                }
            
            return {
                'impulse_responses': irf_results,
                'periods': periods,
                'variable_names': var_names
            }
            
        except Exception as e:
            self.log.error(f"VECM impulse response analysis failed: {e}")
            return {'error': str(e)}
