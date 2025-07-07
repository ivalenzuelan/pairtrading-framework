#!/usr/bin/env python3
"""
Enhanced statistical tests for cointegration analysis with proper VECM implementation
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import linregress, jarque_bera, normaltest
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import grangercausalitytests, coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from typing import List, Dict, Tuple, Optional, Union
import warnings
from scipy.stats import chi2
from statsmodels.stats.stattools import durbin_watson
warnings.filterwarnings('ignore')


class VECMGrangerCausalityMixin:
    """Mixin class for VECM-based Granger causality testing"""
    
    def vecm_granger_causality(self, data: pd.DataFrame, coint_rank: int = 1, 
                              lags: int = 1, test_type: str = 'wald') -> Dict:
        """
        Test Granger causality using VECM framework
        
        Args:
            data: DataFrame with cointegrated time series
            coint_rank: Number of cointegrating relationships
            lags: Number of lags in VECM
            test_type: Type of test ('wald', 'lr')
            
        Returns:
            Dictionary with causality test results
        """
        try:
            # Fit VECM model
            vecm_model = VECM(data, k_ar_diff=lags, coint_rank=coint_rank)
            vecm_results = vecm_model.fit()
            
            # Number of variables
            n_vars = data.shape[1]
            causality_results = {}
            
            # Test causality between all pairs
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        var_i = data.columns[i]
                        var_j = data.columns[j]
                        
                        try:
                            # Test if variable i Granger causes variable j
                            test_result = vecm_results.test_granger_causality(
                                caused=j, causing=i, kind='f'
                            )
                            
                            causality_results[f"{var_i}_causes_{var_j}"] = {
                                'statistic': test_result.statistic,
                                'pvalue': test_result.pvalue,
                                'critical_value': test_result.critical_value,
                                'significant': test_result.pvalue < 0.05
                            }
                            
                        except Exception as e:
                            self.log.warning(f"VECM causality test failed for {var_i}->{var_j}: {e}")
                            continue
            
            return {
                'causality_results': causality_results,
                'vecm_summary': str(vecm_results.summary()),
                'alpha': vecm_results.alpha,  # Error correction coefficients
                'beta': vecm_results.beta,    # Cointegrating vectors
                'aic': vecm_results.aic,
                'bic': vecm_results.bic
            }
            
        except Exception as e:
            self.log.error(f"VECM Granger causality test failed: {e}")
            return {'causality_results': {}}


class CointegrationAnalyzer(VECMGrangerCausalityMixin):
    """Enhanced cointegration analysis with comprehensive statistical validation"""
    
    def __init__(self, config: dict, log: logging.Logger = None):
        self.config = config
        self.log = log or logging.getLogger(__name__)
        self.vecm_config = config.get('vecm', {
            'max_lags': 10,
            'coint_rank': 1,
            'deterministic': 'ci'  # constant inside cointegrating relationship
        })
        
    def _validate_series(self, s1: pd.Series, s2: pd.Series, min_obs: int = 50) -> bool:
        """Validate input series for analysis"""
        if len(s1) < min_obs or len(s2) < min_obs:
            self.log.warning(f"Insufficient data: s1={len(s1)}, s2={len(s2)}, min={min_obs}")
            return False
        
        # Check for constant series
        if s1.std() == 0 or s2.std() == 0:
            self.log.warning(f"Constant series detected: s1_std={s1.std()}, s2_std={s2.std()}")
            return False
        
        # Check for excessive missing values
        s1_missing = s1.isnull().sum() / len(s1)
        s2_missing = s2.isnull().sum() / len(s2)
        if s1_missing > 0.1 or s2_missing > 0.1:
            self.log.warning(f"Excessive missing values: s1_missing={s1_missing:.2%}, s2_missing={s2_missing:.2%}")
            return False
        
        return True
    
    def fit_vecm_model(self, data: pd.DataFrame, coint_rank: int = None, 
                       lags: int = None, deterministic: str = None) -> Optional[Dict]:
        """
        Fit Vector Error Correction Model
        
        Args:
            data: DataFrame with cointegrated time series
            coint_rank: Number of cointegrating relationships (auto-detect if None)
            lags: Number of lags (auto-select if None)
            deterministic: Deterministic terms ('n', 'co', 'ci', 'lo', 'li')
            
        Returns:
            Dictionary with VECM results
        """
        try:
            # Auto-detect cointegrating rank if not specified
            if coint_rank is None:
                johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
                coint_rank = np.sum(johansen_result.lr1 > johansen_result.cvt[:, 1])
                self.log.info(f"Auto-detected cointegrating rank: {coint_rank}")
            
            # Auto-select lags if not specified
            if lags is None:
                lags = self._select_optimal_lags(data, self.vecm_config['max_lags'])
                self.log.info(f"Auto-selected lags: {lags}")
            
            # Use config defaults if not specified
            deterministic = deterministic or self.vecm_config['deterministic']
            
            # Fit VECM model
            vecm_model = VECM(
                data, 
                k_ar_diff=lags,
                coint_rank=coint_rank,
                deterministic=deterministic
            )
            vecm_results = vecm_model.fit()
            
            # Extract key results, using getattr for optional attributes
            return {
                'model': vecm_results,
                'alpha': vecm_results.alpha,  # Error correction coefficients
                'beta': vecm_results.beta,    # Cointegrating vectors
                'gamma': getattr(vecm_results, 'gamma', None),  # Short-run coefficients
                'aic': getattr(vecm_results, 'aic', None),
                'bic': getattr(vecm_results, 'bic', None),
                'loglik': getattr(vecm_results, 'llf', None),
                'coint_rank': coint_rank,
                'lags': lags,
                'deterministic': deterministic,
                'residuals': vecm_results.resid,
                'fitted_values': vecm_results.fittedvalues,
                'summary': str(vecm_results.summary())
            }
            
        except Exception as e:
            self.log.error(f"VECM fitting failed: {e}")
            return None
    
    def vecm_diagnostics(self, vecm_results: Dict) -> Dict:
        """
        Comprehensive VECM model diagnostics
        
        Args:
            vecm_results: Results from fit_vecm_model
            
        Returns:
            Dictionary with diagnostic test results
        """
        try:
            model = vecm_results['model']
            residuals = vecm_results['residuals']
            
            diagnostics = {}
            
            # Portmanteau test for residual autocorrelation
            try:
                portmanteau = model.test_serial_correlation(lags=10)
                diagnostics['portmanteau_test'] = {
                    'statistic': portmanteau.statistic,
                    'pvalue': portmanteau.pvalue,
                    'no_autocorr': portmanteau.pvalue > 0.05
                }
            except:
                diagnostics['portmanteau_test'] = {'pvalue': np.nan}
            
            # Jarque-Bera test for normality
            try:
                jb_test = model.test_normality()
                diagnostics['jarque_bera_test'] = {
                    'statistic': jb_test.statistic,
                    'pvalue': jb_test.pvalue,
                    'normal': jb_test.pvalue > 0.05
                }
            except:
                diagnostics['jarque_bera_test'] = {'pvalue': np.nan}
            
            # Test for heteroscedasticity
            try:
                het_test = model.test_heteroscedasticity(lags=5)
                diagnostics['heteroscedasticity_test'] = {
                    'statistic': het_test.statistic,
                    'pvalue': het_test.pvalue,
                    'homoscedastic': het_test.pvalue > 0.05
                }
            except:
                diagnostics['heteroscedasticity_test'] = {'pvalue': np.nan}
            
            # Information criteria
            diagnostics['information_criteria'] = {
                'aic': vecm_results['aic'],
                'bic': vecm_results['bic'],
                'loglik': vecm_results['loglik']
            }
            
            # Error correction terms significance
            alpha = vecm_results['alpha']
            diagnostics['error_correction'] = {
                'alpha_coefficients': alpha.tolist() if hasattr(alpha, 'tolist') else alpha,
                'alpha_significant': np.abs(alpha) > 0.1 if isinstance(alpha, np.ndarray) else False
            }
            
            return diagnostics
            
        except Exception as e:
            self.log.error(f"VECM diagnostics failed: {e}")
            return {}
    
    def adf_test(self, series: pd.Series, maxlag: int = None) -> Dict[str, float]:
        """Enhanced Augmented Dickey-Fuller test with automatic lag selection"""
        try:
            if maxlag is None:
                maxlag = int(12 * (len(series) / 100) ** 0.25)  # Schwert criterion
            
            result = adfuller(series.dropna(), maxlag=maxlag, autolag='AIC')
            
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'used_lag': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            self.log.error(f"ADF test failed: {e}")
            return {'pvalue': np.nan, 'is_stationary': False}
    
    def engle_granger_test(self, s1: pd.Series, s2: pd.Series, trend: str = 'c') -> Dict[str, float]:
        """
        Enhanced Engle-Granger cointegration test with multiple trend specifications
        
        Args:
            s1: First price series
            s2: Second price series
            trend: Trend specification ('c', 'ct', 'ctt', 'nc')
            
        Returns:
            Dictionary with test results
        """
        try:
            # Align series and validate
            aligned = pd.concat([s1, s2], axis=1).dropna()
            if not self._validate_series(aligned.iloc[:, 0], aligned.iloc[:, 1]):
                return {'pvalue': np.nan, 'cointegrated': False}
            
            # Test multiple trend specifications
            best_result = None
            best_pvalue = 1.0
            
            trends = [trend] if trend != 'auto' else ['c', 'ct', 'ctt', 'nc']
            
            for t in trends:
                try:
                    stat, pvalue, crit_val = coint(aligned.iloc[:, 0], aligned.iloc[:, 1], trend=t)
                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_result = {
                            'statistic': stat,
                            'pvalue': pvalue,
                            'critical_value': crit_val,
                            'trend': t,
                            'cointegrated': pvalue < self.config.get('eg_p_threshold', 0.05)
                        }
                except:
                    continue
            
            return best_result or {'pvalue': np.nan, 'cointegrated': False}
        
        except Exception as e:
            self.log.error(f"Engle-Granger test failed: {e}")
            return {'pvalue': np.nan, 'cointegrated': False}
    
    def johansen_test(self, df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Enhanced Johansen cointegration test with multiple specifications
        
        Args:
            df: DataFrame with price series
            det_order: Deterministic trend order
            k_ar_diff: Number of lags in VAR
            
        Returns:
            Dictionary with comprehensive test results
        """
        try:
            if not self._validate_series(df.iloc[:, 0], df.iloc[:, 1]):
                return {'cointegrated': False, 'n_coint_relations': 0}
            
            # Auto-select optimal lag length
            if k_ar_diff == 'auto':
                k_ar_diff = self._select_optimal_lags(df)
            
            # Run Johansen test
            result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Check for cointegration at different confidence levels
            conf_levels = [0.90, 0.95, 0.99]
            cointegrated = {}
            
            for i, conf in enumerate(conf_levels):
                # Trace statistic test
                trace_coint = result.lr1[0] > result.cvt[0, i]
                # Max eigenvalue test
                max_eig_coint = result.lr2[0] > result.cvm[0, i]
                
                cointegrated[f'{conf:.0%}'] = trace_coint and max_eig_coint
            
            return {
                'cointegrated': cointegrated['95%'],
                'cointegrated_levels': cointegrated,
                'trace_stat': result.lr1[0],
                'max_eig_stat': result.lr2[0],
                'trace_crit_95': result.cvt[0, 1],
                'max_eig_crit_95': result.cvm[0, 1],
                'eigenvectors': result.evec,
                'eigenvalues': result.eig,
                'n_coint_relations': np.sum(result.lr1 > result.cvt[:, 1])
            }
            
        except Exception as e:
            self.log.error(f"Johansen test failed: {e}")
            return {'cointegrated': False, 'n_coint_relations': 0}
    
    def _select_optimal_lags(self, df: pd.DataFrame, max_lags: int = 10) -> int:
        """Select optimal lag length using information criteria"""
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            
            model = VAR(df)
            lag_order = model.select_order(maxlags=max_lags)
            
            # Prefer AIC, fall back to BIC
            return lag_order.aic if lag_order.aic is not None else (lag_order.bic or 1)
        except:
            return 1
    
    def validate_cointegration(self, s1: pd.Series, s2: pd.Series) -> Optional[dict]:
        """
        Comprehensive cointegration validation pipeline
        
        Args:
            s1: First price series
            s2: Second price series
            
        Returns:
            Dictionary with validation results or None if validation fails
        """
        try:
            # Validate input series
            if not self._validate_series(s1, s2):
                self.log.debug(f"Series validation failed for {s1.name if hasattr(s1, 'name') else 'Series1'}/{s2.name if hasattr(s2, 'name') else 'Series2'}")
                return None
            
            # Align series
            aligned = pd.concat([s1, s2], axis=1).dropna()
            if len(aligned) < 50:
                self.log.warning(f"Insufficient aligned data for cointegration analysis: {len(aligned)} < 50")
                return None
            
            s1_aligned, s2_aligned = aligned.iloc[:, 0], aligned.iloc[:, 1]
            
            # Step 1: Engle-Granger test
            eg_result = self.engle_granger_test(s1_aligned, s2_aligned)
            self.log.debug(f"Engle-Granger result: cointegrated={eg_result.get('cointegrated', False)}, pvalue={eg_result.get('pvalue', 'N/A')}")
            
            # Step 2: Johansen test
            johansen_result = self.johansen_test(aligned)
            self.log.debug(f"Johansen result: cointegrated={johansen_result.get('cointegrated', False)}")
            
            # Step 3: Calculate hedge ratio and spread statistics
            hedge_ratio, intercept, spread_stats = self._calculate_spread_components(s1_aligned, s2_aligned)
            
            # Step 4: ADF test on residuals (if Engle-Granger passed)
            adf_result = None
            if eg_result.get('cointegrated', False):
                # Calculate residuals from Engle-Granger regression
                try:
                    from statsmodels.regression.linear_model import OLS
                    model = OLS(s2_aligned, s1_aligned).fit()
                    residuals = model.resid
                    adf_result = self.adf_test(residuals)
                except:
                    self.log.warning("Failed to calculate residuals for ADF test")
            
            # Step 5: VECM analysis (if Johansen test passed)
            vecm_results = None
            if johansen_result.get('cointegrated', False):
                try:
                    vecm_results = self.fit_vecm_model(aligned)
                except:
                    self.log.warning("VECM analysis failed")
            
            # Determine overall cointegration status
            eg_cointegrated = eg_result.get('cointegrated', False)
            johansen_cointegrated = johansen_result.get('cointegrated', False)
            
            # Consider cointegrated if either test passes
            is_cointegrated = eg_cointegrated or johansen_cointegrated
            
            if not is_cointegrated:
                self.log.debug(f"Neither Engle-Granger nor Johansen test showed cointegration")
                return None
            
            # Return results in the expected format for pair_screening.py
            return {
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'mean': spread_stats['mean'],
                'std': spread_stats['std'],
                's1_vol': spread_stats['s1_vol'],
                's2_vol': spread_stats['s2_vol'],
                'eg_pvalue': eg_result.get('pvalue', 1.0),
                'johansen': johansen_result.get('cointegrated', False),
                'cointegrated': is_cointegrated,
                'confidence_score': self._calculate_cointegration_confidence({
                    'engle_granger': eg_result,
                    'johansen': johansen_result,
                    'adf_residuals': adf_result,
                    'data_points': len(aligned)
                }),
                'data_points': len(aligned),
                'series_names': [s1.name if hasattr(s1, 'name') else 'Series1', 
                               s2.name if hasattr(s2, 'name') else 'Series2']
            }
            
        except Exception as e:
            self.log.error(f"Cointegration validation failed: {e}")
            return None
    
    def _calculate_spread_components(self, s1: pd.Series, s2: pd.Series) -> Tuple[float, float, dict]:
        """
        Calculate hedge ratio, intercept, and spread statistics
        
        Args:
            s1, s2: Aligned price series
            
        Returns:
            Tuple of (hedge_ratio, intercept, spread_stats)
        """
        try:
            # Calculate hedge ratio using OLS regression
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            
            # Regress s2 on s1: s2 = beta * s1 + alpha
            model = OLS(s2, add_constant(s1)).fit()
            hedge_ratio = model.params[1]  # Coefficient of s1
            intercept = model.params[0]    # Intercept
            
            # Calculate spread: spread = s2 - hedge_ratio * s1
            spread = s2 - hedge_ratio * s1
            
            # Calculate spread statistics
            spread_stats = {
                'mean': spread.mean(),
                'std': spread.std(),
                's1_vol': s1.std(),
                's2_vol': s2.std(),
                'spread_series': spread
            }
            
            return hedge_ratio, intercept, spread_stats
            
        except Exception as e:
            self.log.error(f"Failed to calculate spread components: {e}")
            # Return default values
            return 1.0, 0.0, {
                'mean': 0.0,
                'std': 1.0,
                's1_vol': s1.std(),
                's2_vol': s2.std(),
                'spread_series': s2 - s1
            }
    
    def _calculate_cointegration_confidence(self, results: dict) -> float:
        """Calculate confidence score for cointegration results"""
        score = 0.0
        
        # Engle-Granger test contribution
        eg_result = results.get('engle_granger', {})
        if eg_result.get('cointegrated', False):
            pvalue = eg_result.get('pvalue', 1.0)
            score += max(0, 0.4 * (1 - pvalue))  # Higher score for lower p-value
        
        # Johansen test contribution
        johansen_result = results.get('johansen', {})
        if johansen_result.get('cointegrated', False):
            score += 0.4
        
        # ADF test on residuals contribution
        adf_result = results.get('adf_residuals', {})
        if adf_result.get('is_stationary', False):
            score += 0.2
        
        # Data quality contribution
        data_points = results.get('data_points', 0)
        if data_points >= 200:
            score += 0.1
        elif data_points >= 100:
            score += 0.05
        
        return min(1.0, score)
    
    def determine_lead_lag_relationship(self, s1: pd.Series, s2: pd.Series, 
                                      symbol1: str, symbol2: str) -> Optional[dict]:
        """
        Determine lead-lag relationship between two series
        
        Args:
            s1, s2: Price series
            symbol1, symbol2: Symbol names
            
        Returns:
            Dictionary with lead-lag analysis results
        """
        try:
            # Align series
            aligned = pd.concat([s1, s2], axis=1).dropna()
            if len(aligned) < 50:
                return None
            
            s1_aligned, s2_aligned = aligned.iloc[:, 0], aligned.iloc[:, 1]
            
            # Cross-correlation analysis
            max_lag = min(20, len(aligned) // 4)
            cross_corr = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # s1 leads s2
                    corr = s1_aligned.iloc[:-lag].corr(s2_aligned.iloc[lag:])
                elif lag > 0:
                    # s2 leads s1
                    corr = s1_aligned.iloc[lag:].corr(s2_aligned.iloc[:-lag])
                else:
                    # No lag
                    corr = s1_aligned.corr(s2_aligned)
                
                cross_corr.append((lag, corr))
            
            # Find maximum correlation
            max_corr_lag, max_corr = max(cross_corr, key=lambda x: abs(x[1]))
            
            # Determine leader and follower
            if max_corr_lag < 0:
                leader, follower = symbol1, symbol2
                lag_periods = abs(max_corr_lag)
            elif max_corr_lag > 0:
                leader, follower = symbol2, symbol1
                lag_periods = max_corr_lag
            else:
                leader, follower = symbol1, symbol2
                lag_periods = 0
            
            return {
                'leader': leader,
                'follower': follower,
                'lag_periods': lag_periods,
                'max_correlation': max_corr,
                'cross_correlation': cross_corr,
                'relationship_strength': abs(max_corr)
            }
            
        except Exception as e:
            self.log.error(f"Lead-lag analysis failed: {e}")
            return None
    
    def comprehensive_vecm_analysis(self, data: pd.DataFrame, 
                                   symbol_names: List[str] = None) -> Dict:
        """
        Comprehensive VECM analysis including cointegration, causality, and diagnostics
        
        Args:
            data: DataFrame with time series data
            symbol_names: List of symbol names for labeling
            
        Returns:
            Dictionary with complete VECM analysis results
        """
        try:
            if symbol_names is None:
                symbol_names = [f"Series_{i}" for i in range(data.shape[1])]
            
            # Step 1: Test for cointegration
            johansen_result = self.johansen_test(data)
            
            if not johansen_result['cointegrated']:
                return {
                    'cointegrated': False,
                    'message': 'No cointegration found - VECM analysis not applicable'
                }
            
            # Step 2: Fit VECM model
            coint_rank = johansen_result['n_coint_relations']
            vecm_results = self.fit_vecm_model(data, coint_rank=coint_rank)
            
            if vecm_results is None:
                return {
                    'cointegrated': True,
                    'vecm_fitted': False,
                    'message': 'VECM model fitting failed'
                }
            
            # Step 3: Model diagnostics
            diagnostics = self.vecm_diagnostics(vecm_results)
            
            # Step 4: Granger causality tests
            causality_results = self.vecm_granger_causality(
                data, 
                coint_rank=coint_rank,
                lags=vecm_results['lags']
            )
            
            # Step 5: Impulse response analysis (if available)
            try:
                irf = vecm_results['model'].irf(periods=10)
                impulse_response = {
                    'available': True,
                    'periods': 10,
                    'summary': str(irf)
                }
            except:
                impulse_response = {'available': False}
            
            return {
                'cointegrated': True,
                'vecm_fitted': True,
                'cointegration_results': johansen_result,
                'vecm_results': {
                    'alpha': vecm_results['alpha'].tolist() if hasattr(vecm_results['alpha'], 'tolist') else vecm_results['alpha'],
                    'beta': vecm_results['beta'].tolist() if hasattr(vecm_results['beta'], 'tolist') else vecm_results['beta'],
                    'aic': vecm_results['aic'],
                    'bic': vecm_results['bic'],
                    'lags': vecm_results['lags'],
                    'coint_rank': vecm_results['coint_rank']
                },
                'diagnostics': diagnostics,
                'granger_causality': causality_results,
                'impulse_response': impulse_response,
                'symbol_names': symbol_names
            }
            
        except Exception as e:
            self.log.error(f"Comprehensive VECM analysis failed: {e}")
            return {
                'cointegrated': False,
                'error': str(e)
            }
    
    # [Rest of the methods remain the same as in original code]
    # Including: granger_causality_test, calculate_hedge_ratio, calculate_spread_stats,
    # _calculate_half_life, _calculate_hurst_exponent, _test_normality, 
    # _test_autocorrelation, validate_cointegration, _calculate_quality_score,
    # determine_lead_lag_relationship, _cross_correlation_analysis


# Example usage
def example_vecm_analysis():
    """Example of how to use the VECM functionality"""
    # Create sample data
    np.random.seed(42)
    n_obs = 500
    
    # Generate cointegrated series
    error_term = np.random.normal(0, 1, n_obs)
    series1 = np.cumsum(np.random.normal(0, 1, n_obs)) + error_term
    series2 = 0.8 * series1 + np.cumsum(np.random.normal(0, 0.5, n_obs)) + error_term
    
    data = pd.DataFrame({
        'Asset1': series1,
        'Asset2': series2
    })
    
    # Initialize analyzer
    config = {
        'vecm': {
            'max_lags': 5,
            'coint_rank': 1,
            'deterministic': 'ci'
        }
    }
    
    analyzer = CointegrationAnalyzer(config)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_vecm_analysis(data, ['Asset1', 'Asset2'])
    
    return results


# Convenience functions for backward compatibility
def engle_granger_test(s1: pd.Series, s2: pd.Series, log: logging.Logger = None) -> float:
    """Backward compatibility wrapper"""
    config = {'eg_p_threshold': 0.05}
    analyzer = CointegrationAnalyzer(config, log)
    result = analyzer.engle_granger_test(s1, s2)
    return result.get('pvalue', np.nan)


def johansen_test(df: pd.DataFrame, conf_level: float = 0.95, log: logging.Logger = None) -> bool:
    """Backward compatibility wrapper"""
    config = {}
    analyzer = CointegrationAnalyzer(config, log)
    result = analyzer.johansen_test(df)
    return result.get('cointegrated', False)


def validate_cointegration(s1: pd.Series, s2: pd.Series, config: dict, log: logging.Logger) -> Optional[dict]:
    """Backward compatibility wrapper"""
    analyzer = CointegrationAnalyzer(config, log)
    return analyzer.validate_cointegration(s1, s2)


def determine_lead_lag_relationship(s1: pd.Series, s2: pd.Series, symbol1: str, symbol2: str,
                                   config: dict, log: logging.Logger) -> Optional[Tuple[str, str]]:
    """Backward compatibility wrapper"""
    analyzer = CointegrationAnalyzer(config, log)
    result = analyzer.determine_lead_lag_relationship(s1, s2, symbol1, symbol2)
    
    if result:
        return result['leader'], result['follower']
    return None