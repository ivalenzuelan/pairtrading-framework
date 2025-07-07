from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

class CausalityFilter:
    def __init__(self, max_lag=4, alpha=0.05):
        self.max_lag = max_lag
        self.alpha = alpha
        
    def test_pair(self, returns_i: pd.Series, returns_j: pd.Series) -> Tuple[bool, Optional[str]]:
        """
        Test for lead-lag relationship between two assets
        Returns: (passes_test, leader) or (False, None)
        """
        # Align and prepare data
        data = pd.DataFrame({
            'asset_i': returns_i,
            'asset_j': returns_j
        }).dropna()
        
        # Check sufficient data
        if len(data) < self.max_lag * 2:
            return False, None
        
        # Test i -> j direction
        ij_passed, ij_lag = self._test_direction(data, 'asset_i', 'asset_j')
        
        # Test j -> i direction
        ji_passed, ji_lag = self._test_direction(data, 'asset_j', 'asset_i')
        
        if ij_passed and ji_passed:
            # Both significant - choose stronger relationship
            return True, 'asset_i' if ij_lag < ji_lag else 'asset_j'
        elif ij_passed:
            return True, 'asset_i'
        elif ji_passed:
            return True, 'asset_j'
        return False, None

    def _test_direction(self, data: pd.DataFrame, cause: str, effect: str) -> Tuple[bool, int]:
        """Test one direction of causality"""
        # Fit VAR model and select optimal lag
        model = VAR(data)
        lag_order = model.select_order(maxlags=self.max_lag)
        p = lag_order.aic
        
        # Check if lag is within practical range (1-4)
        if not (1 <= p <= self.max_lag):
            return False, 0
        
        # Fit VAR model with optimal lag
        results = model.fit(p)
        
        # Granger causality test
        gc_test = results.test_causality(effect, [cause], kind='f')
        if gc_test.pvalue > self.alpha:
            return False, 0
        
        # Check for at least one positive significant coefficient
        for lag in range(1, p + 1):
            coef = results.params.get(f'{cause}_L{lag}')
            pvalue = results.pvalues.get(f'{cause}_L{lag}')
            if coef and pvalue and coef > 0 and pvalue < self.alpha:
                return True, lag
                
        return False, 0