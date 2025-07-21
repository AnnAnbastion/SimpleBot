# pricer.py
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Union
import logging

# Set up logger first
logger = logging.getLogger(__name__)

# Try to import professional libraries
try:
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    VOLLIB_AVAILABLE = True
    logger.info("py_vollib library available")
except ImportError as e:
    VOLLIB_AVAILABLE = False
    logger.warning(f"py_vollib not available: {e}, using fallback implementation")

# Always import scipy as fallback
from scipy.stats import norm
try:
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy.optimize not available")

class OptionsPricer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default
        
    def _safe_time_to_expiry(self, expiration) -> float:
        """Safely calculate time to expiry in years"""
        try:
            if isinstance(expiration, datetime):
                now = datetime.now()
                time_diff = (expiration - now).total_seconds()
                years = time_diff / (365.25 * 24 * 3600)
                return max(years, 1/365)  # Minimum 1 day
            else:
                logger.warning(f"Invalid expiration type: {type(expiration)}")
                return 1/365
        except Exception as e:
            logger.error(f"Error calculating time to expiry: {e}")
            return 1/365

    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> Optional[float]:
        """Calculate IV using best available method"""
        try:
            # Input validation
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            market_price = self._safe_float(market_price)
            
            if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
                return None
            
            # Check intrinsic value bounds
            if option_type.lower() == 'call':
                intrinsic = max(S - K, 0)
            else:
                intrinsic = max(K - S, 0)
            
            # Market price must be at least intrinsic value (with small tolerance)
            if market_price < intrinsic * 0.95:
                logger.debug(f"Market price {market_price} below intrinsic {intrinsic}")
                return None
            
            # Try py_vollib first if available
            if VOLLIB_AVAILABLE:
                try:
                    flag = 'c' if option_type.lower() == 'call' else 'p'
                    iv = implied_volatility(market_price, S, K, T, r, flag)
                    
                    if 0.001 <= iv <= 10:  # Between 0.1% and 1000%
                        return iv
                except Exception as e:
                    logger.debug(f"py_vollib IV failed: {e}")
            
            # Fallback to Brent's method
            if SCIPY_AVAILABLE:
                return self._calculate_iv_brent(market_price, S, K, T, r, option_type)
            else:
                return self._calculate_iv_bisection(market_price, S, K, T, r, option_type)
                
        except Exception as e:
            logger.debug(f"Error in IV calculation: {e}")
            return None

    def _calculate_iv_brent(self, market_price: float, S: float, K: float, 
                           T: float, r: float, option_type: str) -> Optional[float]:
        """Calculate IV using Brent's method"""
        try:
            def price_diff(vol):
                try:
                    bs_price = self._black_scholes_fallback(S, K, T, r, vol, option_type)
                    return bs_price - market_price
                except:
                    return float('inf')
            
            # Try to find IV between 0.1% and 500%
            iv = brentq(price_diff, 0.001, 5.0, xtol=1e-6, maxiter=100)
            return iv if 0.001 <= iv <= 5.0 else None
                
        except Exception as e:
            logger.debug(f"Brent IV calculation failed: {e}")
            return None

    def _calculate_iv_bisection(self, market_price: float, S: float, K: float, 
                               T: float, r: float, option_type: str) -> Optional[float]:
        """Simple bisection method for IV calculation"""
        try:
            low_vol = 0.001
            high_vol = 5.0
            
            for _ in range(50):  # Max iterations
                mid_vol = (low_vol + high_vol) / 2
                
                bs_price = self._black_scholes_fallback(S, K, T, r, mid_vol, option_type)
                
                if abs(bs_price - market_price) < 1e-6:
                    return mid_vol
                
                if bs_price > market_price:
                    high_vol = mid_vol
                else:
                    low_vol = mid_vol
                    
                if abs(high_vol - low_vol) < 1e-6:
                    break
            
            return (low_vol + high_vol) / 2
            
        except Exception as e:
            logger.debug(f"Bisection IV calculation failed: {e}")
            return None

    def _black_scholes_price(self, S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str) -> float:
        """Calculate BS price - works with both libraries and fallback"""
        try:
            # Try py_vollib first
            if VOLLIB_AVAILABLE:
                try:
                    flag = 'c' if option_type.lower() == 'call' else 'p'
                    price = black_scholes(flag, S, K, T, r, sigma)
                    return max(price, 0.0)
                except Exception as e:
                    logger.debug(f"py_vollib BS failed: {e}")
            
            # Fallback implementation
            return self._black_scholes_fallback(S, K, T, r, sigma, option_type)
            
        except Exception as e:
            logger.error(f"Error in BS price calculation: {e}")
            return 0.0

    def _black_scholes_fallback(self, S: float, K: float, T: float, r: float, 
                               sigma: float, option_type: str) -> float:
        """Fallback BS implementation using scipy"""
        try:
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)
            
            if S <= 0 or K <= 0 or sigma <= 0:
                return 0.0
                
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return max(price, 0)
            
        except Exception as e:
            logger.error(f"Error in fallback BS price: {e}")
            return 0.0

    def _calculate_greeks(self, S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate Greeks using best available method"""
        try:
            # Input validation
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)
            
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            # Try py_vollib first
            if VOLLIB_AVAILABLE:
                try:
                    flag = 'c' if option_type.lower() == 'call' else 'p'
                    
                    delta_val = delta(flag, S, K, T, r, sigma)
                    gamma_val = gamma(flag, S, K, T, r, sigma)
                    theta_val = theta(flag, S, K, T, r, sigma) / 365  # Daily theta
                    vega_val = vega(flag, S, K, T, r, sigma) / 100   # 1% vol change
                    rho_val = rho(flag, S, K, T, r, sigma) / 100     # 1% rate change
                    
                    return {
                        'delta': self._safe_float(delta_val),
                        'gamma': self._safe_float(gamma_val),
                        'theta': self._safe_float(theta_val),
                        'vega': self._safe_float(vega_val),
                        'rho': self._safe_float(rho_val)
                    }
                except Exception as e:
                    logger.debug(f"py_vollib Greeks failed: {e}")
            
            # Fallback Greeks calculation
            return self._calculate_greeks_fallback(S, K, T, r, sigma, option_type)
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def _calculate_greeks_fallback(self, S: float, K: float, T: float, r: float, 
                                  sigma: float, option_type: str) -> Dict[str, float]:
        """Fallback Greeks calculation"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta_val = norm.cdf(d1)
            else:
                delta_val = norm.cdf(d1) - 1
                
            # Gamma (same for calls and puts)
            gamma_val = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta_val = theta_part1 - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:
                theta_val = theta_part1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_val = theta_val / 365  # Convert to daily theta
            
            # Vega (same for calls and puts)
            vega_val = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
            
            # Rho
            if option_type.lower() == 'call':
                rho_val = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
                
            return {
                'delta': self._safe_float(delta_val),
                'gamma': self._safe_float(gamma_val),
                'theta': self._safe_float(theta_val),
                'vega': self._safe_float(vega_val),
                'rho': self._safe_float(rho_val)
            }
            
        except Exception as e:
            logger.error(f"Error in fallback Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def calculate_all(self) -> Dict:
        """Calculate pricing and risk metrics for all instruments"""
        try:
            market_data = self.data_manager.get_all_market_data()
            spot_price = self._safe_float(market_data.get('spot_price', 0))
            
            if spot_price <= 0:
                logger.debug("No valid spot price available")
                return {}
                
            results = {}
            instruments = market_data.get('instruments', {})
            orderbooks = market_data.get('orderbooks', {})
            
            logger.debug(f"Processing {len(instruments)} instruments with {len(orderbooks)} orderbooks")
            
            for instrument_name, instrument in instruments.items():
                try:
                    # Get orderbook data
                    orderbook = orderbooks.get(instrument_name)
                    if not orderbook:
                        continue
                        
                    best_bid = self._safe_float(orderbook.best_bid)
                    best_ask = self._safe_float(orderbook.best_ask)
                    
                    # Check if we have valid bid/ask
                    if best_bid <= 0 or best_ask <= 0:
                        continue
                    
                    # Check for reasonable spread
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    
                    if spread <= 0:
                        continue
                    
                    # Basic instrument data
                    time_to_expiry = self._safe_time_to_expiry(instrument.expiration)
                    if time_to_expiry <= 0:
                        continue
                        
                    strike = self._safe_float(instrument.strike)
                    if strike <= 0:
                        continue
                    
                    market_price = mid_price
                    
                    # Calculate intrinsic value
                    if instrument.option_type.lower() == 'call':
                        intrinsic = max(spot_price - strike, 0)
                    else:
                        intrinsic = max(strike - spot_price, 0)
                    
                    # Basic validation - market price should be reasonable
                    if market_price < intrinsic * 0.9:
                        continue
                    
                    # Calculate IV
                    implied_vol = self.calculate_implied_volatility(
                        market_price, spot_price, strike, 
                        time_to_expiry, self.risk_free_rate, instrument.option_type
                    )
                    
                    if implied_vol is None or implied_vol <= 0 or implied_vol > 5:
                        continue
                        
                    # Calculate theoretical price
                    theoretical_price = self._black_scholes_price(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, instrument.option_type
                    )
                    
                    # Calculate Greeks
                    greeks = self._calculate_greeks(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, instrument.option_type
                    )
                    
                    # Store results
                    results[instrument_name] = {
                        'spot_price': spot_price,
                        'strike': strike,
                        'time_to_expiry': time_to_expiry,
                        'option_type': instrument.option_type,
                        'market_price': market_price,
                        'bid': best_bid,
                        'ask': best_ask,
                        'theoretical_price': theoretical_price,
                        'implied_volatility': implied_vol,
                        'price_diff': market_price - theoretical_price,
                        'intrinsic_value': intrinsic,
                        'time_value': market_price - intrinsic,
                        **greeks,
                        'timestamp': datetime.now()
                    }
                    
                    # Debug log for first few successful calculations
                    if len(results) <= 3:
                        logger.info(f"Calculated {instrument_name}: Price=${market_price:.2f}, IV={implied_vol:.1%}, Delta={greeks['delta']:.3f}")
                        
                except Exception as e:
                    logger.debug(f"Error processing instrument {instrument_name}: {e}")
                    continue
                    
            logger.info(f"Successfully calculated pricing for {len(results)} instruments")
            return results
            
        except Exception as e:
            logger.error(f"Error in calculate_all: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_cached_results(self) -> Dict:
        """Get cached pricing results from data manager"""
        return self.data_manager.get_latest_pricing_results()