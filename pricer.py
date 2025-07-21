# pricer.py
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

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
        """Calculate time to expiry in years"""
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

    def _normal_cdf(self, x: float) -> float:
        """Calculate cumulative normal distribution N(x)"""
        try:
            # Using the error function approximation
            return 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0 * x * x / np.pi)))
        except:
            # Fallback to scipy if available, otherwise simple approximation
            try:
                from scipy.stats import norm
                return norm.cdf(x)
            except:
                # Simple approximation for extreme cases
                if x > 6:
                    return 1.0
                elif x < -6:
                    return 0.0
                else:
                    return 0.5

    def _normal_pdf(self, x: float) -> float:
        """Calculate normal probability density function φ(x)"""
        try:
            return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)
        except:
            return 0.0

    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> float:
        """
        Calculate Black-Scholes option price using direct equations
        
        S: Current stock price
        K: Strike price  
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        try:
            # Input validation and conversion
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)
            
            if S <= 0 or K <= 0 or sigma <= 0:
                return 0.0
                
            # Handle expiration case
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if option_type.lower() == 'call':
                price = S * self._normal_cdf(d1) - K * np.exp(-r * T) * self._normal_cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * self._normal_cdf(-d1)
                
            return max(price, 0.0)
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return 0.0

    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks using direct equations
        
        Returns: Dictionary with delta, gamma, theta, vega, rho
        """
        try:
            # Input validation
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)
            
            # Default values for invalid inputs
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate Greeks
            
            # Delta: ∂V/∂S
            if option_type.lower() == 'call':
                delta = self._normal_cdf(d1)
            else:  # put
                delta = self._normal_cdf(d1) - 1.0
            
            # Gamma: ∂²V/∂S² (same for calls and puts)
            gamma = self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta: -∂V/∂t (time decay)
            theta_common = -(S * self._normal_pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta = theta_common - r * K * np.exp(-r * T) * self._normal_cdf(d2)
            else:  # put
                theta = theta_common + r * K * np.exp(-r * T) * self._normal_cdf(-d2)
            
            # Convert theta to daily (divide by 365)
            theta_daily = theta / 365.0
            
            # Vega: ∂V/∂σ (same for calls and puts)
            vega = S * self._normal_pdf(d1) * np.sqrt(T)
            # Convert to per 1% volatility change
            vega_percent = vega / 100.0
            
            # Rho: ∂V/∂r
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * self._normal_cdf(d2)
            else:  # put
                rho = -K * T * np.exp(-r * T) * self._normal_cdf(-d2)
            
            # Convert rho to per 1% rate change
            rho_percent = rho / 100.0
            
            # Validate results
            delta = max(-1.0, min(1.0, delta))  # Delta should be between -1 and 1
            gamma = max(0.0, gamma)  # Gamma should be positive
            vega_percent = max(0.0, vega_percent)  # Vega should be positive
            
            return {
                'delta': self._safe_float(delta),
                'gamma': self._safe_float(gamma),
                'theta': self._safe_float(theta_daily),
                'vega': self._safe_float(vega_percent),
                'rho': self._safe_float(rho_percent)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Uses the direct Black-Scholes equations with numerical methods
        """
        def objective(vol):
            return self.black_scholes_price(S, K, T, r, vol, option_type) - market_price
        
        try:
            return brentq(objective, 0.001, 5.0)  # Search between 0.1% and 500%
        except:
            return None

    def calculate_all(self) -> Dict:
        """
        Calculate pricing and risk metrics for all instruments
        
        Main calculation loop that processes all available options
        """
        try:
            # Get market data
            display_data = self.data_manager.get_display_data()
            spot_price = self._safe_float(display_data.get('spot_price', 0))
            
            if spot_price <= 0:
                logger.debug("No valid spot price available")
                return {}
                
            results = {}
            instruments = display_data.get('instruments', {})
            orderbooks = display_data.get('orderbooks', {})
            
            logger.info(f"Processing {len(instruments)} instruments with spot price ${spot_price:,.2f}")
            
            for instrument_name, instrument in instruments.items():
                try:
                    # Get market data for this instrument
                    orderbook = orderbooks.get(instrument_name)
                    if not orderbook:
                        continue
                        
                    best_bid = self._safe_float(orderbook.best_bid)
                    best_ask = self._safe_float(orderbook.best_ask)
                    
                    # Basic validation
                    if best_bid <= 0 or best_ask <= 0:
                        continue
                    
                    if best_ask <= best_bid:  # Invalid spread
                        continue
                    
                    # Calculate market price
                    market_price = (best_bid + best_ask) / 2.0
                    
                    # Get instrument parameters
                    strike = self._safe_float(instrument.strike)
                    time_to_expiry = self._safe_time_to_expiry(instrument.expiration)
                    option_type = instrument.option_type.lower()
                    
                    if strike <= 0 or time_to_expiry <= 0:
                        continue
                    
                    # Calculate intrinsic and time value
                    if option_type == 'call':
                        intrinsic_value = max(spot_price - strike, 0)
                    else:
                        intrinsic_value = max(strike - spot_price, 0)
                    
                    time_value = max(market_price - intrinsic_value, 0)
                    
                    # Calculate implied volatility
                    implied_vol = self.calculate_implied_volatility(
                        market_price, spot_price, strike, 
                        time_to_expiry, self.risk_free_rate, option_type
                    )
                    
                    if implied_vol is None:
                        logger.debug(f"Could not calculate IV for {instrument_name}")
                        continue
                    
                    # Calculate theoretical price using calculated IV
                    theoretical_price = self.black_scholes_price(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, option_type
                    )
                    
                    # Calculate Greeks
                    greeks = self.calculate_greeks(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, option_type
                    )
                    
                    # Store results
                    results[instrument_name] = {
                        'spot_price': spot_price,
                        'strike': strike,
                        'time_to_expiry': time_to_expiry,
                        'option_type': option_type,
                        'market_price': market_price,
                        'bid': best_bid,
                        'ask': best_ask,
                        'theoretical_price': theoretical_price,
                        'implied_volatility': implied_vol,
                        'price_diff': market_price - theoretical_price,
                        'intrinsic_value': intrinsic_value,
                        'time_value': time_value,
                        **greeks,
                        'timestamp': datetime.now()
                    }
                    
                    # Log first few successful calculations
                    if len(results) <= 5:
                        logger.info(f"✅ {instrument_name}: ${market_price:.2f}, IV={implied_vol:.1%}, Δ={greeks['delta']:.3f}")
                        
                except Exception as e:
                    logger.debug(f"Error processing {instrument_name}: {e}")
                    continue
            
            # Summary statistics
            calls = sum(1 for r in results.values() if r['option_type'] == 'call')
            puts = sum(1 for r in results.values() if r['option_type'] == 'put')
            
            logger.info(f"Successfully calculated pricing for {len(results)} instruments ({calls} calls, {puts} puts)")
            return results
            
        except Exception as e:
            logger.error(f"Error in calculate_all: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_cached_results(self) -> Dict:
        """Get cached pricing results from data manager"""
        return self.data_manager.get_latest_pricing_results()