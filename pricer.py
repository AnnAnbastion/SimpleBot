# pricer.py
import numpy as np
from scipy.stats import norm
from datetime import datetime
from typing import Dict, Optional, Union
import logging

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
        """Safely calculate time to expiry"""
        try:
            if isinstance(expiration, datetime):
                now = datetime.now()
                time_diff = (expiration - now).total_seconds()
                return max(time_diff / (365.25 * 24 * 3600), 0.0001)  # Minimum 1 hour
            else:
                logger.warning(f"Invalid expiration type: {type(expiration)}")
                return 0.0001
        except Exception as e:
            logger.error(f"Error calculating time to expiry: {e}")
            return 0.0001
        
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price with error handling"""
        try:
            # Convert all inputs to safe floats
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)  # Default 30% vol
            
            # Validate inputs
            if S <= 0 or K <= 0 or sigma <= 0:
                return 0.0
                
            if T <= 0:
                # Handle expired options
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
                    
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return max(price, 0)
            
        except Exception as e:
            logger.error(f"Error in black_scholes_price: {e}")
            return 0.0
        
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks with error handling"""
        try:
            # Convert all inputs to safe floats
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)
            
            # Default Greeks for invalid inputs
            default_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
            if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return default_greeks
                
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
                
            # Gamma (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta = theta_part1 - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:
                theta = theta_part1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = theta / 365  # Convert to daily theta
            
            # Vega (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
            
            # Rho
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
                
            return {
                'delta': self._safe_float(delta),
                'gamma': self._safe_float(gamma),
                'theta': self._safe_float(theta),
                'vega': self._safe_float(vega),
                'rho': self._safe_float(rho)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> Optional[float]:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            # Convert inputs to safe floats
            market_price = self._safe_float(market_price)
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            
            if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
                return None
                
            # Initial guess
            sigma = 0.3
            
            for iteration in range(50):  # Reduced iterations
                try:
                    price = self.black_scholes_price(S, K, T, r, sigma, option_type)
                    greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
                    vega = greeks['vega'] * 100  # Convert to per unit vol change
                    
                    if abs(vega) < 1e-6:
                        break
                        
                    diff = market_price - price
                    if abs(diff) < 1e-6:
                        return sigma
                        
                    sigma_new = sigma + diff / vega if vega != 0 else sigma
                    
                    # Bounds checking
                    if sigma_new <= 0.001:
                        sigma_new = 0.001
                    elif sigma_new > 10:
                        sigma_new = 10
                        
                    # Check for convergence
                    if abs(sigma_new - sigma) < 1e-6:
                        break
                        
                    sigma = sigma_new
                    
                except Exception as e:
                    logger.warning(f"Error in IV iteration {iteration}: {e}")
                    break
                    
            return sigma if 0.001 <= sigma <= 10 else None
            
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return None
        
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
                    orderbook = orderbooks.get(instrument_name)
                    
                    if not orderbook:
                        logger.debug(f"No orderbook for {instrument_name}")
                        continue
                        
                    # Check if we have valid bid/ask data
                    best_bid = self._safe_float(orderbook.best_bid)
                    best_ask = self._safe_float(orderbook.best_ask)
                    
                    if best_bid <= 0 or best_ask <= 0:
                        logger.debug(f"Invalid bid/ask for {instrument_name}: bid={best_bid}, ask={best_ask}")
                        continue
                        
                    # Calculate time to expiration
                    time_to_expiry = self._safe_time_to_expiry(instrument.expiration)
                    
                    if time_to_expiry <= 0:
                        logger.debug(f"Expired instrument: {instrument_name}")
                        continue
                        
                    # Market data
                    market_price = (best_bid + best_ask) / 2
                    strike = self._safe_float(instrument.strike)
                    
                    if strike <= 0:
                        logger.debug(f"Invalid strike for {instrument_name}: {strike}")
                        continue
                        
                    # Calculate implied volatility
                    implied_vol = self.calculate_implied_volatility(
                        market_price, spot_price, strike, 
                        time_to_expiry, self.risk_free_rate, instrument.option_type
                    )
                    
                    if implied_vol is None or implied_vol <= 0:
                        logger.debug(f"Could not calculate IV for {instrument_name}")
                        continue
                        
                    # Calculate theoretical price and Greeks
                    theoretical_price = self.black_scholes_price(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, instrument.option_type
                    )
                    
                    greeks = self.calculate_greeks(
                        spot_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, instrument.option_type
                    )
                    
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
                        **greeks,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing instrument {instrument_name}: {e}")
                    continue
                    
            logger.debug(f"Successfully calculated pricing for {len(results)} instruments")
            return results
            
        except Exception as e:
            logger.error(f"Error in calculate_all: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    # Add this method to your OptionsPricer class in pricer.py
    def get_cached_results(self) -> Dict:
        """Get cached pricing results from data manager"""
        return self.data_manager.get_latest_pricing_results()