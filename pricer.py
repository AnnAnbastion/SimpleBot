# pricer.py
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import logging
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.interpolate import interp1d
import pandas as pd

logger = logging.getLogger(__name__)

class OptionsPricer:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.risk_free_rate = 0.05  # Keep original BS model rate
        
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
                return max(years, 1/365)  # Minimum 1 day to avoid division by zero
            else:
                logger.warning(f"Invalid expiration type: {type(expiration)}")
                return 1/365
        except Exception as e:
            logger.error(f"Error calculating time to expiry: {e}")
            return 1/365

    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> float:
        """
        Original Black-Scholes option price using proven scipy functions
        
        S: Current stock price
        K: Strike price  
        T: Time to expiration (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        """
        try:
            # Input validation and conversion
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            sigma = self._safe_float(sigma, 0.3)  # Default to 30% vol, not 100%
            
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
            
            # Calculate option price using scipy's norm functions
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return max(price, 0.0)
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return 0.0

    def black_model_price(self, F: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """
        Black model for options on futures with zero risk-free rate
        
        F: Future price
        K: Strike price  
        T: Time to expiration (in years)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        """
        try:
            # Input validation and conversion
            F = self._safe_float(F)
            K = self._safe_float(K)
            T = self._safe_float(T)
            sigma = self._safe_float(sigma, 0.3)
            
            if F <= 0 or K <= 0 or sigma <= 0:
                return 0.0
                
            # Handle expiration case
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(F - K, 0)
                else:
                    return max(K - F, 0)
            
            # Calculate d1 and d2 for Black model (r=0)
            d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price using Black model
            if option_type.lower() == 'call':
                price = F * norm.cdf(d1) - K * norm.cdf(d2)
            else:  # put
                price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
                
            return max(price, 0.0)
            
        except Exception as e:
            logger.error(f"Error in Black model calculation: {e}")
            return 0.0

    def get_synthetic_future_price_method_a(self, option_expiry: datetime, 
                                          instruments: Dict, orderbooks: Dict) -> Optional[float]:
        """
        Method A: Deribit's "minimal parity gap" shortcut
        Find the strike with minimal put-call parity gap and extract forward price
        """
        try:
            # Find all options with the same expiry
            same_expiry_options = {}
            
            for instrument_name, instrument in instruments.items():
                if (instrument.option_type in ['call', 'put'] and 
                    instrument.expiration == option_expiry and
                    instrument_name in orderbooks):
                    
                    orderbook = orderbooks[instrument_name]
                    strike = self._safe_float(instrument.strike)
                    
                    if strike <= 0:
                        continue
                        
                    bid = self._safe_float(orderbook.best_bid)
                    ask = self._safe_float(orderbook.best_ask)
                    
                    if bid <= 0 or ask <= 0 or ask <= bid:
                        continue
                        
                    mid_price = (bid + ask) / 2.0
                    
                    if strike not in same_expiry_options:
                        same_expiry_options[strike] = {}
                    
                    same_expiry_options[strike][instrument.option_type] = mid_price
            
            # Calculate parity gaps for strikes that have both call and put
            parity_gaps = {}
            
            for strike, options in same_expiry_options.items():
                if 'call' in options and 'put' in options:
                    call_price = options['call']
                    put_price = options['put']
                    
                    # Parity gap = |C - P| (with r=0, forward = C - P + K)
                    parity_gap = abs(call_price - put_price)
                    parity_gaps[strike] = {
                        'gap': parity_gap,
                        'call_price': call_price,
                        'put_price': put_price
                    }
            
            if not parity_gaps:
                logger.debug(f"No valid call-put pairs found for expiry {option_expiry}")
                return None
                
            # Find strike with minimal parity gap
            min_strike = min(parity_gaps.keys(), key=lambda k: parity_gaps[k]['gap'])
            min_gap_data = parity_gaps[min_strike]
            
            # Extract synthetic forward price: F = C - P + K
            synthetic_forward = (min_gap_data['call_price'] - 
                               min_gap_data['put_price'] + min_strike)
            
            logger.debug(f"Method A: Strike {min_strike}, Gap {min_gap_data['gap']:.6f}, "
                        f"Forward {synthetic_forward:.2f}")
            
            return synthetic_forward if synthetic_forward > 0 else None
            
        except Exception as e:
            logger.error(f"Error in Method A synthetic future calculation: {e}")
            return None

    def get_synthetic_future_price_method_b(self, option_expiry: datetime, 
                                          spot_price: float, instruments: Dict, 
                                          orderbooks: Dict) -> Optional[float]:
        """
        Method B: Interpolate the futures basis curve
        Calculate annualized basis for all futures and interpolate to option expiry
        """
        try:
            # Collect all available futures data
            futures_data = []
            
            for instrument_name, instrument in instruments.items():
                if (instrument.option_type == 'future' and 
                    instrument_name in orderbooks):
                    
                    orderbook = orderbooks[instrument_name]
                    bid = self._safe_float(orderbook.best_bid)
                    ask = self._safe_float(orderbook.best_ask)
                    
                    if bid <= 0 or ask <= 0 or ask <= bid:
                        continue
                        
                    future_price = (bid + ask) / 2.0
                    time_to_expiry = self._safe_time_to_expiry(instrument.expiration)
                    
                    if time_to_expiry <= 0 or future_price <= 0:
                        continue
                    
                    # Calculate annualized basis: b = ln(F/S) / T
                    basis = np.log(future_price / spot_price) / time_to_expiry
                    
                    futures_data.append({
                        'time_to_expiry': time_to_expiry,
                        'basis': basis,
                        'future_price': future_price,
                        'instrument_name': instrument_name
                    })
            
            if len(futures_data) < 2:
                logger.debug("Need at least 2 futures for basis curve interpolation")
                return None
                
            # Sort by time to expiry
            futures_data.sort(key=lambda x: x['time_to_expiry'])
            
            # Extract time and basis arrays
            times = np.array([f['time_to_expiry'] for f in futures_data])
            bases = np.array([f['basis'] for f in futures_data])
            
            # Calculate target time to expiry for option
            target_time = self._safe_time_to_expiry(option_expiry)
            
            if target_time <= 0:
                return None
            
            # Interpolate basis (linear interpolation, can be changed to cubic spline)
            if target_time <= times[0]:
                # Extrapolate using first two points
                interpolated_basis = bases[0]
            elif target_time >= times[-1]:
                # Extrapolate using last two points
                interpolated_basis = bases[-1]
            else:
                # Interpolate
                interp_func = interp1d(times, bases, kind='linear', 
                                     fill_value='extrapolate')
                interpolated_basis = float(interp_func(target_time))
            
            # Calculate synthetic forward: F = S * exp(b * T)
            synthetic_forward = spot_price * np.exp(interpolated_basis * target_time)
            
            logger.debug(f"Method B: Target time {target_time:.4f}, "
                        f"Interpolated basis {interpolated_basis:.6f}, "
                        f"Forward {synthetic_forward:.2f}")
            
            return synthetic_forward if synthetic_forward > 0 else None
            
        except Exception as e:
            logger.error(f"Error in Method B synthetic future calculation: {e}")
            return None

    def get_future_price_for_expiry(self, expiry: datetime, spot_price: float,
                                  instruments: Dict, orderbooks: Dict, 
                                  method: str = 'A') -> float:
        """
        Get future price for a specific expiry date
        First try to find exact match, then use synthetic methods
        
        method: 'A' for minimal parity gap, 'B' for basis curve interpolation
        """
        try:
            # First, try to find an exact future match
            for instrument_name, instrument in instruments.items():
                if (instrument.option_type == 'future' and 
                    instrument.expiration == expiry and
                    instrument_name in orderbooks):
                    
                    orderbook = orderbooks[instrument_name]
                    bid = self._safe_float(orderbook.best_bid)
                    ask = self._safe_float(orderbook.best_ask)
                    
                    if bid > 0 and ask > 0 and ask > bid:
                        future_price = (bid + ask) / 2.0
                        logger.debug(f"Found exact future match: {future_price:.2f}")
                        return future_price
            
            # No exact match, use synthetic methods
            if method.upper() == 'A':
                synthetic_price = self.get_synthetic_future_price_method_a(
                    expiry, instruments, orderbooks)
            elif method.upper() == 'B':
                synthetic_price = self.get_synthetic_future_price_method_b(
                    expiry, spot_price, instruments, orderbooks)
            else:
                logger.error(f"Unknown method: {method}")
                return spot_price
                
            if synthetic_price is not None and synthetic_price > 0:
                logger.debug(f"Using synthetic future price (Method {method.upper()}): {synthetic_price:.2f}")
                return synthetic_price
            else:
                logger.debug(f"Synthetic method {method.upper()} failed, using spot price")
                return spot_price
                
        except Exception as e:
            logger.error(f"Error getting future price: {e}")
            return spot_price

    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks using proven scipy functions
        
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
            
            # Calculate Greeks using scipy's norm functions
            
            # Delta: ∂V/∂S
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1.0
            
            # Gamma: ∂²V/∂S² (same for calls and puts)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta: -∂V/∂t (time decay)
            theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type.lower() == 'call':
                theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)
            
            # Convert theta to daily (divide by 365)
            theta_daily = theta / 365.0
            
            # Vega: ∂V/∂σ (same for calls and puts)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            # Convert to per 1% volatility change
            vega_percent = vega / 100.0
            
            # Rho: ∂V/∂r
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
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

    def calculate_black_greeks(self, F: float, K: float, T: float, 
                             sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate option Greeks for Black model (r=0)
        
        Returns: Dictionary with delta, gamma, theta, vega
        """
        try:
            # Input validation
            F = self._safe_float(F)
            K = self._safe_float(K)
            T = self._safe_float(T)
            sigma = self._safe_float(sigma, 0.3)
            
            # Default values for invalid inputs
            if F <= 0 or K <= 0 or sigma <= 0 or T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            # Calculate d1 and d2 for Black model
            d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta (with respect to future price)
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1.0
            
            # Gamma (with respect to future price)
            gamma = norm.pdf(d1) / (F * sigma * np.sqrt(T))
            
            # Theta (time decay)
            theta_common = -(F * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            theta = theta_common  # No discounting term in Black model with r=0
            
            # Convert theta to daily
            theta_daily = theta / 365.0
            
            # Vega
            vega = F * norm.pdf(d1) * np.sqrt(T)
            vega_percent = vega / 100.0
            
            # Validate results
            delta = max(-1.0, min(1.0, delta))
            gamma = max(0.0, gamma)
            vega_percent = max(0.0, vega_percent)
            
            return {
                'delta': self._safe_float(delta),
                'gamma': self._safe_float(gamma),
                'theta': self._safe_float(theta_daily),
                'vega': self._safe_float(vega_percent)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Black model Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    def calculate_implied_volatility(self, market_price: float, S: float, K: float, 
                                   T: float, r: float, option_type: str) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method with better bounds
        """
        try:
            # Input validation
            S = self._safe_float(S)
            K = self._safe_float(K)
            T = self._safe_float(T)
            r = self._safe_float(r)
            market_price = self._safe_float(market_price)
            
            if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
                return None
            
            # Calculate intrinsic value for bounds checking
            if option_type.lower() == 'call':
                intrinsic_value = max(S - K, 0)
            else:
                intrinsic_value = max(K - S, 0)
            
            # Market price should be at least intrinsic value
            if market_price < intrinsic_value * 0.9:
                logger.debug(f"Market price {market_price} below intrinsic {intrinsic_value}")
                return None
            
            def objective(vol):
                return self.black_scholes_price(S, K, T, r, vol, option_type) - market_price
            
            # Test bounds to ensure we have a valid bracket
            low_vol = 0.001   # 0.1%
            high_vol = 20.0   # 2000%
            
            # Check if we have opposite signs (required for Brent method)
            low_price = objective(low_vol)
            high_price = objective(high_vol)
            
            if low_price * high_price > 0:
                # Same sign, try different bounds
                if low_price > 0:  # Both positive, try lower vol
                    low_vol = 0.0001
                    low_price = objective(low_vol)
                else:  # Both negative, try higher vol
                    high_vol = 30.0
                    high_price = objective(high_vol)
            
            # Use Brent's method
            iv = brentq(objective, low_vol, high_vol, xtol=1e-6, maxiter=1000)
            
            # Validate result
            if 0.0001 <= iv <= 20.0:
                return iv
            else:
                logger.debug(f"IV out of reasonable range: {iv}")
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating IV: {e}")
            return None

    def calculate_black_implied_volatility(self, market_price: float, F: float, K: float, 
                                         T: float, option_type: str) -> Optional[float]:
        """
        Calculate implied volatility using Black model (r=0)
        """
        try:
            # Input validation
            F = self._safe_float(F)
            K = self._safe_float(K)
            T = self._safe_float(T)
            market_price = self._safe_float(market_price)
            
            if T <= 0 or market_price <= 0 or F <= 0 or K <= 0:
                return None
            
            # Calculate intrinsic value for bounds checking
            if option_type.lower() == 'call':
                intrinsic_value = max(F - K, 0)
            else:
                intrinsic_value = max(K - F, 0)
            
            # Market price should be at least intrinsic value
            if market_price < intrinsic_value * 0.9:
                logger.debug(f"Market price {market_price} below intrinsic {intrinsic_value}")
                return None
            
            def objective(vol):
                return self.black_model_price(F, K, T, vol, option_type) - market_price
            
            # Test bounds
            low_vol = 0.001   # 0.1%
            high_vol = 20.0   # 2000%
            
            # Check if we have opposite signs
            low_price = objective(low_vol)
            high_price = objective(high_vol)
            
            if low_price * high_price > 0:
                if low_price > 0:
                    low_vol = 0.0001
                    low_price = objective(low_vol)
                else:
                    high_vol = 30.0
                    high_price = objective(high_vol)
            
            # Use Brent's method
            iv = brentq(objective, low_vol, high_vol, xtol=1e-6, maxiter=1000)
            
            # Validate result
            if 0.0001 <= iv <= 20.0:
                return iv
            else:
                logger.debug(f"Black IV out of reasonable range: {iv}")
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating Black IV: {e}")
            return None

    def calculate_all(self, synthetic_method: str = 'A', use_black_model: bool = True) -> Dict:
        """
        Calculate pricing and risk metrics for all instruments
        
        synthetic_method: 'A' for minimal parity gap, 'B' for basis curve interpolation
        use_black_model: True to use Black model with synthetic futures, False for original BS
        """
        try:
            # Get market data
            display_data = self.data_manager.get_display_data()
            spot_price = self._safe_float(display_data.get('spot_price', 0))
            
            if spot_price <= 0:
                logger.debug("No valid spot price available")
                return {}
                
            instruments = display_data.get('instruments', {})
            orderbooks = display_data.get('orderbooks', {})
            
            # Update synthetic futures cache if needed (max once per minute)
            if use_black_model:
                self.update_synthetic_futures_cache_if_needed(
                    instruments, orderbooks, spot_price, synthetic_method
                )
            
            results = {}
            
            logger.info(f"Processing {len(instruments)} instruments with spot price ${spot_price:,.2f}")
            logger.info(f"Using {'Black model' if use_black_model else 'Black-Scholes model'} "
                       f"with synthetic method {synthetic_method}")
            
            # Show cache status
            if use_black_model:
                cache_status = self.data_manager.get_synthetic_futures_cache_status()
                logger.info(f"Synthetic futures cache: {cache_status['cache_size']} entries, "
                           f"{'fresh' if not cache_status['is_stale'] else 'stale'}")
            
            for instrument_name, instrument in instruments.items():
                try:
                    # Skip futures for now
                    if instrument.option_type == 'future':
                        continue
                        
                    # Get market data for this instrument
                    orderbook = orderbooks.get(instrument_name)
                    if not orderbook:
                        continue
                        
                    # Convert BTC prices to USD
                    best_bid_btc = self._safe_float(orderbook.best_bid)
                    best_ask_btc = self._safe_float(orderbook.best_ask)
                    best_bid_usd = best_bid_btc * spot_price
                    best_ask_usd = best_ask_btc * spot_price
                    
                    # Basic validation
                    if best_bid_usd <= 0 or best_ask_usd <= 0 or best_ask_usd <= best_bid_usd:
                        continue
                    
                    market_price_usd = (best_bid_usd + best_ask_usd) / 2.0
                    
                    # Get instrument parameters
                    strike = self._safe_float(instrument.strike)
                    time_to_expiry = self._safe_time_to_expiry(instrument.expiration)
                    option_type = instrument.option_type.lower()
                    
                    if strike <= 0 or time_to_expiry <= 0:
                        continue
                    
                    if use_black_model:
                        # Get future price using cache
                        future_price, was_cached = self.get_future_price_for_expiry_cached(
                            instrument.expiration, spot_price, instruments, 
                            orderbooks, synthetic_method)
                        
                        # Calculate using Black model
                        if option_type == 'call':
                            intrinsic_value_usd = max(future_price - strike, 0)
                        else:
                            intrinsic_value_usd = max(strike - future_price, 0)
                        
                        time_value_usd = max(market_price_usd - intrinsic_value_usd, 0)
                        
                        # Calculate implied volatility using Black model
                        implied_vol = self.calculate_black_implied_volatility(
                            market_price_usd, future_price, strike, 
                            time_to_expiry, option_type
                        )
                        
                        if implied_vol is None:
                            continue
                        
                        # Calculate theoretical price and Greeks
                        theoretical_price_usd = self.black_model_price(
                            future_price, strike, time_to_expiry, implied_vol, option_type
                        )
                        
                        greeks = self.calculate_black_greeks(
                            future_price, strike, time_to_expiry, implied_vol, option_type
                        )
                        
                        reference_price = future_price
                        model_used = 'Black'
                        
                    else:
                        # Use original Black-Scholes model (unchanged)
                        if option_type == 'call':
                            intrinsic_value_usd = max(spot_price - strike, 0)
                        else:
                            intrinsic_value_usd = max(strike - spot_price, 0)
                        
                        time_value_usd = max(market_price_usd - intrinsic_value_usd, 0)
                        
                        implied_vol = self.calculate_implied_volatility(
                            market_price_usd, spot_price, strike, 
                            time_to_expiry, self.risk_free_rate, option_type
                        )
                        
                        if implied_vol is None:
                            continue
                        
                        theoretical_price_usd = self.black_scholes_price(
                            spot_price, strike, time_to_expiry,
                            self.risk_free_rate, implied_vol, option_type
                        )
                        
                        greeks = self.calculate_greeks(
                            spot_price, strike, time_to_expiry,
                            self.risk_free_rate, implied_vol, option_type
                        )
                        
                        reference_price = spot_price
                        model_used = 'Black-Scholes'
                        was_cached = False
                    
                    # Store results
                    results[instrument_name] = {
                        'spot_price': spot_price,
                        'reference_price': reference_price,
                        'strike': strike,
                        'time_to_expiry': time_to_expiry,
                        'option_type': option_type,
                        'model_used': model_used,
                        'synthetic_method': synthetic_method if use_black_model else None,
                        'was_cached': was_cached,  # NEW: Track if future price was cached
                        'market_price': market_price_usd,
                        'market_price_btc': (best_bid_btc + best_ask_btc) / 2.0,
                        'bid': best_bid_usd,
                        'ask': best_ask_usd,
                        'bid_btc': best_bid_btc,
                        'ask_btc': best_ask_btc,
                        'theoretical_price': theoretical_price_usd,
                        'implied_volatility': implied_vol,
                        'price_diff': market_price_usd - theoretical_price_usd,
                        'intrinsic_value': intrinsic_value_usd,
                        'time_value': time_value_usd,
                        **greeks,
                        'timestamp': datetime.now()
                    }
                    
                    # Log first few successful calculations with cache info
                    if len(results) <= 5:
                        cache_info = "🔄" if was_cached else "⚡"
                        logger.info(f"✅ {instrument_name}: ${market_price_usd:.2f}, "
                                  f"IV={implied_vol:.1%}, Δ={greeks['delta']:.3f}, "
                                  f"Ref=${reference_price:.2f} ({model_used}) {cache_info}")
                        
                except Exception as e:
                    logger.debug(f"Error processing {instrument_name}: {e}")
                    continue
            
            # Summary statistics
            calls = sum(1 for r in results.values() if r['option_type'] == 'call')
            puts = sum(1 for r in results.values() if r['option_type'] == 'put')
            cached_count = sum(1 for r in results.values() if r.get('was_cached', False))
            
            logger.info(f"Successfully calculated pricing for {len(results)} instruments "
                       f"({calls} calls, {puts} puts). "
                       f"Used cache for {cached_count} future prices.")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in calculate_all: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_future_price_for_expiry_cached(self, expiry: datetime, spot_price: float,
                                         instruments: Dict, orderbooks: Dict, 
                                         method: str = 'A') -> Tuple[float, bool]:
        """
        Get future price for a specific expiry date with caching
        Returns (future_price, was_cached)
        
        First checks cache, then tries exact match, then uses synthetic methods
        """
        try:
            # Check cache first
            cached_price = self.data_manager.get_cached_synthetic_future_price(expiry, method)
            if cached_price is not None:
                logger.debug(f"Using cached synthetic future price: {cached_price:.2f} (Method {method})")
                return cached_price, True
            
            # Cache miss or stale - need to calculate
            logger.debug(f"Cache miss for expiry {expiry.strftime('%Y-%m-%d')}, calculating...")
            
            # First, try to find an exact future match
            for instrument_name, instrument in instruments.items():
                if (instrument.option_type == 'future' and 
                    instrument.expiration.date() == expiry.date() and  # Compare dates only
                    instrument_name in orderbooks):
                    
                    orderbook = orderbooks[instrument_name]
                    bid = self._safe_float(orderbook.best_bid)
                    ask = self._safe_float(orderbook.best_ask)
                    
                    if bid > 0 and ask > 0 and ask > bid:
                        future_price = (bid + ask) / 2.0
                        
                        # Store in cache
                        self.data_manager.store_synthetic_future_price(
                            expiry, method, future_price, 
                            {'type': 'exact_match', 'instrument': instrument_name}
                        )
                        
                        logger.debug(f"Found exact future match: {future_price:.2f}")
                        return future_price, False
            
            # No exact match, use synthetic methods
            synthetic_price = None
            calculation_details = {}
            
            if method.upper() == 'A':
                synthetic_price = self.get_synthetic_future_price_method_a(
                    expiry, instruments, orderbooks)
                calculation_details['type'] = 'method_a_parity_gap'
                
            elif method.upper() == 'B':
                synthetic_price = self.get_synthetic_future_price_method_b(
                    expiry, spot_price, instruments, orderbooks)
                calculation_details['type'] = 'method_b_basis_curve'
                
            else:
                logger.error(f"Unknown synthetic method: {method}")
                return spot_price, False
            
            if synthetic_price is not None and synthetic_price > 0:
                # Store successful calculation in cache
                self.data_manager.store_synthetic_future_price(
                    expiry, method, synthetic_price, calculation_details
                )
                
                logger.debug(f"Calculated synthetic future price (Method {method.upper()}): {synthetic_price:.2f}")
                return synthetic_price, False
            else:
                # Calculation failed, store spot price as fallback
                self.data_manager.store_synthetic_future_price(
                    expiry, method, spot_price, 
                    {'type': 'fallback_spot', 'reason': 'synthetic_calculation_failed'}
                )
                
                logger.debug(f"Synthetic method {method.upper()} failed, using spot price: {spot_price:.2f}")
                return spot_price, False
                
        except Exception as e:
            logger.error(f"Error getting cached future price: {e}")
            return spot_price, False

    def update_synthetic_futures_cache_if_needed(self, instruments: Dict, orderbooks: Dict, 
                                               spot_price: float, method: str = 'A'):
        """
        Update the entire synthetic futures cache if it's stale
        This runs once per minute maximum
        """
        try:
            if not self.data_manager.is_synthetic_futures_cache_stale():
                return  # Cache is still fresh
            
            logger.info(f"Updating synthetic futures cache (Method {method})...")
            
            # Clear old cache entries
            self.data_manager.clear_stale_synthetic_futures_cache()
            
            # Get all unique option expiry dates
            option_expiries = set()
            for instrument in instruments.values():
                if instrument.option_type in ['call', 'put']:
                    option_expiries.add(instrument.expiration.date())
            
            successful_calculations = 0
            
            # Calculate synthetic future for each expiry
            for expiry_date in option_expiries:
                try:
                    # Convert back to datetime for calculations
                    expiry_datetime = datetime.combine(expiry_date, datetime.min.time())
                    
                    # Force recalculation by not checking cache
                    future_price, _ = self._calculate_synthetic_future_direct(
                        expiry_datetime, spot_price, instruments, orderbooks, method
                    )
                    
                    if future_price > 0:
                        successful_calculations += 1
                        
                except Exception as e:
                    logger.debug(f"Error calculating synthetic future for {expiry_date}: {e}")
                    continue
            
            # Mark cache as updated
            self.data_manager.update_synthetic_futures_cache_timestamp()
            
            logger.info(f"Updated synthetic futures cache: {successful_calculations}/{len(option_expiries)} successful")
            
        except Exception as e:
            logger.error(f"Error updating synthetic futures cache: {e}")

    def _calculate_synthetic_future_direct(self, expiry: datetime, spot_price: float,
                                         instruments: Dict, orderbooks: Dict, 
                                         method: str) -> Tuple[float, Dict]:
        """
        Calculate synthetic future price directly without checking cache
        Returns (future_price, calculation_details)
        """
        try:
            # First, try exact future match
            for instrument_name, instrument in instruments.items():
                if (instrument.option_type == 'future' and 
                    instrument.expiration.date() == expiry.date() and
                    instrument_name in orderbooks):
                    
                    orderbook = orderbooks[instrument_name]
                    bid = self._safe_float(orderbook.best_bid)
                    ask = self._safe_float(orderbook.best_ask)
                    
                    if bid > 0 and ask > 0 and ask > bid:
                        future_price = (bid + ask) / 2.0
                        details = {'type': 'exact_match', 'instrument': instrument_name}
                        
                        # Store in cache
                        self.data_manager.store_synthetic_future_price(expiry, method, future_price, details)
                        return future_price, details
            
            # No exact match, use synthetic methods
            details = {}
            
            if method.upper() == 'A':
                synthetic_price = self.get_synthetic_future_price_method_a(expiry, instruments, orderbooks)
                details = {'type': 'method_a_parity_gap'}
                
            elif method.upper() == 'B':
                synthetic_price = self.get_synthetic_future_price_method_b(expiry, spot_price, instruments, orderbooks)
                details = {'type': 'method_b_basis_curve'}
                
            else:
                synthetic_price = None
                details = {'type': 'error', 'reason': f'unknown_method_{method}'}
            
            if synthetic_price is not None and synthetic_price > 0:
                # Store successful calculation
                self.data_manager.store_synthetic_future_price(expiry, method, synthetic_price, details)
                return synthetic_price, details
            else:
                # Use spot price as fallback
                details = {'type': 'fallback_spot', 'reason': 'synthetic_calculation_failed'}
                self.data_manager.store_synthetic_future_price(expiry, method, spot_price, details)
                return spot_price, details
                
        except Exception as e:
            logger.error(f"Error in direct synthetic future calculation: {e}")
            details = {'type': 'error', 'reason': str(e)}
            return spot_price, details
    
    def get_cached_results(self) -> Dict:
        """Get cached pricing results from data manager"""
        return self.data_manager.get_latest_pricing_results()