# data_manager.py (update the OrderBook class and update_orderbook method)
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBook:
    bids: List[List[float]] = field(default_factory=list)
    asks: List[List[float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
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
    
    @property
    def best_bid(self) -> float:
        if self.bids and len(self.bids) > 0 and len(self.bids[0]) > 0:
            return self._safe_float(self.bids[0][0])
        return 0.0
        
    @property
    def best_ask(self) -> float:
        if self.asks and len(self.asks) > 0 and len(self.asks[0]) > 0:
            return self._safe_float(self.asks[0][0])
        return 0.0
        
    @property
    def mid_price(self) -> float:
        bid = self.best_bid
        ask = self.best_ask
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return 0.0

@dataclass
class Instrument:
    instrument_name: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    contract_size: float
    tick_size: float
    min_trade_amount: float

class DataManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._instruments: Dict[str, Instrument] = {}
        self._orderbooks: Dict[str, OrderBook] = {}
        self._persistent_orderbooks: Dict[str, OrderBook] = {}  # Never gets cleared
        self._persistent_pricing: Dict[str, Dict] = {}          # Never gets cleared
        self._last_seen_data: Dict[str, datetime] = {}          # Track data age
        self._spot_price: float = 0.0
        self._last_update: datetime = datetime.now()
        self._update_count: int = 0
        self._latest_pricing_results: Dict = {}
        self._last_pricing_update: datetime = datetime.now()
        
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
        
    def update_spot_price(self, price: float):
        """Update BTC spot price"""
        with self._lock:
            self._spot_price = self._safe_float(price)
            self._last_update = datetime.now()
            self._update_count += 1
            
    def get_spot_price(self) -> float:
        """Get current BTC spot price"""
        with self._lock:
            return self._spot_price
            
    def update_instruments(self, instruments_data: List[Dict]):
        """Update instruments list"""
        logger.info(f"Updating {len(instruments_data)} instruments")
        
        with self._lock:
            for i, data in enumerate(instruments_data):
                try:
                    # Safely extract and convert data
                    strike = self._safe_float(data.get('strike', 0.0))
                    contract_size = self._safe_float(data.get('contract_size', 1.0))
                    tick_size = self._safe_float(data.get('tick_size', 0.0001))
                    min_trade_amount = self._safe_float(data.get('min_trade_amount', 0.1))
                    
                    instrument = Instrument(
                        instrument_name=str(data.get('instrument_name', '')),
                        strike=strike,
                        expiration=self._parse_expiration(data.get('expiration_timestamp')),
                        option_type=str(data.get('option_type', '')).lower(),
                        contract_size=contract_size,
                        tick_size=tick_size,
                        min_trade_amount=min_trade_amount
                    )
                    self._instruments[instrument.instrument_name] = instrument
                    
                    # Debug: Print first few instruments
                    if i < 3:
                        logger.info(f"Added instrument: {instrument.instrument_name}, Strike: {instrument.strike}, Type: {instrument.option_type}")
                        
                except Exception as e:
                    logger.error(f"Error processing instrument {i}: {e}")
                    continue
                    
    def get_instruments(self) -> List[Dict]:
        """Get all instruments as dict format"""
        with self._lock:
            return [
                {
                    'instrument_name': inst.instrument_name,
                    'strike': inst.strike,
                    'expiration_timestamp': inst.expiration.timestamp(),
                    'option_type': inst.option_type
                }
                for inst in self._instruments.values()
            ]
            
    def update_orderbook(self, params: Dict):
        """Update order book data with persistence and safe type handling"""
        try:
            channel = params.get('channel', '')
            instrument_name = channel.split('.')[1] if '.' in channel else ''
            data = params.get('data', {})
            
            with self._lock:
                # Safely process bids and asks
                raw_bids = data.get('bids', [])
                raw_asks = data.get('asks', [])
                
                # Convert to safe float format
                safe_bids = []
                for bid in raw_bids:
                    if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                        price = self._safe_float(bid[0])
                        size = self._safe_float(bid[1])
                        if price > 0:  # Only include valid bids
                            safe_bids.append([price, size])
                
                safe_asks = []
                for ask in raw_asks:
                    if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                        price = self._safe_float(ask[0])
                        size = self._safe_float(ask[1])
                        if price > 0:  # Only include valid asks
                            safe_asks.append([price, size])
                
                orderbook = OrderBook(
                    bids=safe_bids,
                    asks=safe_asks,
                    timestamp=datetime.now()
                )
                
                # Update current orderbooks (for real-time processing)
                self._orderbooks[instrument_name] = orderbook
                
                # Update persistent storage (never cleared, only updated when we have good data)
                best_bid = orderbook.best_bid  # This now safely returns float
                best_ask = orderbook.best_ask  # This now safely returns float
                
                if best_bid > 0 or best_ask > 0:
                    self._persistent_orderbooks[instrument_name] = orderbook
                    self._last_seen_data[instrument_name] = datetime.now()
                
                self._update_count += 1
                
        except Exception as e:
            logger.error(f"Error updating orderbook for {instrument_name}: {e}")
            import traceback
            traceback.print_exc()
            
    def get_orderbook(self, instrument_name: str) -> Optional[OrderBook]:
        """Get order book for specific instrument"""
        with self._lock:
            return self._orderbooks.get(instrument_name)
            
    def get_all_market_data(self) -> Dict:
        """Get all current market data (for compatibility)"""
        with self._lock:
            return {
                'spot_price': self._spot_price,
                'instruments': dict(self._instruments),
                'orderbooks': dict(self._orderbooks),
                'last_update': self._last_update
            }

    def get_display_data(self) -> Dict:
        """Get data for GUI display (uses persistent storage)"""
        with self._lock:
            return {
                'spot_price': self._spot_price,
                'instruments': dict(self._instruments),
                'orderbooks': dict(self._persistent_orderbooks),  # Use persistent data
                'pricing': dict(self._persistent_pricing),        # Use persistent pricing
                'last_update': self._last_update
            }

    def get_data_age(self, instrument_name: str) -> float:
        """Get age of data in seconds"""
        with self._lock:
            last_seen = self._last_seen_data.get(instrument_name)
            if last_seen:
                return (datetime.now() - last_seen).total_seconds()
            return float('inf')

    def store_pricing_results(self, pricing_results: Dict):
        """Store pricing results with persistence"""
        with self._lock:
            self._latest_pricing_results = pricing_results
            self._last_pricing_update = datetime.now()
            
            # Update persistent pricing (keep all good results)
            for instrument, result in pricing_results.items():
                try:
                    market_price = self._safe_float(result.get('market_price', 0))
                    iv = self._safe_float(result.get('implied_volatility', 0))
                    
                    if market_price > 0 and 0 < iv < 5:  # Filter out extreme IVs
                        self._persistent_pricing[instrument] = result
                except Exception as e:
                    logger.debug(f"Error storing pricing for {instrument}: {e}")
                    continue

    def get_latest_pricing_results(self) -> Dict:
        """Get the latest stored pricing results"""
        with self._lock:
            return dict(self._latest_pricing_results)

    def get_last_pricing_update(self) -> datetime:
        """Get timestamp of last pricing update"""
        with self._lock:
            return self._last_pricing_update
            
    def get_update_count(self) -> int:
        """Get total number of updates received"""
        with self._lock:
            return self._update_count
            
    def _parse_expiration(self, timestamp) -> datetime:
        """Parse expiration timestamp"""
        try:
            if timestamp:
                # Handle both string and numeric timestamps
                if isinstance(timestamp, str):
                    timestamp_float = float(timestamp)
                else:
                    timestamp_float = float(timestamp)
                
                # Deribit timestamps are in milliseconds
                return datetime.fromtimestamp(timestamp_float / 1000)
            return datetime.now()
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse expiration timestamp {timestamp}: {e}")
            return datetime.now()