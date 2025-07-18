# data_manager.py
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd

@dataclass
class OrderBook:
    bids: List[List[float]] = field(default_factory=list)
    asks: List[List[float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
        
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0
        
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2
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
        self._spot_price: float = 0.0
        self._last_update: datetime = datetime.now()
        self._update_count = 0  # Add this line
        self._persistent_orderbooks = {}  # Never gets cleared, only updated
        self._persistent_pricing = {}     # Never gets cleared, only updated
        self._last_seen_data = {}        # Track when we last saw data for each instrument
        
    def update_spot_price(self, price: float):
        """Update BTC spot price"""
        with self._lock:
            self._spot_price = price
            self._last_update = datetime.now()
            self._update_count += 1  # Add this line
            
    def get_spot_price(self) -> float:
        """Get current BTC spot price"""
        with self._lock:
            return self._spot_price
            
    def update_instruments(self, instruments_data: List[Dict]):
        """Update instruments list"""
        #logger.info(f"Updating {len(instruments_data)} instruments")
        
        with self._lock:
            for i, data in enumerate(instruments_data):
                try:
                    # Safely extract and convert data
                    strike = float(data.get('strike', 0.0)) if data.get('strike') else 0.0
                    contract_size = float(data.get('contract_size', 1.0)) if data.get('contract_size') else 1.0
                    tick_size = float(data.get('tick_size', 0.0001)) if data.get('tick_size') else 0.0001
                    min_trade_amount = float(data.get('min_trade_amount', 0.1)) if data.get('min_trade_amount') else 0.1
                    
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
                    #if i < 3:
                        #logger.info(f"Added instrument: {instrument.instrument_name}, Strike: {instrument.strike}, Type: {instrument.option_type}")
                        
                except Exception as e:
                    #logger.error(f"Error processing instrument {i}: {e}")
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
        """Update order book data"""
        try:
            channel = params.get('channel', '')
            instrument_name = channel.split('.')[1] if '.' in channel else ''
            data = params.get('data', {})
            
            with self._lock:
                orderbook = OrderBook(
                    bids=data.get('bids', []),
                    asks=data.get('asks', []),
                    timestamp=datetime.now()
                )
                self._orderbooks[instrument_name] = orderbook
                self._update_count += 1  # Add this line
                
        except Exception as e:
            print(f"Error updating orderbook: {e}")

    def get_update_count(self) -> int:  # Add this method
        """Get total number of updates received"""
        with self._lock:
            return self._update_count

    def get_orderbook(self, instrument_name: str) -> Optional[OrderBook]:
        """Get order book for specific instrument"""
        with self._lock:
            return self._orderbooks.get(instrument_name)
            
    def get_all_market_data(self) -> Dict:
        """Get all current market data"""
        with self._lock:
            return {
                'spot_price': self._spot_price,
                'instruments': dict(self._instruments),
                'orderbooks': dict(self._orderbooks),
                'last_update': self._last_update
            }
            
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
            #logger.warning(f"Could not parse expiration timestamp {timestamp}: {e}")
            return datetime.now()
        
    # Add this method to your DataManager class in data_manager.py
    def store_pricing_results(self, pricing_results: Dict):
        """Store the latest pricing results"""
        with self._lock:
            self._latest_pricing_results = pricing_results
            self._last_pricing_update = datetime.now()

    def get_latest_pricing_results(self) -> Dict:
        """Get the latest stored pricing results"""
        with self._lock:
            return getattr(self, '_latest_pricing_results', {})

    def get_last_pricing_update(self) -> datetime:
        """Get timestamp of last pricing update"""
        with self._lock:
            return getattr(self, '_last_pricing_update', datetime.now())
        