# deribit_client.py
import websockets
import json
import asyncio
import logging
import ssl
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DeribitClient:
    def __init__(self, data_manager):
        self.ws_url = "wss://www.deribit.com/ws/api/v2"
        self.websocket = None
        self.data_manager = data_manager
        self.request_id = 1
        self.pending_requests = {}  # Track pending requests
        self.max_subscriptions = 50  # Increase limit
        
    async def connect(self):
        """Connect to Deribit WebSocket"""
        try:
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                self.ws_url,
                ssl=ssl_context
            )
            logger.info("Connected to Deribit WebSocket")
            
            # Start message handler
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
            
    async def _send_request(self, method: str, params: Dict = None) -> Dict:
        """Send request to Deribit API and wait for response"""
        if not self.websocket:
            raise Exception("Not connected to WebSocket")
            
        request_id = self.request_id
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        
        if params:
            request["params"] = params
            
        # Create future to wait for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        await self.websocket.send(json.dumps(request))
        self.request_id += 1
        
        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=10.0)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            return {}

    async def _get_real_btc_spot_price(self) -> float:
        """Get real-time BTC spot price from Deribit"""
        try:
            # Try to get BTC index price
            response = await self._send_request("public/get_index", {
                "currency": "BTC"
            })
            
            if "result" in response:
                btc_data = response["result"]
                spot_price = float(btc_data.get("BTC", 0))
                if spot_price > 0:
                    logger.info(f"Real BTC spot price: ${spot_price:,.2f}")
                    self.data_manager.update_spot_price(spot_price)
                    return spot_price
            
            # Fallback: try perpetual future
            response2 = await self._send_request("public/get_book_summary_by_currency", {
                "currency": "BTC",
                "kind": "future"
            })
            
            if "result" in response2 and response2["result"]:
                futures = response2["result"]
                for future in futures:
                    if "PERPETUAL" in future.get("instrument_name", ""):
                        mark_price = float(future.get("mark_price", 0))
                        if mark_price > 0:
                            logger.info(f"BTC price from perpetual: ${mark_price:,.2f}")
                            self.data_manager.update_spot_price(mark_price)
                            return mark_price
            
            logger.warning("Could not get real BTC price, using fallback")
            return 50000  # Fallback
            
        except Exception as e:
            logger.error(f"Error getting BTC spot price: {e}")
            return 50000

    def _select_best_expiries(self, instruments: List[Dict]) -> List[Dict]:
        """Select best expiries: some short-term, some long-term"""
        try:
            # Group by expiry
            by_expiry = {}
            for inst in instruments:
                expiry = inst.get('expiration_timestamp')
                if expiry not in by_expiry:
                    by_expiry[expiry] = {
                        'instruments': [],
                        'days_to_expiry': inst['days_to_expiry'],
                        'expiry_timestamp': expiry
                    }
                by_expiry[expiry]['instruments'].append(inst)
                
            # Convert to list and sort by days to expiry
            expiry_list = list(by_expiry.values())
            expiry_list.sort(key=lambda x: x['days_to_expiry'])
            
            # Add readable names
            for exp in expiry_list:
                exp_date = datetime.fromtimestamp(exp['expiry_timestamp'] / 1000)
                exp['name'] = exp_date.strftime('%d%b%y').upper()
                
            logger.info(f"Available expiries: {len(expiry_list)} total")
            
            # Selection strategy
            selected = []
            
            # 1. Short-term: 1-2 expiries within 30 days
            short_term = [exp for exp in expiry_list if exp['days_to_expiry'] <= 30]
            if short_term:
                selected.extend(short_term[:3])
                
            # 2. Medium-term: 1-2 expiries between 30-90 days
            medium_term = [exp for exp in expiry_list if 30 < exp['days_to_expiry'] <= 90]
            if medium_term:
                selected.extend(medium_term[:3])
                
            # 3. Long-term: 1 expiry beyond 90 days
            long_term = [exp for exp in expiry_list if exp['days_to_expiry'] > 90]
            if long_term:
                selected.extend(long_term[:3])
                
            # If we don't have enough, fill with closest expiries
            if len(selected) < 6:
                for exp in expiry_list:
                    if exp not in selected and len(selected) < 9:
                        selected.append(exp)
                        
            return selected[:6]  # Maximum 5 expiries
            
        except Exception as e:
            logger.error(f"Error selecting expiries: {e}")
            return list(by_expiry.values())[:3]

    def _select_strikes_for_expiry(self, expiry_instruments: List[Dict], spot_price: float, strikes_count: int = 5) -> List[Dict]:
        """Select strikes around ATM for a specific expiry"""
        try:
            # Group by strike
            by_strike = {}
            for inst in expiry_instruments:
                strike = float(inst.get('strike', 0))
                if strike not in by_strike:
                    by_strike[strike] = {'call': None, 'put': None, 'distance': abs(strike - spot_price)}
                    
                option_type = inst.get('option_type', '').lower()
                if option_type in ['call', 'put']:
                    by_strike[strike][option_type] = inst
                    
            # Sort strikes by distance from spot
            strike_list = list(by_strike.keys())
            strike_list.sort(key=lambda s: abs(s - spot_price))
            
            # Select strikes around ATM
            selected_strikes = strike_list[:strikes_count//2*2]  # Ensure even number
            
            # Collect both calls and puts for selected strikes
            selected_instruments = []
            for strike in selected_strikes:
                strike_data = by_strike[strike]
                if strike_data['call']:
                    selected_instruments.append(strike_data['call'])
                if strike_data['put']:
                    selected_instruments.append(strike_data['put'])
                    
            return selected_instruments
            
        except Exception as e:
            logger.error(f"Error selecting strikes: {e}")
            return expiry_instruments[:20]  # Fallback

    def _select_best_instruments(self, all_instruments: List[Dict], spot_price: float) -> List[Dict]:
        """Select the most relevant instruments with 10 strikes per expiry"""
        try:
            current_time = datetime.now().timestamp() * 1000
            
            # Filter valid instruments
            valid_instruments = []
            
            for inst in all_instruments:
                try:
                    # Skip expired instruments
                    expiry = inst.get('expiration_timestamp', 0)
                    if expiry <= current_time:
                        continue
                        
                    # Calculate time to expiry in days
                    time_to_expiry = (expiry - current_time) / (1000 * 60 * 60 * 24)
                    
                    # Skip instruments too close to expiry (less than 2 days)
                    if time_to_expiry < 2:
                        continue
                        
                    # Skip instruments too far out (more than 180 days)
                    if time_to_expiry > 180:
                        continue
                        
                    # Add calculated fields
                    inst['days_to_expiry'] = time_to_expiry
                    inst['strike_distance'] = abs(float(inst.get('strike', 0)) - spot_price)
                    
                    valid_instruments.append(inst)
                    
                except Exception as e:
                    continue
                    
            logger.info(f"Filtered to {len(valid_instruments)} valid instruments")
            
            if not valid_instruments:
                return []
                
            # Select best expiries
            selected_expiries = self._select_best_expiries(valid_instruments)
            logger.info(f"Selected {len(selected_expiries)} expiries")
            
            # For each selected expiry, get 10 strikes around ATM
            selected_instruments = []
            
            for expiry_info in selected_expiries:
                expiry_instruments = expiry_info['instruments']
                strikes_for_expiry = self._select_strikes_for_expiry(expiry_instruments, spot_price, strikes_count=10)
                selected_instruments.extend(strikes_for_expiry)
                
            logger.info(f"Total selected instruments: {len(selected_instruments)}")
            return selected_instruments[:self.max_subscriptions]
            
        except Exception as e:
            logger.error(f"Error selecting instruments: {e}")
            return all_instruments[:self.max_subscriptions]
        
    async def get_btc_instruments(self) -> List[Dict]:
        """Get all BTC instruments"""
        logger.info("Requesting BTC instruments...")
        
        # First get the real BTC spot price
        spot_price = await self._get_real_btc_spot_price()
        
        response = await self._send_request("public/get_instruments", {
            "currency": "BTC",
            "kind": "option"
        })
        
        if "result" in response:
            all_instruments = response["result"]
            logger.info(f"Received {len(all_instruments)} total BTC instruments")
            
            # Filter and select the best instruments using real spot price
            selected_instruments = self._select_best_instruments(all_instruments, spot_price)
            logger.info(f"Selected {len(selected_instruments)} instruments for trading")
            
            self.data_manager.update_instruments(selected_instruments)
            return selected_instruments
        else:
            logger.error("Failed to get instruments")
            return []

    async def _get_initial_orderbooks(self, instruments: List[Dict]):
        """Get initial order book snapshots for all instruments"""
        logger.info("Getting initial order book snapshots...")
        
        successful_snapshots = 0
        
        for i, instrument in enumerate(instruments):
            try:
                instrument_name = instrument.get('instrument_name')
                if instrument_name:
                    response = await self._send_request("public/get_order_book", {
                        "instrument_name": instrument_name
                    })
                    
                    if "result" in response:
                        orderbook_data = response["result"]
                        bids = orderbook_data.get('bids', [])
                        asks = orderbook_data.get('asks', [])
                        
                        # Only process if we have actual bid/ask data
                        if bids or asks:
                            self.data_manager.update_orderbook({
                                'channel': f'book.{instrument_name}',
                                'data': {
                                    'bids': bids,
                                    'asks': asks
                                }
                            })
                            successful_snapshots += 1
                            
                            # Log first few successful ones
                            if successful_snapshots <= 3:
                                best_bid = bids[0][0] if bids else 0
                                best_ask = asks[0][0] if asks else 0
                                logger.info(f"Got orderbook for {instrument_name}: bid={best_bid}, ask={best_ask}")
                        
            except Exception as e:
                logger.debug(f"Error getting orderbook for {instrument.get('instrument_name', 'unknown')}: {e}")
                continue
                
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.05)  # Reduced delay
            
            # Stop after trying 20 instruments to speed up startup
            if i >= 19:
                break
        
        logger.info(f"Successfully got {successful_snapshots} initial orderbooks")
        
    async def subscribe_market_data(self, instruments: List[Dict]):
        """Subscribe to order book data for instruments"""
        if not instruments:
            logger.warning("No instruments provided for subscription")
            return
            
        channels = []
        
        # Subscribe to BTC index (spot price)
        channels.append("deribit_price_index.btc_usd")
        
        # Subscribe to order books for selected instruments
        for instrument in instruments:
            instrument_name = instrument.get('instrument_name')
            if instrument_name:
                channels.append(f"book.{instrument_name}.100ms")
                
        logger.info(f"Subscribing to {len(channels)} channels (including BTC index)")
        
        if len(channels) > 1:  # More than just BTC index
            response = await self._send_request("public/subscribe", {
                "channels": channels
            })
            
            if "result" in response:
                logger.info(f"Successfully subscribed to {len(channels)} channels")
                
                # Also get initial order book snapshots
                await self._get_initial_orderbooks(instruments)
            else:
                logger.error("Subscription failed")
        
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._process_message(data)
                
        except Exception as e:
            logger.error(f"Error handling messages: {e}")
            
    async def _process_message(self, data: Dict):
        """Process individual messages"""
        try:
            # Handle API responses
            if "id" in data and data["id"] in self.pending_requests:
                future = self.pending_requests.pop(data["id"])
                if not future.done():
                    future.set_result(data)
                return
                
            # Handle subscription notifications
            if "method" in data and data["method"] == "subscription":
                params = data.get("params", {})
                channel = params.get("channel", "")
                
                if channel.startswith("book."):
                    # Order book update
                    self.data_manager.update_orderbook(params)
                elif channel == "deribit_price_index.btc_usd":
                    # BTC spot price update
                    price_data = params.get("data", {})
                    if "price" in price_data:
                        self.data_manager.update_spot_price(price_data["price"])
                        
        except Exception as e:
            logger.error(f"Error processing message: {e}")