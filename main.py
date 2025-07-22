# main.py
import asyncio
import logging
import threading
import tkinter as tk
from deribit_client import DeribitClient
from data_manager import DataManager
from pricer import OptionsPricer
from gui_manager import MarketDataGUI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingApplication:
    def __init__(self):
        self.data_manager = DataManager()
        self.deribit_client = DeribitClient(self.data_manager)
        self.pricer = OptionsPricer(self.data_manager)
        self.gui = None
        self.running = False
        self.selected_instruments = []
        
        # NEW: Configuration for pricing model and synthetic method
        self.use_black_model = True  # Set to False to use Black-Scholes
        self.synthetic_method = 'A'  # 'A' for minimal parity gap, 'B' for basis curve
        
    async def start_async_components(self):
        """Start the async trading components"""
        try:
            logger.info("Starting async components...")
            logger.info(f"Using {'Black model' if self.use_black_model else 'Black-Scholes model'} "
                       f"with synthetic method {self.synthetic_method}")
            
            # Connect to Deribit
            await self.deribit_client.connect()
            await asyncio.sleep(2)
            
            # Get BTC instruments (this now gets spot price automatically and selects best ones)
            btc_instruments = await self.deribit_client.get_btc_instruments()
            logger.info(f"Found {len(btc_instruments)} BTC instruments")
            
            if btc_instruments:
                self.selected_instruments = btc_instruments
                
                # Get futures for Black model
                btc_futures = await self.deribit_client.get_btc_futures()
                await self.deribit_client.subscribe_futures_data(btc_futures)
                
                # Subscribe to market data (WebSocket real-time updates)
                await self.deribit_client.subscribe_market_data(btc_instruments)
                await asyncio.sleep(3)  # Wait for initial WebSocket data
                
                # Check what data we have
                market_data = self.data_manager.get_all_market_data()
                display_data = self.data_manager.get_display_data()
                spot_price = self.data_manager.get_spot_price()
                logger.info(f"Current BTC spot: ${spot_price:,.2f}")
                logger.info(f"Options instruments: {len(market_data.get('instruments', {}))}")
                logger.info(f"Futures instruments: {len(display_data.get('futures_instruments', {}))}")
                
                # Count active orderbooks safely
                active_count = 0
                for ob in market_data.get('orderbooks', {}).values():
                    try:
                        if hasattr(ob, 'best_bid') and hasattr(ob, 'best_ask'):
                            bid = float(ob.best_bid) if ob.best_bid else 0
                            ask = float(ob.best_ask) if ob.best_ask else 0
                            if bid > 0 and ask > 0:
                                active_count += 1
                    except:
                        continue
                        
                # Count active futures orderbooks
                futures_active_count = 0
                for ob in display_data.get('futures_orderbooks', {}).values():
                    try:
                        if hasattr(ob, 'best_bid') and hasattr(ob, 'best_ask'):
                            bid = float(ob.best_bid) if ob.best_bid else 0
                            ask = float(ob.best_ask) if ob.best_ask else 0
                            if bid > 0 and ask > 0:
                                futures_active_count += 1
                    except:
                        continue
                        
                logger.info(f"Active options orderbooks: {active_count}")
                logger.info(f"Active futures orderbooks: {futures_active_count}")
                
                # Start all loops concurrently
                await asyncio.gather(
                    self.data_update_loop(),        # Downloads data every 100ms
                    self.pricing_update_loop(),     # Calculates pricing every 100ms
                    self.cache_maintenance_loop()   # NEW: Maintains cache every 30 seconds
                )
            else:
                logger.error("No instruments found, cannot continue")
                
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            import traceback
            traceback.print_exc()
    
    async def data_update_loop(self):
        """Download fresh market data every 100ms"""
        logger.info("Starting data update loop (100ms intervals)...")
        iteration = 0
        
        # Use cached instruments list (calculated only once)
        instruments_to_update = self.deribit_client.get_cached_instruments()
        
        while self.running:
            try:
                iteration += 1
                
                # Update orderbooks in batches (using cached instrument list)
                instruments_per_batch = 5
                start_idx = (iteration * instruments_per_batch) % len(instruments_to_update)
                end_idx = min(start_idx + instruments_per_batch, len(instruments_to_update))
                
                batch_instruments = instruments_to_update[start_idx:end_idx]
                await self.update_orderbook_batch(batch_instruments)
                
                # Update BTC spot price every 10 iterations
                if iteration % 10 == 0:
                    await self.update_spot_price()
                
                # Log progress occasionally
                if iteration % 100 == 0:
                    display_data = self.data_manager.get_display_data()
                    active_obs = len(display_data.get('orderbooks', {}))
                    active_pricing = len(display_data.get('pricing', {}))
                    
                    # NEW: Show cache status
                    if self.use_black_model:
                        cache_status = self.data_manager.get_synthetic_futures_cache_status()
                        logger.info(f"Data cycle {iteration}: {active_obs} orderbooks, {active_pricing} priced instruments, "
                                  f"cache: {cache_status['cache_size']} entries")
                    else:
                        logger.info(f"Data cycle {iteration}: {active_obs} orderbooks, {active_pricing} priced instruments")
                
                await asyncio.sleep(0.1)  # 100ms
                
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(1)
    
    async def update_orderbook_batch(self, instruments_batch):
        """Update orderbooks for a batch of instruments"""
        tasks = []
        for instrument in instruments_batch:
            task = self.get_single_orderbook(instrument.get('instrument_name'))
            tasks.append(task)
        
        if tasks:
            # Execute all requests concurrently for speed
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_single_orderbook(self, instrument_name):
        """Get orderbook for a single instrument"""
        try:
            if not instrument_name:
                return
                
            response = await self.deribit_client._send_request("public/get_order_book", {
                "instrument_name": instrument_name
            })
            
            if "result" in response:
                orderbook_data = response["result"]
                bids = orderbook_data.get('bids', [])
                asks = orderbook_data.get('asks', [])
                
                # Only update if we have real data
                if bids or asks:
                    self.data_manager.update_orderbook({
                        'channel': f'book.{instrument_name}',
                        'data': {
                            'bids': bids,
                            'asks': asks
                        }
                    })
                    
        except Exception as e:
            # Don't log every individual failure to avoid spam
            pass
    
    async def update_spot_price(self):
        """Update BTC spot price"""
        try:
            response = await self.deribit_client._send_request("public/get_index", {
                "currency": "BTC"
            })
            
            if "result" in response:
                btc_data = response["result"]
                spot_price = float(btc_data.get("BTC", 0))
                
                if 10000 <= spot_price <= 200000:  # Reasonable range
                    self.data_manager.update_spot_price(spot_price)
                    
        except Exception as e:
            logger.debug(f"Error updating spot price: {e}")
            
    async def pricing_update_loop(self):
        """Calculate pricing every 100ms immediately after data updates"""
        logger.info("Starting pricing calculation loop (100ms intervals)...")
        iteration = 0
        
        # Small delay to let first data update complete
        await asyncio.sleep(0.05)  # 50ms offset from data updates
        
        while self.running:
            try:
                iteration += 1
                
                # Calculate prices and greeks with selected model
                pricing_results = self.pricer.calculate_all(
                    synthetic_method=self.synthetic_method,
                    use_black_model=self.use_black_model
                )
                
                # Store results in data manager for GUI access
                self.data_manager.store_pricing_results(pricing_results)
                
                # Log occasionally with model info
                if iteration % 50 == 0:  # Every 5 seconds
                    spot_price = self.data_manager.get_spot_price()
                    model_info = f"{'Black' if self.use_black_model else 'BS'}"
                    
                    # Show cache efficiency for Black model
                    if self.use_black_model and pricing_results:
                        cached_count = sum(1 for r in pricing_results.values() if r.get('was_cached', False))
                        cache_efficiency = cached_count / len(pricing_results) * 100 if pricing_results else 0
                        logger.info(f"Pricing update ({model_info}): Spot=${spot_price:,.2f}, "
                                  f"Priced {len(pricing_results)} instruments, "
                                  f"Cache efficiency: {cache_efficiency:.1f}%")
                    else:
                        logger.info(f"Pricing update ({model_info}): Spot=${spot_price:,.2f}, "
                                  f"Priced {len(pricing_results)} instruments")
                
                await asyncio.sleep(0.1)  # 100ms
                
            except Exception as e:
                logger.error(f"Error in pricing loop: {e}")
                await asyncio.sleep(1)
    
    async def cache_maintenance_loop(self):
        """NEW: Maintain synthetic futures cache every 30 seconds"""
        if not self.use_black_model:
            return  # No cache maintenance needed for Black-Scholes
            
        logger.info("Starting cache maintenance loop (30s intervals)...")
        
        # Wait for initial data to be available
        await asyncio.sleep(10)
        
        while self.running:
            try:
                # Get current market data
                display_data = self.data_manager.get_display_data()
                instruments = display_data.get('instruments', {})
                orderbooks = display_data.get('orderbooks', {})
                spot_price = self.data_manager.get_spot_price()
                
                if instruments and orderbooks and spot_price > 0:
                    # Force cache update by calling the pricer method
                    self.pricer.update_synthetic_futures_cache_if_needed(
                        instruments, orderbooks, spot_price, self.synthetic_method
                    )
                    
                    # Log cache status
                    cache_status = self.data_manager.get_synthetic_futures_cache_status()
                    logger.info(f"Cache maintenance: {cache_status['cache_size']} entries, "
                               f"last update: {cache_status['time_since_update_seconds']:.1f}s ago")
                
                await asyncio.sleep(30)  # 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cache maintenance loop: {e}")
                await asyncio.sleep(30)
    
    def start_gui(self):
        """Start GUI on main thread"""
        self.running = True
        
        # Create GUI
        self.gui = MarketDataGUI(self.data_manager, self.pricer)
        
        # Start async components in background thread
        async_thread = threading.Thread(
            target=self.run_async_loop, 
            daemon=True
        )
        async_thread.start()
        
        # Start GUI update timer (5 second intervals)
        self.schedule_gui_update()
        
        logger.info("Starting GUI...")
        
        try:
            # Start GUI main loop (this blocks until window is closed)
            self.gui.start()
        except Exception as e:
            logger.error(f"GUI error: {e}")
        finally:
            self.running = False
            logger.info("Application shutting down...")
            
    def run_async_loop(self):
        """Run async event loop in background thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run async components
            loop.run_until_complete(self.start_async_components())
            
        except Exception as e:
            logger.error(f"Error in async loop: {e}")
        finally:
            try:
                loop.close()
            except:
                pass
                
    def schedule_gui_update(self):
        """Schedule GUI updates every 5 seconds"""
        if self.running and self.gui:
            try:
                # GUI updates every 5 seconds with the latest pricing data
                self.gui.update_display()
            except Exception as e:
                logger.error(f"Error updating GUI: {e}")
            
            # Schedule next update
            if self.gui and self.gui.root:
                self.gui.root.after(5000, self.schedule_gui_update)  # 5 seconds

    def set_pricing_model(self, use_black_model: bool, synthetic_method: str = 'A'):
        """NEW: Method to change pricing model configuration"""
        self.use_black_model = use_black_model
        self.synthetic_method = synthetic_method
        
        logger.info(f"Pricing model changed to: {'Black' if use_black_model else 'Black-Scholes'}")
        if use_black_model:
            logger.info(f"Synthetic method set to: {synthetic_method}")
            
            # Clear cache when switching methods
            self.data_manager.clear_stale_synthetic_futures_cache()

def main():
    """Main entry point"""
    app = TradingApplication()
    
    # CONFIGURATION: Change these to switch models
    app.set_pricing_model(
        use_black_model=True,     # True for Black model, False for Black-Scholes
        synthetic_method='A'      # 'A' for minimal parity gap, 'B' for basis curve
    )
    
    app.start_gui()

if __name__ == "__main__":
    main()