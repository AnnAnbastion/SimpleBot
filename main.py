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
        
    async def start_async_components(self):
        """Start the async trading components"""
        try:
            logger.info("Starting async components...")
            
            # Connect to Deribit
            await self.deribit_client.connect()
            await asyncio.sleep(2)
            
            # Get BTC instruments (this now gets spot price automatically and selects best ones)
            btc_instruments = await self.deribit_client.get_btc_instruments()
            logger.info(f"Found {len(btc_instruments)} BTC instruments")
            
            if btc_instruments:
                self.selected_instruments = btc_instruments
                
                # Subscribe to market data (WebSocket real-time updates)
                await self.deribit_client.subscribe_market_data(btc_instruments)
                await asyncio.sleep(3)  # Wait for initial WebSocket data
                
                # Check what data we have
                market_data = self.data_manager.get_all_market_data()
                spot_price = self.data_manager.get_spot_price()
                logger.info(f"Current BTC spot: ${spot_price:,.2f}")
                logger.info(f"Instruments: {len(market_data.get('instruments', {}))}")
                
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
                        
                logger.info(f"Active orderbooks: {active_count}")
                
                # Start both loops concurrently
                await asyncio.gather(
                    self.data_update_loop(),      # Downloads data every 100ms
                    self.pricing_update_loop()    # Calculates pricing every 100ms
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
        
        while self.running:
            try:
                iteration += 1
                
                # Get fresh orderbook data for a subset of instruments each iteration
                # This prevents overwhelming the API while keeping data fresh
                instruments_per_batch = 5  # Update 5 instruments per 100ms cycle
                start_idx = (iteration * instruments_per_batch) % len(self.selected_instruments)
                end_idx = min(start_idx + instruments_per_batch, len(self.selected_instruments))
                
                batch_instruments = self.selected_instruments[start_idx:end_idx]
                
                # Update orderbooks for this batch
                await self.update_orderbook_batch(batch_instruments)
                
                # Also update BTC spot price every 10 iterations (1 second)
                if iteration % 10 == 0:
                    await self.update_spot_price()
                
                # Log progress occasionally
                if iteration % 100 == 0:  # Every 10 seconds
                    market_data = self.data_manager.get_all_market_data()
                    active_obs = sum(1 for ob in market_data.get('orderbooks', {}).values() 
                                   if hasattr(ob, 'best_bid') and float(getattr(ob, 'best_bid', 0)) > 0)
                    logger.info(f"Data update cycle {iteration}: {active_obs} active orderbooks")
                
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
                
                # Calculate prices and greeks immediately with current data
                pricing_results = self.pricer.calculate_all()
                
                # Store results in data manager for GUI access
                self.data_manager.store_pricing_results(pricing_results)
                
                # Log occasionally
                if iteration % 50 == 0:  # Every 5 seconds
                    spot_price = self.data_manager.get_spot_price()
                    logger.info(f"Pricing update: Spot=${spot_price:,.2f}, Priced {len(pricing_results)} instruments")
                
                await asyncio.sleep(0.1)  # 100ms
                
            except Exception as e:
                logger.error(f"Error in pricing loop: {e}")
                await asyncio.sleep(1)
                
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

def main():
    """Main entry point"""
    app = TradingApplication()
    app.start_gui()

if __name__ == "__main__":
    main()