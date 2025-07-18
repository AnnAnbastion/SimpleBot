# gui_manager.py
import tkinter as tk
from tkinter import ttk
import pandas as pd
import threading
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MarketDataGUI:
    def __init__(self, data_manager, pricer):
        self.data_manager = data_manager
        self.pricer = pricer
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Deribit Options Trading Dashboard")
        self.root.geometry("1400x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_market_data_tab()
        self.create_pricing_tab()
        self.create_summary_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Update flag
        self.updating = False
        
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
        
    def _safe_format_number(self, value, decimal_places=2, default_text="N/A") -> str:
        """Safely format number for display"""
        try:
            num_value = self._safe_float(value)
            if num_value == 0:
                return default_text
            return f"{num_value:.{decimal_places}f}"
        except:
            return default_text
            
    def _safe_format_percentage(self, value, default_text="N/A") -> str:
        """Safely format percentage for display"""
        try:
            num_value = self._safe_float(value)
            if num_value == 0:
                return default_text
            return f"{num_value:.1%}"
        except:
            return default_text
        
    def create_market_data_tab(self):
        """Create market data tab"""
        market_frame = ttk.Frame(self.notebook)
        self.notebook.add(market_frame, text="Market Data")
        
        # Title
        title_label = ttk.Label(market_frame, text="Real-time Market Data", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # BTC Price frame
        btc_frame = ttk.LabelFrame(market_frame, text="BTC Spot Price")
        btc_frame.pack(fill='x', padx=10, pady=5)
        
        self.btc_price_var = tk.StringVar(value="$0.00")
        self.btc_time_var = tk.StringVar(value="Last Update: Never")
        
        ttk.Label(btc_frame, textvariable=self.btc_price_var, font=('Arial', 24, 'bold')).pack()
        ttk.Label(btc_frame, textvariable=self.btc_time_var).pack()
        
        # Market data table
        table_frame = ttk.LabelFrame(market_frame, text="Options Market Data")
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for market data
        columns = ('Instrument', 'Strike', 'Type', 'Expiry', 'Bid', 'Ask', 'Mid', 'Last Update')
        self.market_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define column headings and widths
        column_widths = {'Instrument': 200, 'Strike': 80, 'Type': 60, 'Expiry': 100, 
                        'Bid': 80, 'Ask': 80, 'Mid': 80, 'Last Update': 120}
        
        for col in columns:
            self.market_tree.heading(col, text=col)
            self.market_tree.column(col, width=column_widths.get(col, 100))
            
        # Scrollbars
        market_scrolly = ttk.Scrollbar(table_frame, orient='vertical', command=self.market_tree.yview)
        market_scrollx = ttk.Scrollbar(table_frame, orient='horizontal', command=self.market_tree.xview)
        self.market_tree.configure(yscrollcommand=market_scrolly.set, xscrollcommand=market_scrollx.set)
        
        # Pack tree and scrollbars
        self.market_tree.pack(side='left', fill='both', expand=True)
        market_scrolly.pack(side='right', fill='y')
        market_scrollx.pack(side='bottom', fill='x')
        
    def create_pricing_tab(self):
        """Create pricing analysis tab"""
        pricing_frame = ttk.Frame(self.notebook)
        self.notebook.add(pricing_frame, text="Pricing Analysis")
        
        # Title
        title_label = ttk.Label(pricing_frame, text="Options Pricing & Greeks", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Pricing table
        table_frame = ttk.LabelFrame(pricing_frame, text="Theoretical Pricing & Risk Metrics")
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        columns = ('Instrument', 'Market Price', 'Theo Price', 'Diff', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega')
        self.pricing_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Define column headings and widths
        column_widths = {'Instrument': 200, 'Market Price': 100, 'Theo Price': 100, 'Diff': 80,
                        'IV': 80, 'Delta': 80, 'Gamma': 80, 'Theta': 80, 'Vega': 80}
        
        for col in columns:
            self.pricing_tree.heading(col, text=col)
            self.pricing_tree.column(col, width=column_widths.get(col, 100))
            
        # Scrollbars
        pricing_scrolly = ttk.Scrollbar(table_frame, orient='vertical', command=self.pricing_tree.yview)
        pricing_scrollx = ttk.Scrollbar(table_frame, orient='horizontal', command=self.pricing_tree.xview)
        self.pricing_tree.configure(yscrollcommand=pricing_scrolly.set, xscrollcommand=pricing_scrollx.set)
        
        # Pack tree and scrollbars
        self.pricing_tree.pack(side='left', fill='both', expand=True)
        pricing_scrolly.pack(side='right', fill='y')
        pricing_scrollx.pack(side='bottom', fill='x')
        
    def create_summary_tab(self):
        """Create summary statistics tab"""
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        # Title
        title_label = ttk.Label(summary_frame, text="Market Summary", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(summary_frame, text="Statistics")
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        # Create statistics labels
        self.stats_vars = {
            'total_instruments': tk.StringVar(value="Total Instruments: 0"),
            'active_orderbooks': tk.StringVar(value="Active Order Books: 0"),
            'avg_iv': tk.StringVar(value="Average IV: 0.00%"),
            'data_updates': tk.StringVar(value="Data Updates: 0"),
            'last_pricing': tk.StringVar(value="Last Pricing Update: Never")
        }
        
        for var in self.stats_vars.values():
            ttk.Label(stats_frame, textvariable=var, font=('Arial', 12)).pack(anchor='w', padx=10, pady=2)
            
    def update_display(self):
        """Update all display elements"""
        if self.updating:
            return
            
        self.updating = True
        
        try:
            # Update BTC price
            self.update_btc_price()
            
            # Update market data table
            self.update_market_data_table()
            
            # Update pricing table
            self.update_pricing_table()
            
            # Update summary
            self.update_summary()
            
            # Update status
            self.status_var.set(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Update Error: {str(e)}")
            
        finally:
            self.updating = False
            
    def update_btc_price(self):
        """Update BTC spot price display"""
        try:
            spot_price = self._safe_float(self.data_manager.get_spot_price())
            if spot_price > 0:
                self.btc_price_var.set(f"${spot_price:,.2f}")
                self.btc_time_var.set(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.btc_price_var.set("$0.00")
                self.btc_time_var.set("Waiting for data...")
        except Exception as e:
            logger.error(f"Error updating BTC price: {e}")
            
    def update_market_data_table(self):
        """Update market data table with sorted display"""
        try:
            # Clear existing items
            for item in self.market_tree.get_children():
                self.market_tree.delete(item)
                
            # Get display data (uses persistent storage)
            display_data = self.data_manager.get_display_data()
            instruments = display_data.get('instruments', {})
            orderbooks = display_data.get('orderbooks', {})
            
            # Group and sort instruments
            sorted_instruments = self._sort_instruments_for_display(instruments)
            
            # Populate table
            for instrument_name, instrument in sorted_instruments:
                try:
                    orderbook = orderbooks.get(instrument_name)
                    
                    if orderbook:
                        best_bid = self._safe_float(orderbook.best_bid)
                        best_ask = self._safe_float(orderbook.best_ask)
                        mid_price = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0
                        
                        # Show data age
                        data_age = self.data_manager.get_data_age(instrument_name)
                        age_str = f"{data_age:.0f}s" if data_age < 3600 else "old"
                        
                        values = (
                            str(instrument_name),
                            self._safe_format_number(instrument.strike, 0),
                            str(instrument.option_type).upper(),
                            instrument.expiration.strftime('%Y-%m-%d') if hasattr(instrument.expiration, 'strftime') else "N/A",
                            self._safe_format_number(best_bid, 4),
                            self._safe_format_number(best_ask, 4),
                            self._safe_format_number(mid_price, 4),
                            f"{orderbook.timestamp.strftime('%H:%M:%S')} ({age_str})" if hasattr(orderbook.timestamp, 'strftime') else "N/A"
                        )
                        self.market_tree.insert('', 'end', values=values)
                        
                except Exception as e:
                    logger.error(f"Error processing instrument {instrument_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error updating market data table: {e}")
                
    def update_pricing_table(self):
        """Update pricing analysis table with sorted display"""
        try:
            # Clear existing items
            for item in self.pricing_tree.get_children():
                self.pricing_tree.delete(item)
                
            # Get display data
            display_data = self.data_manager.get_display_data()
            pricing_results = display_data.get('pricing', {})
            instruments = display_data.get('instruments', {})
            
            # Sort instruments for display
            sorted_instruments = self._sort_instruments_for_display(instruments)
            
            # Populate table with sorted data
            for instrument_name, instrument in sorted_instruments:
                try:
                    result = pricing_results.get(instrument_name)
                    if not result:
                        continue
                        
                    values = (
                        str(instrument_name),
                        self._safe_format_number(result.get('market_price'), 4),
                        self._safe_format_number(result.get('theoretical_price'), 4),
                        self._safe_format_number(result.get('price_diff'), 4, "+0.0000"),
                        self._safe_format_percentage(result.get('implied_volatility')),
                        self._safe_format_number(result.get('delta'), 3, "+0.000"),
                        self._safe_format_number(result.get('gamma'), 3),
                        self._safe_format_number(result.get('theta'), 3, "+0.000"),
                        self._safe_format_number(result.get('vega'), 3)
                    )
                    self.pricing_tree.insert('', 'end', values=values)
                    
                except Exception as e:
                    logger.error(f"Error processing pricing result {instrument_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error updating pricing table: {e}")
            
    def update_summary(self):
        """Update summary statistics"""
        try:
            market_data = self.data_manager.get_all_market_data()
            pricing_results = self.pricer.calculate_all()
            
            # Calculate statistics safely
            total_instruments = len(market_data.get('instruments', {}))
            
            # Count active orderbooks
            active_orderbooks = 0
            for orderbook in market_data.get('orderbooks', {}).values():
                try:
                    if (self._safe_float(orderbook.best_bid) > 0 and 
                        self._safe_float(orderbook.best_ask) > 0):
                        active_orderbooks += 1
                except:
                    continue
            
            # Calculate average IV
            avg_iv = 0
            if pricing_results:
                valid_ivs = []
                for result in pricing_results.values():
                    try:
                        iv = self._safe_float(result.get('implied_volatility'))
                        if iv > 0:
                            valid_ivs.append(iv)
                    except:
                        continue
                        
                avg_iv = sum(valid_ivs) / len(valid_ivs) if valid_ivs else 0
                
            # Update display
            self.stats_vars['total_instruments'].set(f"Total Instruments: {total_instruments}")
            self.stats_vars['active_orderbooks'].set(f"Active Order Books: {active_orderbooks}")
            self.stats_vars['avg_iv'].set(f"Average IV: {avg_iv:.1%}" if avg_iv > 0 else "Average IV: N/A")
            self.stats_vars['data_updates'].set(f"Data Updates: {self.data_manager.get_update_count()}")
            self.stats_vars['last_pricing'].set(f"Last Pricing Update: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
        
    def start(self):
        """Start the GUI"""
        self.root.mainloop()
        
    def close(self):
        """Close the GUI"""
        self.root.quit()
        self.root.destroy()

    def _sort_instruments_for_display(self, instruments: dict) -> list[tuple]:
        """Sort instruments by expiry, then strike, then calls before puts"""
        try:
            instrument_list = list(instruments.items())
            
            def sort_key(item):
                instrument_name, instrument = item
                
                # Extract sorting criteria
                expiry = getattr(instrument, 'expiration', datetime.now()).timestamp()
                strike = float(getattr(instrument, 'strike', 0))
                option_type = getattr(instrument, 'option_type', 'call').lower()
                
                # Sort by: expiry, strike, then calls before puts
                type_order = 0 if option_type == 'call' else 1
                
                return (expiry, strike, type_order)
            
            sorted_list = sorted(instrument_list, key=sort_key)
            return sorted_list
            
        except Exception as e:
            logger.error(f"Error sorting instruments: {e}")
            return list(instruments.items())