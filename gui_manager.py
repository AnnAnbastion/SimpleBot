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
        self.root.geometry("1600x900")  # Made wider for split layout
        
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
        """Create market data tab with call/put split layout"""
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
        
        # Create split frame for calls and puts
        split_frame = ttk.Frame(market_frame)
        split_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - CALLS
        calls_frame = ttk.LabelFrame(split_frame, text="CALLS Market Data")
        calls_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Right side - PUTS  
        puts_frame = ttk.LabelFrame(split_frame, text="PUTS Market Data")
        puts_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Configure grid weights for equal sizing
        split_frame.grid_columnconfigure(0, weight=1)
        split_frame.grid_columnconfigure(1, weight=1)
        split_frame.grid_rowconfigure(0, weight=1)
        
        # Create calls table
        calls_columns = ('Strike', 'Expiry', 'Bid', 'Ask', 'Mid', 'Update')
        self.market_calls_tree = ttk.Treeview(calls_frame, columns=calls_columns, show='headings', height=15)
        
        for col in calls_columns:
            self.market_calls_tree.heading(col, text=col)
            self.market_calls_tree.column(col, width=80, anchor='center')
        
        # Scrollbars for calls
        calls_scrolly = ttk.Scrollbar(calls_frame, orient='vertical', command=self.market_calls_tree.yview)
        calls_scrollx = ttk.Scrollbar(calls_frame, orient='horizontal', command=self.market_calls_tree.xview)
        self.market_calls_tree.configure(yscrollcommand=calls_scrolly.set, xscrollcommand=calls_scrollx.set)
        
        # Grid layout for calls
        self.market_calls_tree.grid(row=0, column=0, sticky='nsew')
        calls_scrolly.grid(row=0, column=1, sticky='ns')
        calls_scrollx.grid(row=1, column=0, sticky='ew')
        
        calls_frame.grid_columnconfigure(0, weight=1)
        calls_frame.grid_rowconfigure(0, weight=1)
        
        # Create puts table  
        puts_columns = ('Strike', 'Expiry', 'Bid', 'Ask', 'Mid', 'Update')
        self.market_puts_tree = ttk.Treeview(puts_frame, columns=puts_columns, show='headings', height=15)
        
        for col in puts_columns:
            self.market_puts_tree.heading(col, text=col)
            self.market_puts_tree.column(col, width=80, anchor='center')
        
        # Scrollbars for puts
        puts_scrolly = ttk.Scrollbar(puts_frame, orient='vertical', command=self.market_puts_tree.yview)
        puts_scrollx = ttk.Scrollbar(puts_frame, orient='horizontal', command=self.market_puts_tree.xview)
        self.market_puts_tree.configure(yscrollcommand=puts_scrolly.set, xscrollcommand=puts_scrollx.set)
        
        # Grid layout for puts
        self.market_puts_tree.grid(row=0, column=0, sticky='nsew')
        puts_scrolly.grid(row=0, column=1, sticky='ns')
        puts_scrollx.grid(row=1, column=0, sticky='ew')
        
        puts_frame.grid_columnconfigure(0, weight=1)
        puts_frame.grid_rowconfigure(0, weight=1)
        
        logger.info("Market data tables created successfully")
        
    def create_market_table(self, parent_frame, option_type):
        """Create market data table for specific option type"""
        # Create treeview
        columns = ('Strike', 'Expiry', 'Bid', 'Ask', 'Mid', 'Last Update')
        
        if option_type == 'calls':
            self.market_calls_tree = ttk.Treeview(parent_frame, columns=columns, show='headings', height=15)
            tree = self.market_calls_tree
        else:
            self.market_puts_tree = ttk.Treeview(parent_frame, columns=columns, show='headings', height=15)
            tree = self.market_puts_tree
        
        # Define column headings and widths
        column_widths = {'Strike': 80, 'Expiry': 100, 'Bid': 80, 'Ask': 80, 'Mid': 80, 'Last Update': 120}
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=column_widths.get(col, 80))
            
        # Create frame for tree and scrollbars
        tree_frame = ttk.Frame(parent_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        scrolly = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        scrollx = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=scrolly.set, xscrollcommand=scrollx.set)
        
        # Pack tree and scrollbars
        tree.pack(in_=tree_frame, side='left', fill='both', expand=True)
        scrolly.pack(in_=tree_frame, side='right', fill='y')
        scrollx.pack(in_=tree_frame, side='bottom', fill='x')
        
    def create_pricing_tab(self):
        """Create pricing analysis tab with call/put split layout"""
        pricing_frame = ttk.Frame(self.notebook)
        self.notebook.add(pricing_frame, text="Pricing Analysis")
        
        # Title
        title_label = ttk.Label(pricing_frame, text="Options Pricing & Greeks", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create split frame for calls and puts
        split_frame = ttk.Frame(pricing_frame)
        split_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - CALLS
        calls_frame = ttk.LabelFrame(split_frame, text="CALLS Pricing & Greeks")
        calls_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Right side - PUTS
        puts_frame = ttk.LabelFrame(split_frame, text="PUTS Pricing & Greeks")
        puts_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Configure grid weights
        split_frame.grid_columnconfigure(0, weight=1)
        split_frame.grid_columnconfigure(1, weight=1)
        split_frame.grid_rowconfigure(0, weight=1)
        
        # Create calls pricing table
        pricing_columns = ('Strike', 'Expiry', 'Market', 'Theo', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega')
        self.pricing_calls_tree = ttk.Treeview(calls_frame, columns=pricing_columns, show='headings', height=20)
        
        column_widths = {'Strike': 70, 'Expiry': 70, 'Market': 70, 'Theo': 70, 'IV': 60, 
                        'Delta': 60, 'Gamma': 60, 'Theta': 60, 'Vega': 60}
        
        for col in pricing_columns:
            self.pricing_calls_tree.heading(col, text=col)
            self.pricing_calls_tree.column(col, width=column_widths.get(col, 60), anchor='center')
        
        # Scrollbars for calls pricing
        calls_p_scrolly = ttk.Scrollbar(calls_frame, orient='vertical', command=self.pricing_calls_tree.yview)
        calls_p_scrollx = ttk.Scrollbar(calls_frame, orient='horizontal', command=self.pricing_calls_tree.xview)
        self.pricing_calls_tree.configure(yscrollcommand=calls_p_scrolly.set, xscrollcommand=calls_p_scrollx.set)
        
        # Grid layout for calls pricing
        self.pricing_calls_tree.grid(row=0, column=0, sticky='nsew')
        calls_p_scrolly.grid(row=0, column=1, sticky='ns')
        calls_p_scrollx.grid(row=1, column=0, sticky='ew')
        
        calls_frame.grid_columnconfigure(0, weight=1)
        calls_frame.grid_rowconfigure(0, weight=1)
        
        # Create puts pricing table
        self.pricing_puts_tree = ttk.Treeview(puts_frame, columns=pricing_columns, show='headings', height=20)
        
        for col in pricing_columns:
            self.pricing_puts_tree.heading(col, text=col)
            self.pricing_puts_tree.column(col, width=column_widths.get(col, 60), anchor='center')
        
        # Scrollbars for puts pricing
        puts_p_scrolly = ttk.Scrollbar(puts_frame, orient='vertical', command=self.pricing_puts_tree.yview)
        puts_p_scrollx = ttk.Scrollbar(puts_frame, orient='horizontal', command=self.pricing_puts_tree.xview)
        self.pricing_puts_tree.configure(yscrollcommand=puts_p_scrolly.set, xscrollcommand=puts_p_scrollx.set)
        
        # Grid layout for puts pricing
        self.pricing_puts_tree.grid(row=0, column=0, sticky='nsew')
        puts_p_scrolly.grid(row=0, column=1, sticky='ns')
        puts_p_scrollx.grid(row=1, column=0, sticky='ew')
        
        puts_frame.grid_columnconfigure(0, weight=1)
        puts_frame.grid_rowconfigure(0, weight=1)
        
        logger.info("Pricing tables created successfully")
        
    def create_pricing_table(self, parent_frame, option_type):
        """Create pricing table for specific option type"""
        columns = ('Strike', 'Expiry', 'Market Price', 'Theo Price', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega')
        
        if option_type == 'calls':
            self.pricing_calls_tree = ttk.Treeview(parent_frame, columns=columns, show='headings', height=20)
            tree = self.pricing_calls_tree
        else:
            self.pricing_puts_tree = ttk.Treeview(parent_frame, columns=columns, show='headings', height=20)
            tree = self.pricing_puts_tree
        
        # Define column headings and widths
        column_widths = {'Strike': 80, 'Expiry': 100, 'Market Price': 90, 'Theo Price': 90, 
                        'IV': 70, 'Delta': 70, 'Gamma': 70, 'Theta': 70, 'Vega': 70}
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=column_widths.get(col, 70))
            
        # Create frame for tree and scrollbars
        tree_frame = ttk.Frame(parent_frame)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        scrolly = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        scrollx = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=scrolly.set, xscrollcommand=scrollx.set)
        
        # Pack tree and scrollbars
        tree.pack(in_=tree_frame, side='left', fill='both', expand=True)
        scrolly.pack(in_=tree_frame, side='right', fill='y')
        scrollx.pack(in_=tree_frame, side='bottom', fill='x')
        
    def create_summary_tab(self):
        """Create summary statistics tab"""
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        # Title
        title_label = ttk.Label(summary_frame, text="Market Summary", font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Create split layout for call/put statistics
        split_frame = ttk.Frame(summary_frame)
        split_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - Overall statistics
        overall_frame = ttk.LabelFrame(split_frame, text="Overall Statistics")
        overall_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right side - Call/Put breakdown
        breakdown_frame = ttk.LabelFrame(split_frame, text="Call/Put Breakdown")
        breakdown_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Overall statistics
        self.stats_vars = {
            'total_instruments': tk.StringVar(value="Total Instruments: 0"),
            'active_orderbooks': tk.StringVar(value="Active Order Books: 0"),
            'avg_iv': tk.StringVar(value="Average IV: 0.00%"),
            'data_updates': tk.StringVar(value="Data Updates: 0"),
            'last_pricing': tk.StringVar(value="Last Pricing Update: Never")
        }
        
        for var in self.stats_vars.values():
            ttk.Label(overall_frame, textvariable=var, font=('Arial', 12)).pack(anchor='w', padx=10, pady=2)
        
        # Call/Put breakdown statistics
        self.breakdown_vars = {
            'total_calls': tk.StringVar(value="Total Calls: 0"),
            'total_puts': tk.StringVar(value="Total Puts: 0"),
            'active_calls': tk.StringVar(value="Active Calls: 0"),
            'active_puts': tk.StringVar(value="Active Puts: 0"),
            'avg_call_iv': tk.StringVar(value="Avg Call IV: N/A"),
            'avg_put_iv': tk.StringVar(value="Avg Put IV: N/A")
        }
        
        for var in self.breakdown_vars.values():
            ttk.Label(breakdown_frame, textvariable=var, font=('Arial', 12)).pack(anchor='w', padx=10, pady=2)
            
    def update_display(self):
        """Update all display elements"""
        if self.updating:
            return
            
        self.updating = True
        
        try:
            # Add debug call (remove this after debugging)
            self.debug_all_data_sources()
            
            # Update BTC price
            self.update_btc_price()
            
            # Update market data tables
            self.update_market_data_tables()
            
            # Update pricing tables
            self.update_pricing_tables()
            
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
            
    # Replace these methods in gui_manager.py

    def update_market_data_tables(self):
        """Update market data tables split by call/put using persistent data"""
        try:
            # Clear existing items
            for item in self.market_calls_tree.get_children():
                self.market_calls_tree.delete(item)
            for item in self.market_puts_tree.get_children():
                self.market_puts_tree.delete(item)
                    
            # Use get_display_data() which uses persistent storage
            display_data = self.data_manager.get_display_data()
            instruments = display_data.get('instruments', {})
            orderbooks = display_data.get('orderbooks', {})  # This uses persistent orderbooks
            
            logger.info(f"Market data: {len(instruments)} instruments, {len(orderbooks)} persistent orderbooks")
            
            if not instruments:
                logger.warning("No instruments found!")
                return
                
            if not orderbooks:
                logger.warning("No persistent orderbooks found!")
                return
            
            # Sort instruments first
            sorted_instruments = self._sort_instruments_for_display(instruments)
            
            calls_count = 0
            puts_count = 0
            
            for instrument_name, instrument in sorted_instruments:
                try:
                    orderbook = orderbooks.get(instrument_name)
                    if not orderbook:
                        continue
                            
                    best_bid = self._safe_float(orderbook.best_bid)
                    best_ask = self._safe_float(orderbook.best_ask)
                    mid_price = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0
                    
                    # Show data age
                    data_age = self.data_manager.get_data_age(instrument_name)
                    age_str = f"{data_age:.0f}s" if data_age < 3600 else "old"
                    
                    values = (
                        self._safe_format_number(instrument.strike, 0),
                        instrument.expiration.strftime('%m/%d') if hasattr(instrument.expiration, 'strftime') else "N/A",
                        self._safe_format_number(best_bid, 4),
                        self._safe_format_number(best_ask, 4),
                        self._safe_format_number(mid_price, 4),
                        f"{orderbook.timestamp.strftime('%H:%M:%S')} ({age_str})" if hasattr(orderbook.timestamp, 'strftime') else "N/A"
                    )
                    
                    # Add to appropriate table
                    if instrument.option_type.lower() == 'call':
                        self.market_calls_tree.insert('', 'end', values=values)
                        calls_count += 1
                    else:
                        self.market_puts_tree.insert('', 'end', values=values)
                        puts_count += 1
                            
                except Exception as e:
                    logger.error(f"Error processing instrument {instrument_name}: {e}")
                    continue
            
            logger.info(f"Updated market tables: {calls_count} calls, {puts_count} puts")
                        
        except Exception as e:
            logger.error(f"Error updating market data tables: {e}")
            import traceback
            traceback.print_exc()
                    
    def update_pricing_tables(self):
        """Update pricing tables split by call/put using persistent data"""
        try:
            # Clear existing items
            for item in self.pricing_calls_tree.get_children():
                self.pricing_calls_tree.delete(item)
            for item in self.pricing_puts_tree.get_children():
                self.pricing_puts_tree.delete(item)
                    
            # Use display data which includes persistent pricing
            display_data = self.data_manager.get_display_data()
            pricing_results = display_data.get('pricing', {})  # This uses persistent pricing
            instruments = display_data.get('instruments', {})
            
            logger.info(f"Pricing data: {len(pricing_results)} persistent results, {len(instruments)} instruments")
            
            if not pricing_results:
                logger.warning("No persistent pricing results found!")
                return
                
            if not instruments:
                logger.warning("No instruments found for pricing!")
                return
            
            # Sort instruments first
            sorted_instruments = self._sort_instruments_for_display(instruments)
            
            calls_count = 0
            puts_count = 0
            
            for instrument_name, instrument in sorted_instruments:
                try:
                    result = pricing_results.get(instrument_name)
                    if not result:
                        continue
                            
                    values = (
                        self._safe_format_number(instrument.strike, 0),
                        instrument.expiration.strftime('%m/%d') if hasattr(instrument.expiration, 'strftime') else "N/A",
                        self._safe_format_number(result.get('market_price'), 4),
                        self._safe_format_number(result.get('theoretical_price'), 4),
                        self._safe_format_percentage(result.get('implied_volatility')),
                        self._safe_format_number(result.get('delta'), 3, "+0.000"),
                        self._safe_format_number(result.get('gamma'), 3),
                        self._safe_format_number(result.get('theta'), 3, "+0.000"),
                        self._safe_format_number(result.get('vega'), 3)
                    )
                    
                    # Add to appropriate table
                    if instrument.option_type.lower() == 'call':
                        self.pricing_calls_tree.insert('', 'end', values=values)
                        calls_count += 1
                    else:
                        self.pricing_puts_tree.insert('', 'end', values=values)
                        puts_count += 1
                            
                except Exception as e:
                    logger.error(f"Error processing pricing result {instrument_name}: {e}")
                    continue
            
            logger.info(f"Updated pricing tables: {calls_count} calls, {puts_count} puts")
                        
        except Exception as e:
            logger.error(f"Error updating pricing tables: {e}")
            import traceback
            traceback.print_exc()

    def update_summary(self):
        """Update summary statistics using persistent data"""
        try:
            # Use display data for persistent storage
            display_data = self.data_manager.get_display_data()
            pricing_results = display_data.get('pricing', {})  # Persistent pricing
            instruments = display_data.get('instruments', {})
            orderbooks = display_data.get('orderbooks', {})   # Persistent orderbooks
            
            # Overall statistics
            total_instruments = len(instruments)
            
            # Count active orderbooks from persistent storage
            active_orderbooks = len(orderbooks)  # All persistent orderbooks are "active"
            
            # Calculate average IV from persistent pricing
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
            
            # Call/Put breakdown
            total_calls = len([inst for inst in instruments.values() if inst.option_type.lower() == 'call'])
            total_puts = len([inst for inst in instruments.values() if inst.option_type.lower() == 'put'])
            
            # Active calls/puts from persistent data
            active_calls = 0
            active_puts = 0
            call_ivs = []
            put_ivs = []
            
            for name, result in pricing_results.items():
                try:
                    inst = instruments.get(name)
                    if inst:
                        iv = self._safe_float(result.get('implied_volatility'))
                        if iv > 0:
                            if inst.option_type.lower() == 'call':
                                active_calls += 1
                                call_ivs.append(iv)
                            else:
                                active_puts += 1
                                put_ivs.append(iv)
                except:
                    continue
            
            avg_call_iv = sum(call_ivs) / len(call_ivs) if call_ivs else 0
            avg_put_iv = sum(put_ivs) / len(put_ivs) if put_ivs else 0
                
            # Update overall display
            self.stats_vars['total_instruments'].set(f"Total Instruments: {total_instruments}")
            self.stats_vars['active_orderbooks'].set(f"Persistent Order Books: {active_orderbooks}")
            self.stats_vars['avg_iv'].set(f"Average IV: {avg_iv:.1%}" if avg_iv > 0 else "Average IV: N/A")
            self.stats_vars['data_updates'].set(f"Data Updates: {self.data_manager.get_update_count()}")
            self.stats_vars['last_pricing'].set(f"Last Pricing Update: {datetime.now().strftime('%H:%M:%S')}")
            
            # Update breakdown display
            self.breakdown_vars['total_calls'].set(f"Total Calls: {total_calls}")
            self.breakdown_vars['total_puts'].set(f"Total Puts: {total_puts}")
            self.breakdown_vars['active_calls'].set(f"Persistent Calls: {active_calls}")
            self.breakdown_vars['active_puts'].set(f"Persistent Puts: {active_puts}")
            self.breakdown_vars['avg_call_iv'].set(f"Avg Call IV: {avg_call_iv:.1%}" if avg_call_iv > 0 else "Avg Call IV: N/A")
            self.breakdown_vars['avg_put_iv'].set(f"Avg Put IV: {avg_put_iv:.1%}" if avg_put_iv > 0 else "Avg Put IV: N/A")
            
            logger.info(f"Summary updated: {total_instruments} instruments, {active_orderbooks} persistent orderbooks, {len(pricing_results)} persistent pricing results")
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
            import traceback
            traceback.print_exc()
        
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
    
    # Add this method to your MarketDataGUI class in gui_manager.py
    def debug_all_data_sources(self):
        """Comprehensive debug of all data sources"""
        try:
            logger.info("=== COMPREHENSIVE DATA DEBUG ===")
            
            # 1. Check data manager basic info
            spot_price = self.data_manager.get_spot_price()
            update_count = self.data_manager.get_update_count()
            logger.info(f"Data Manager - Spot Price: {spot_price}, Updates: {update_count}")
            
            # 2. Check market data
            market_data = self.data_manager.get_all_market_data()
            logger.info(f"Market Data Keys: {list(market_data.keys())}")
            logger.info(f"Instruments Count: {len(market_data.get('instruments', {}))}")
            logger.info(f"Orderbooks Count: {len(market_data.get('orderbooks', {}))}")
            
            # 3. Show first few instruments
            instruments = market_data.get('instruments', {})
            logger.info("First 3 instruments:")
            for i, (name, inst) in enumerate(list(instruments.items())[:3]):
                logger.info(f"  {i+1}. {name} - Strike: {inst.strike} - Type: {inst.option_type} - Expiry: {inst.expiration}")
            
            # 4. Show first few orderbooks
            orderbooks = market_data.get('orderbooks', {})
            logger.info("First 3 orderbooks:")
            for i, (name, ob) in enumerate(list(orderbooks.items())[:3]):
                logger.info(f"  {i+1}. {name} - Bid: {ob.best_bid} - Ask: {ob.best_ask} - Time: {ob.timestamp}")
            
            # 5. Check pricing results
            pricing_results = self.data_manager.get_latest_pricing_results()
            logger.info(f"Pricing Results Count: {len(pricing_results)}")
            
            # Show first few pricing results
            logger.info("First 3 pricing results:")
            for i, (name, result) in enumerate(list(pricing_results.items())[:3]):
                market_price = result.get('market_price', 'N/A')
                iv = result.get('implied_volatility', 'N/A')
                logger.info(f"  {i+1}. {name} - Market Price: {market_price} - IV: {iv}")
            
            # 6. Check if pricer is working
            fresh_pricing = self.pricer.calculate_all()
            logger.info(f"Fresh Pricing Calculation Count: {len(fresh_pricing)}")
            
            logger.info("=== END DATA DEBUG ===")
            
        except Exception as e:
            logger.error(f"Debug failed: {e}")
            import traceback
            traceback.print_exc()