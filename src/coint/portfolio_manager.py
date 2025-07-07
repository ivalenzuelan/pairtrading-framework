#!/usr/bin/env python3
"""
Portfolio Management Module
Handles portfolio state, position tracking, and value calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class Portfolio:
    """Manages overall portfolio state"""
    def __init__(self, initial_cash: float = 1000000):
        self.cash = initial_cash
        self.positions = {}  # Format: {'asset': {'qty': quantity}}
        self.value_history = []  # (timestamp, total_value)
        self.trades = []
        
    def update_position(self, asset: str, qty: float):
        """Update position for an asset"""
        asset_key = f"{asset}_position"
        
        if asset_key not in self.positions:
            self.positions[asset_key] = {"qty": 0}
        
        self.positions[asset_key]["qty"] += qty
        
        # Clean up zero positions
        if abs(self.positions[asset_key]["qty"]) < 1e-6:
            del self.positions[asset_key]
    
    def calculate_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        for asset, position in self.positions.items():
            crypto = asset.split('_')[0]
            if crypto in prices:
                total_value += position["qty"] * prices[crypto]
        return total_value
    
    def record_value(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """Record portfolio value at a specific timestamp"""
        current_value = self.calculate_value(prices)
        self.value_history.append((timestamp, current_value))
    
    def process_trade(self, trade: dict):
        """Update portfolio based on trade execution"""
        # Update cash
        cost = 0
        
        # Process leader transaction
        leader_qty = trade["leader_qty"]
        leader_exec = trade["leader_price"]
        leader_side = trade["leader_side"]
        
        if leader_side == "buy":
            cost -= leader_qty * leader_exec
            self.update_position(trade["leader"], leader_qty)
        else:  # sell
            cost += leader_qty * leader_exec
            self.update_position(trade["leader"], -leader_qty)
        
        # Process follower transaction
        follower_qty = trade["follower_qty"]
        follower_exec = trade["follower_price"]
        follower_side = trade["follower_side"]
        
        if follower_side == "buy":
            cost -= follower_qty * follower_exec
            self.update_position(trade["follower"], follower_qty)
        else:  # sell
            cost += follower_qty * follower_exec
            self.update_position(trade["follower"], -follower_qty)
        
        # Apply fees
        self.cash += cost - trade["fee"]
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        if not self.value_history:
            return {}
        
        # Extract portfolio values
        values = [v for _, v in self.value_history]
        returns = pd.Series(values).pct_change().dropna()
        initial_value = values[0]
        final_value = values[-1]
        
        # Calculate returns
        cumulative_return = (final_value / initial_value) - 1
        
        # Calculate volatility (annualized)
        volatility = returns.std() * np.sqrt(365.25 * 24)  # Assuming minute data
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365.25 * 24) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade analysis
        if self.trades:
            trades = pd.DataFrame(self.trades)
            
            # Calculate profit for each trade
            trades["profit"] = 0
            for i, trade in trades.iterrows():
                # Calculate P&L for the trade
                entry_multiplier = -1 if trade["signal"] in ("exit", "stop_loss", "time_stop") else 1
                leader_pnl = trade["leader_qty"] * (trade["leader_price"] if trade["leader_side"] == "sell" else -trade["leader_price"]) * entry_multiplier
                follower_pnl = trade["follower_qty"] * (trade["follower_price"] if trade["follower_side"] == "sell" else -trade["follower_price"]) * entry_multiplier
                trades.at[i, "profit"] = leader_pnl + follower_pnl - trade["fee"]
            
            wins = trades[trades["profit"] > 0]
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            avg_win = wins["profit"].mean() if len(wins) > 0 else 0
            losses = trades[trades["profit"] <= 0]
            avg_loss = losses["profit"].mean() if len(losses) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
            
            # Stop-loss analysis
            stop_loss_trades = trades[trades["signal"] == "stop_loss"]
            time_stop_trades = trades[trades["signal"] == "time_stop"]
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            stop_loss_trades = time_stop_trades = pd.DataFrame()
        
        return {
            "cumulative_return": cumulative_return,
            "annualized_volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trades) if self.trades else 0,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_value": final_value,
            "initial_value": initial_value,
            "stop_loss_count": len(stop_loss_trades),
            "time_stop_count": len(time_stop_trades)
        }