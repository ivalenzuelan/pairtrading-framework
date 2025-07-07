#!/usr/bin/env python3
"""
Individual pair trading logic and signal generation
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class PairTrader:
    """Manages trading for a single cointegrated pair"""
    
    def __init__(self, pair_info: dict, config: dict, interval: str, log: logging.Logger):
        """
        Initialize pair trader
        
        Args:
            pair_info: Dictionary containing pair information from screening
            config: Configuration dictionary
            interval: Trading interval
            log: Logger instance
        """
        # Initialize from pair screening results
        self.leader = pair_info["leader"]
        self.follower = pair_info["follower"]
        self.hedge_ratio = pair_info["hedge_ratio"]
        self.intercept = pair_info["intercept"]
        self.spread_mean = pair_info["mean"]
        self.spread_std = pair_info["std"]
        self.s1_vol = pair_info["s1_vol"]
        self.s2_vol = pair_info["s2_vol"]
        
        # Store configuration and dependencies
        self.config = config
        self.interval = interval
        self.log = log
        
        # Initialize trading state
        self.position = None  # 'long', 'short', or None
        self.entry_time = None
        self.entry_price = None  # (leader_price, follower_price)
        self.entry_z = None
        self.trade_history = []
        
    def calculate_spread(self, leader_price: float, follower_price: float) -> float:
        """
        Calculate RATIO spread (methodology compliant)
        
        Args:
            leader_price: Current leader price
            follower_price: Current follower price
            
        Returns:
            Calculated spread value
        """
        return leader_price / (self.hedge_ratio * follower_price)
    
    def calculate_z_score(self, spread: float) -> float:
        """
        Calculate z-score of spread
        
        Args:
            spread: Current spread value
            
        Returns:
            Z-score of the spread
        """
        return (spread - self.spread_mean) / self.spread_std if self.spread_std > 0 else 0
    
    def check_signal(self, z_score: float, timestamp: pd.Timestamp) -> Optional[str]:
        """
        Generate trading signals based on Bollinger Bands strategy
        
        Args:
            z_score: Current z-score of spread
            timestamp: Current timestamp
            
        Returns:
            Signal string or None if no signal
        """
        # Stop-loss conditions
        if self.position and self.entry_time is not None:
            holding_days = (timestamp - self.entry_time).days
            
            # Z-score stop-loss
            if (self.position == "long" and z_score <= -self.config["stop_loss_z"]) or \
               (self.position == "short" and z_score >= self.config["stop_loss_z"]):
                return "stop_loss"
            
            # Time-based stop-loss
            if holding_days >= self.config["buffer_period"]:
                return "time_stop"
        
        # Exit conditions
        if self.position:
            # Exit when |Z| < 0.5σ (methodology compliant)
            if abs(z_score) < self.config["exit_z"]:
                return "exit"
        
        # Entry conditions (only if not in position)
        if not self.position:
            if z_score <= -self.config["entry_z"]:
                return "long"
            elif z_score >= self.config["entry_z"]:
                return "short"
        
        return None
    
    def execute_trade(self, signal: str, timestamp: pd.Timestamp, 
                      leader_price: float, follower_price: float, 
                      current_z_score: float) -> dict:
        """
        Execute trade with volatility-scaled position sizing
        
        Args:
            signal: Trading signal
            timestamp: Current timestamp
            leader_price: Current leader price
            follower_price: Current follower price
            current_z_score: Current z-score
            
        Returns:
            Dictionary containing trade details
        """
        # Calculate position sizing
        position_info = self._calculate_position_sizing(leader_price, follower_price)
        
        # Determine trade directions
        trade_sides = self._determine_trade_sides(signal)
        
        # Apply slippage
        execution_prices = self._apply_slippage(
            leader_price, follower_price, trade_sides["leader"], trade_sides["follower"]
        )
        
        # Calculate costs and fees
        costs = self._calculate_trade_costs(position_info, execution_prices)
        
        # Create trade record
        trade = self._create_trade_record(
            signal, timestamp, current_z_score, position_info, 
            trade_sides, execution_prices, costs
        )
        
        # Update trader state
        self._update_trader_state(signal, timestamp, leader_price, follower_price, current_z_score)
        
        # Store trade
        self.trade_history.append(trade)
        
        return trade
    
    def _calculate_position_sizing(self, leader_price: float, follower_price: float) -> Dict[str, float]:
        """Calculate volatility-based position sizing"""
        # Volatility-based position sizing (methodology compliant)
        vol_weights = {
            self.leader: 1 / max(self.s1_vol, 1e-6),
            self.follower: 1 / max(self.s2_vol, 1e-6)
        }
        total_weight = sum(vol_weights.values())
        
        # Calculate allocation per leg
        base_alloc = self.config["base_allocation"]
        leader_alloc = base_alloc * vol_weights[self.leader] / total_weight
        follower_alloc = base_alloc * vol_weights[self.follower] / total_weight
        
        # Determine quantities
        leader_qty = leader_alloc / leader_price
        follower_qty = follower_alloc / follower_price
        
        return {
            "leader_qty": leader_qty,
            "follower_qty": follower_qty,
            "leader_alloc": leader_alloc,
            "follower_alloc": follower_alloc
        }
    
    def _determine_trade_sides(self, signal: str) -> Dict[str, str]:
        """Determine buy/sell directions for each asset"""
        if signal == "long":
            # Buy spread: Buy leader, Sell follower
            self.position = "long"
            return {"leader": "buy", "follower": "sell"}
        elif signal == "short":
            # Sell spread: Sell leader, Buy follower
            self.position = "short"
            return {"leader": "sell", "follower": "buy"}
        else:  # Exit or stop-loss
            # Reverse current position
            leader_side = "sell" if self.position == "long" else "buy"
            follower_side = "buy" if self.position == "long" else "sell"
            self.position = None
            return {"leader": leader_side, "follower": follower_side}
    
    def _apply_slippage(self, leader_price: float, follower_price: float, 
                       leader_side: str, follower_side: str) -> Dict[str, float]:
        """Apply slippage to execution prices"""
        slippage = self.config["slippage_rate"]
        
        leader_exec = (leader_price * (1 + slippage) if leader_side == "buy" 
                      else leader_price * (1 - slippage))
        follower_exec = (follower_price * (1 + slippage) if follower_side == "buy" 
                        else follower_price * (1 - slippage))
        
        return {
            "leader_price": leader_exec,
            "follower_price": follower_exec
        }
    
    def _calculate_trade_costs(self, position_info: Dict[str, float], 
                              execution_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate trading costs including fees"""
        leader_cost = position_info["leader_qty"] * execution_prices["leader_price"]
        follower_cost = position_info["follower_qty"] * execution_prices["follower_price"]
        fee = (leader_cost + follower_cost) * self.config["taker_fee"]
        
        return {
            "leader_cost": leader_cost,
            "follower_cost": follower_cost,
            "fee": fee
        }
    
    def _create_trade_record(self, signal: str, timestamp: pd.Timestamp, current_z_score: float,
                            position_info: Dict[str, float], trade_sides: Dict[str, str],
                            execution_prices: Dict[str, float], costs: Dict[str, float]) -> dict:
        """Create comprehensive trade record"""
        return {
            "timestamp": timestamp,
            "signal": signal,
            "position": self.position,
            "leader": self.leader,
            "leader_qty": position_info["leader_qty"],
            "leader_side": trade_sides["leader"],
            "leader_price": execution_prices["leader_price"],
            "follower": self.follower,
            "follower_qty": position_info["follower_qty"],
            "follower_side": trade_sides["follower"],
            "follower_price": execution_prices["follower_price"],
            "fee": costs["fee"],
            "z_score": self.entry_z if signal in ("exit", "stop_loss", "time_stop") else current_z_score,
            "vol_scaled": True,
            "leader_vol": self.s1_vol,
            "follower_vol": self.s2_vol
        }
    
    def _update_trader_state(self, signal: str, timestamp: pd.Timestamp, 
                           leader_price: float, follower_price: float, current_z_score: float):
        """Update trader's internal state"""
        if signal in ("long", "short"):
            self.entry_time = timestamp
            self.entry_price = (leader_price, follower_price)
            self.entry_z = current_z_score
        else:
            self.entry_time = None
            self.entry_price = None
            self.entry_z = None
    
    def get_current_state(self) -> dict:
        """Get current trader state for monitoring"""
        return {
            "pair": f"{self.leader}/{self.follower}",
            "position": self.position,
            "entry_time": self.entry_time,
            "entry_z": self.entry_z,
            "total_trades": len(self.trade_history),
            "hedge_ratio": self.hedge_ratio,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
            "leader_vol": self.s1_vol,
            "follower_vol": self.s2_vol
        }
    
    def get_trade_history(self) -> list:
        """Get complete trade history for this pair"""
        return self.trade_history.copy()
    
    def get_performance_summary(self) -> dict:
        """Calculate performance metrics for this pair"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Calculate P&L for each trade
        pnl_list = []
        for _, trade in trades_df.iterrows():
            # Calculate P&L based on position direction
            if trade["signal"] in ("exit", "stop_loss", "time_stop"):
                # Closing trade - reverse the entry position
                entry_multiplier = -1
            else:
                # Opening trade
                entry_multiplier = 1
            
            leader_pnl = (trade["leader_qty"] * 
                         (trade["leader_price"] if trade["leader_side"] == "sell" else -trade["leader_price"]) * 
                         entry_multiplier)
            follower_pnl = (trade["follower_qty"] * 
                           (trade["follower_price"] if trade["follower_side"] == "sell" else -trade["follower_price"]) * 
                           entry_multiplier)
            
            trade_pnl = leader_pnl + follower_pnl - trade["fee"]
            pnl_list.append(trade_pnl)
        
        trades_df["pnl"] = pnl_list
        
        # Calculate metrics
        total_pnl = trades_df["pnl"].sum()
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else np.inf
        
        # Calculate max drawdown
        cumulative_pnl = trades_df["pnl"].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = peak - cumulative_pnl
        max_drawdown = drawdown.max()
        
        return {
            "total_trades": len(trades_df),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "wins": len(wins),
            "losses": len(losses),
            "stop_loss_trades": len(trades_df[trades_df["signal"] == "stop_loss"]),
            "time_stop_trades": len(trades_df[trades_df["signal"] == "time_stop"])
        }
    
    def reset_trader(self):
        """Reset trader state for new trading period"""
        self.position = None
        self.entry_time = None
        self.entry_price = None
        self.entry_z = None
        # Note: We don't reset trade_history as it's used for performance analysis
    
    def is_active(self) -> bool:
        """Check if trader has an active position"""
        return self.position is not None
    
    def get_current_exposure(self, leader_price: float, follower_price: float) -> dict:
        """Calculate current exposure if position is active"""
        if not self.is_active():
            return {"leader_exposure": 0, "follower_exposure": 0, "total_exposure": 0}
        
        # Calculate notional values based on last trade
        if self.trade_history:
            last_trade = self.trade_history[-1]
            if last_trade["signal"] in ("long", "short"):
                leader_exposure = last_trade["leader_qty"] * leader_price
                follower_exposure = last_trade["follower_qty"] * follower_price
                total_exposure = abs(leader_exposure) + abs(follower_exposure)
                
                return {
                    "leader_exposure": leader_exposure,
                    "follower_exposure": follower_exposure,
                    "total_exposure": total_exposure,
                    "position_type": self.position
                }
        
        return {"leader_exposure": 0, "follower_exposure": 0, "total_exposure": 0}
    
    def __str__(self) -> str:
        """String representation of the trader"""
        return (f"PairTrader({self.leader}/{self.follower}, "
                f"position={self.position}, "
                f"trades={len(self.trade_history)})")
    
    def __repr__(self) -> str:
        """Detailed representation of the trader"""
        return (f"PairTrader(leader='{self.leader}', follower='{self.follower}', "
                f"hedge_ratio={self.hedge_ratio:.4f}, "
                f"position={self.position}, "
                f"trades={len(self.trade_history)})")


class PairTradingSignals:
    """Utility class for generating trading signals with different strategies"""
    
    @staticmethod
    def bollinger_bands_signal(z_score: float, entry_threshold: float = 2.0, 
                              exit_threshold: float = 0.5, stop_loss_threshold: float = 3.0,
                              current_position: Optional[str] = None) -> Optional[str]:
        """
        Generate signals based on Bollinger Bands strategy
        
        Args:
            z_score: Current z-score of spread
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop loss
            current_position: Current position ('long', 'short', or None)
            
        Returns:
            Signal string or None
        """
        # Stop-loss conditions
        if current_position:
            if (current_position == "long" and z_score <= -stop_loss_threshold) or \
               (current_position == "short" and z_score >= stop_loss_threshold):
                return "stop_loss"
        
        # Exit conditions
        if current_position and abs(z_score) < exit_threshold:
            return "exit"
        
        # Entry conditions
        if not current_position:
            if z_score <= -entry_threshold:
                return "long"
            elif z_score >= entry_threshold:
                return "short"
        
        return None
    
    @staticmethod
    def mean_reversion_signal(z_score: float, entry_threshold: float = 1.5,
                             exit_threshold: float = 0.0, stop_loss_threshold: float = 2.5,
                             current_position: Optional[str] = None) -> Optional[str]:
        """
        Generate signals based on mean reversion strategy
        
        Args:
            z_score: Current z-score of spread
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit (typically 0 for mean reversion)
            stop_loss_threshold: Z-score threshold for stop loss
            current_position: Current position ('long', 'short', or None)
            
        Returns:
            Signal string or None
        """
        # Stop-loss conditions
        if current_position:
            if (current_position == "long" and z_score <= -stop_loss_threshold) or \
               (current_position == "short" and z_score >= stop_loss_threshold):
                return "stop_loss"
        
        # Exit conditions (mean reversion to zero)
        if current_position:
            if (current_position == "long" and z_score >= exit_threshold) or \
               (current_position == "short" and z_score <= exit_threshold):
                return "exit"
        
        # Entry conditions
        if not current_position:
            if z_score <= -entry_threshold:
                return "long"
            elif z_score >= entry_threshold:
                return "short"
        
        return None