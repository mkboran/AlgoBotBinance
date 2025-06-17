#!/usr/bin/env python3
"""
🚀 MOMENTUM ML TRADING SYSTEM - ML vs NON-ML PERFORMANCE COMPARISON
🔥 CRITICAL PROJECT: ML Enhancement Performance Analysis

This script runs comprehensive backtests comparing ML-enabled vs ML-disabled strategies
to measure the exact performance improvement from ML integration.

🎯 TARGET METRICS:
- Profit increase: +35-50% with ML
- Sharpe ratio: >2.0 with ML
- Max drawdown: <15% with ML
- Win rate: >65% with ML
- Profit factor: >2.5 with ML
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import argparse

# Import your trading system components
from utils.config import settings
from utils.portfolio import Portfolio
from utils.logger import logger
from strategies.momentum_optimized import EnhancedMomentumStrategy
try:
    from backtest_runner import MomentumBacktester
except ImportError:
    # Alternative import path if needed
    try:
        import backtest_runner
        MomentumBacktester = backtest_runner.MomentumBacktester
    except ImportError:
        print("⚠️ MomentumBacktester not found. Check backtest_runner.py path.")
        MomentumBacktester = None

# Configuration
INITIAL_CAPITAL = 1000.0
DEFAULT_DATA_PATH = Path("historical_data") / "BTCUSDT_15m_20210101_20241231.csv"
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", DEFAULT_DATA_PATH)

class MLPerformanceComparator:
    """🧠 ML vs Non-ML Performance Analysis Engine"""
    
    def __init__(self, data_path: str, initial_capital: float = 1000.0):
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.results = {}
        
    async def run_backtest_scenario(self, 
                                   scenario_name: str,
                                   ml_enabled: bool,
                                   start_date: str,
                                   end_date: str,
                                   strategy_params: Dict = None) -> Dict:
        """Run a single backtest scenario"""
        
        logger.info(f"🚀 Running {scenario_name} (ML: {'ON' if ml_enabled else 'OFF'})")
        
        # Create portfolio
        portfolio = Portfolio(initial_capital_usdt=self.initial_capital)
        
        # Prepare strategy parameters
        if strategy_params is None:
            strategy_params = {}
            
        # Force ML setting
        strategy_params['ml_enabled'] = ml_enabled
        
        # Create strategy instance
        strategy = EnhancedMomentumStrategy(portfolio=portfolio, **strategy_params)
        
        # Create backtester
        backtester = MomentumBacktester(
            csv_path=str(self.data_path),
            initial_capital=self.initial_capital,
            start_date=start_date,
            end_date=end_date,
            symbol=settings.SYMBOL,
            portfolio_instance=portfolio,
            strategy_instance=strategy
        )
        
        # Run backtest
        results = await backtester.run_backtest()
        
        # Add ML status to results
        results['ml_enabled'] = ml_enabled
        results['scenario_name'] = scenario_name
        
        # Get ML performance summary if available
        if hasattr(strategy, 'get_ml_performance_summary'):
            ml_summary = strategy.get_ml_performance_summary()
            results['ml_performance_summary'] = ml_summary
            
        logger.info(f"✅ {scenario_name} completed: "
                   f"Profit: {results.get('total_profit_pct', 0):.2f}%, "
                   f"Trades: {results.get('total_trades', 0)}, "
                   f"Sharpe: {results.get('sharpe_ratio', 0):.2f}")
        
        return results
    
    def calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        metrics = {
            'total_profit_pct': results.get('total_profit_pct', 0),
            'total_profit_usdt': results.get('total_profit_usdt', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'sortino_ratio': results.get('sortino_ratio', 0),
            'max_drawdown_pct': results.get('max_drawdown_pct', 0),
            'win_rate_pct': results.get('win_rate_pct', 0),
            'profit_factor': results.get('profit_factor', 0),
            'total_trades': results.get('total_trades', 0),
            'avg_trade_duration_hours': results.get('avg_trade_duration_hours', 0),
            'avg_profit_per_trade': results.get('avg_profit_per_trade', 0),
            'max_consecutive_wins': results.get('max_consecutive_wins', 0),
            'max_consecutive_losses': results.get('max_consecutive_losses', 0),
            'volatility_annualized': results.get('volatility_annualized', 0),
            'calmar_ratio': results.get('calmar_ratio', 0)
        }
        
        return metrics
    
    def compare_scenarios(self, ml_results: Dict, no_ml_results: Dict) -> Dict:
        """Compare ML vs Non-ML performance"""
        
        ml_metrics = self.calculate_performance_metrics(ml_results)
        no_ml_metrics = self.calculate_performance_metrics(no_ml_results)
        
        comparison = {
            'ml_metrics': ml_metrics,
            'no_ml_metrics': no_ml_metrics,
            'improvements': {},
            'summary': {}
        }
        
        # Calculate improvements
        for metric, ml_value in ml_metrics.items():
            no_ml_value = no_ml_metrics[metric]
            
            if no_ml_value != 0:
                improvement_pct = ((ml_value - no_ml_value) / abs(no_ml_value)) * 100
                improvement_abs = ml_value - no_ml_value
            else:
                improvement_pct = 0 if ml_value == 0 else float('inf')
                improvement_abs = ml_value
                
            comparison['improvements'][metric] = {
                'ml_value': ml_value,
                'no_ml_value': no_ml_value,
                'improvement_pct': improvement_pct,
                'improvement_abs': improvement_abs
            }
        
        # Generate summary
        profit_improvement = comparison['improvements']['total_profit_pct']['improvement_pct']
        sharpe_improvement = comparison['improvements']['sharpe_ratio']['improvement_pct']
        drawdown_change = comparison['improvements']['max_drawdown_pct']['improvement_pct']
        win_rate_improvement = comparison['improvements']['win_rate_pct']['improvement_pct']
        
        comparison['summary'] = {
            'profit_boost_pct': profit_improvement,
            'risk_adjusted_improvement': sharpe_improvement,
            'drawdown_change_pct': drawdown_change,
            'win_rate_boost_pct': win_rate_improvement,
            'ml_advantage': profit_improvement > 0 and sharpe_improvement > 0,
            'meets_targets': {
                'profit_increase_35pct': profit_improvement >= 35.0,
                'sharpe_above_2': ml_metrics['sharpe_ratio'] >= 2.0,
                'drawdown_below_15pct': ml_metrics['max_drawdown_pct'] <= 15.0,
                'win_rate_above_65pct': ml_metrics['win_rate_pct'] >= 65.0,
                'profit_factor_above_2_5': ml_metrics['profit_factor'] >= 2.5
            }
        }
        
        return comparison
    
    async def run_comprehensive_comparison(self, 
                                         test_periods: List[Tuple[str, str]] = None,
                                         strategy_params: Dict = None) -> Dict:
        """Run comprehensive ML vs Non-ML comparison across multiple periods"""
        
        if test_periods is None:
            # Default test periods
            test_periods = [
                ("2024-01-01", "2024-03-31"),  # Q1 2024
                ("2024-04-01", "2024-06-30"),  # Q2 2024
                ("2024-07-01", "2024-09-30"),  # Q3 2024
                ("2024-10-01", "2024-12-31"),  # Q4 2024
            ]
        
        logger.info(f"🚀 Starting Comprehensive ML vs Non-ML Comparison")
        logger.info(f"📅 Test periods: {len(test_periods)}")
        logger.info(f"💰 Initial capital: ${self.initial_capital}")
        logger.info(f"📊 Data source: {self.data_path}")
        
        all_results = {}
        
        for i, (start_date, end_date) in enumerate(test_periods):
            period_name = f"Period_{i+1}_{start_date}_to_{end_date}"
            
            logger.info(f"\n🔄 Testing Period {i+1}: {start_date} to {end_date}")
            
            # Run ML-enabled backtest
            ml_results = await self.run_backtest_scenario(
                scenario_name=f"{period_name}_ML_ENABLED",
                ml_enabled=True,
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy_params
            )
            
            # Run ML-disabled backtest
            no_ml_results = await self.run_backtest_scenario(
                scenario_name=f"{period_name}_ML_DISABLED", 
                ml_enabled=False,
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy_params
            )
            
            # Compare results
            comparison = self.compare_scenarios(ml_results, no_ml_results)
            
            all_results[period_name] = {
                'period': {'start': start_date, 'end': end_date},
                'ml_results': ml_results,
                'no_ml_results': no_ml_results,
                'comparison': comparison
            }
            
            # Log period summary
            profit_boost = comparison['summary']['profit_boost_pct']
            logger.info(f"📈 Period {i+1} Summary: "
                       f"ML Profit Boost: {profit_boost:+.1f}%, "
                       f"Sharpe: {ml_results.get('sharpe_ratio', 0):.2f} vs {no_ml_results.get('sharpe_ratio', 0):.2f}")
        
        # Calculate overall summary
        overall_summary = self.calculate_overall_summary(all_results)
        
        final_results = {
            'comparison_date': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'initial_capital': self.initial_capital,
            'test_periods': test_periods,
            'period_results': all_results,
            'overall_summary': overall_summary
        }
        
        return final_results
    
    def calculate_overall_summary(self, all_results: Dict) -> Dict:
        """Calculate overall performance summary across all periods"""
        
        ml_profits = []
        no_ml_profits = []
        ml_sharpes = []
        no_ml_sharpes = []
        profit_improvements = []
        
        targets_met_count = 0
        total_periods = len(all_results)
        
        for period_name, period_data in all_results.items():
            comparison = period_data['comparison']
            
            ml_profits.append(comparison['ml_metrics']['total_profit_pct'])
            no_ml_profits.append(comparison['no_ml_metrics']['total_profit_pct'])
            ml_sharpes.append(comparison['ml_metrics']['sharpe_ratio'])
            no_ml_sharpes.append(comparison['no_ml_metrics']['sharpe_ratio'])
            profit_improvements.append(comparison['summary']['profit_boost_pct'])
            
            # Count periods meeting targets
            meets_targets = comparison['summary']['meets_targets']
            if all(meets_targets.values()):
                targets_met_count += 1
        
        overall_summary = {
            'average_ml_profit_pct': np.mean(ml_profits),
            'average_no_ml_profit_pct': np.mean(no_ml_profits),
            'average_profit_improvement_pct': np.mean(profit_improvements),
            'average_ml_sharpe': np.mean(ml_sharpes),
            'average_no_ml_sharpe': np.mean(no_ml_sharpes),
            'consistency_score': len([x for x in profit_improvements if x > 0]) / len(profit_improvements),
            'targets_achievement_rate': targets_met_count / total_periods,
            'best_period_improvement_pct': max(profit_improvements),
            'worst_period_improvement_pct': min(profit_improvements),
            'ml_advantage_confirmed': np.mean(profit_improvements) > 0 and np.mean(ml_sharpes) > np.mean(no_ml_sharpes)
        }
        
        return overall_summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save comparison results to JSON file"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/ml_vs_noml_comparison_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {output_path}")
        
    def print_detailed_summary(self, results: Dict):
        """Print detailed summary to console"""
        
        overall = results['overall_summary']
        
        print("\n" + "="*80)
        print("🚀 MOMENTUM ML TRADING SYSTEM - PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
        print(f"   • Average ML Profit:        {overall['average_ml_profit_pct']:+.2f}%")
        print(f"   • Average Non-ML Profit:    {overall['average_no_ml_profit_pct']:+.2f}%")
        print(f"   • Average Improvement:      {overall['average_profit_improvement_pct']:+.2f}%")
        print(f"   • ML Advantage Confirmed:   {'✅ YES' if overall['ml_advantage_confirmed'] else '❌ NO'}")
        
        print(f"\n🎯 RISK-ADJUSTED METRICS:")
        print(f"   • Average ML Sharpe:        {overall['average_ml_sharpe']:.3f}")
        print(f"   • Average Non-ML Sharpe:    {overall['average_no_ml_sharpe']:.3f}")
        print(f"   • Consistency Score:        {overall['consistency_score']*100:.1f}%")
        
        print(f"\n🏆 TARGET ACHIEVEMENT:")
        print(f"   • Periods Meeting Targets:  {overall['targets_achievement_rate']*100:.1f}%")
        print(f"   • Best Period Improvement:  {overall['best_period_improvement_pct']:+.1f}%")
        print(f"   • Worst Period Improvement: {overall['worst_period_improvement_pct']:+.1f}%")
        
        # Target breakdown
        print(f"\n🎯 DETAILED TARGET ANALYSIS:")
        periods_data = results['period_results']
        target_names = {
            'profit_increase_35pct': '+35% Profit Increase',
            'sharpe_above_2': 'Sharpe Ratio >2.0',
            'drawdown_below_15pct': 'Max Drawdown <15%',
            'win_rate_above_65pct': 'Win Rate >65%',
            'profit_factor_above_2_5': 'Profit Factor >2.5'
        }
        
        for target_key, target_name in target_names.items():
            met_count = sum(1 for period_data in periods_data.values() 
                           if period_data['comparison']['summary']['meets_targets'][target_key])
            success_rate = met_count / len(periods_data) * 100
            print(f"   • {target_name:<25}: {met_count}/{len(periods_data)} periods ({success_rate:.1f}%)")
        
        print("\n" + "="*80)
        print("💡 CONCLUSION:")
        
        if overall['ml_advantage_confirmed'] and overall['average_profit_improvement_pct'] >= 35:
            print("🔥 ML ENHANCEMENT IS HIGHLY SUCCESSFUL!")
            print("   The ML system consistently outperforms the baseline strategy.")
            print("   Deploy with confidence for hedge fund level returns!")
        elif overall['ml_advantage_confirmed']:
            print("✅ ML ENHANCEMENT IS BENEFICIAL!")
            print("   The ML system shows positive improvement but below target.")
            print("   Consider further ML parameter optimization.")
        else:
            print("⚠️  ML ENHANCEMENT NEEDS OPTIMIZATION!")
            print("   The current ML configuration may need tuning.")
            print("   Review ML parameters and feature engineering.")
        
        print("="*80)

async def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="ML vs Non-ML Performance Comparison")
    parser.add_argument("--data-path", default=DATA_FILE_PATH, help="Path to historical data CSV")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--periods", type=str, help="Custom test periods as JSON")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Parse custom periods if provided
    test_periods = None
    if args.periods:
        try:
            test_periods = json.loads(args.periods)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for periods")
            return
    
    # Create comparator
    comparator = MLPerformanceComparator(
        data_path=args.data_path,
        initial_capital=args.capital
    )
    
    # Run comprehensive comparison
    logger.info("🚀 Starting ML vs Non-ML Performance Comparison")
    
    results = await comparator.run_comprehensive_comparison(
        test_periods=test_periods
    )
    
    # Save results
    comparator.save_results(results, args.output)
    
    # Print summary
    comparator.print_detailed_summary(results)
    
    logger.info("🏁 Comparison completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())