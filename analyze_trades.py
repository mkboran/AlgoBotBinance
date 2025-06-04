#!/usr/bin/env python3
# analyze_trades.py - Beautiful Trade Analysis Script
"""
🚀 Enhanced Trade Analysis Tool
Run this to get beautiful insights from your trading data!

Usage:
    python analyze_trades.py
    python analyze_trades.py --csv-path "custom/path/trades.csv"
"""

import argparse
from utils.trade_analyzer import TradeAnalyzer
from utils.logger import logger
from utils.config import settings

def main():
    parser = argparse.ArgumentParser(description="🚀 Enhanced Trade Analysis Tool")
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default=getattr(settings, 'TRADES_CSV_LOG_PATH', 'logs/trades.csv'),
        help="Path to the CSV trades file"
    )
    parser.add_argument(
        "--save-report", 
        action="store_true",
        help="Save analysis report to file"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Trade Analysis...")
    print(f"📁 Loading data from: {args.csv_path}")
    
    analyzer = TradeAnalyzer(args.csv_path)
    
    if analyzer.load_data():
        print("✅ Data loaded successfully!")
        analyzer.print_beautiful_summary()
        
        if args.save_report:
            analyzer.save_analysis_report()
            print("💾 Analysis report saved!")
    else:
        print("❌ Failed to load trade data")
        logger.error(f"Could not load data from {args.csv_path}")

if __name__ == "__main__":
    main()
