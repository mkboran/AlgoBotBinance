import re
from typing import List, Dict, Tuple

def parse_trial_results(log_file_path: str) -> List[Dict]:
    """Parse optimization trial results from log file"""
    trials = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match trial completion lines
    pattern = r'TRIAL (\d+) COMPLETED\. Metric \(([^)]+)\): ([0-9.]+), Profit: ([+-]?[0-9.]+)%, Trades: (\d+)'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        trial_num = int(match[0])
        metric_name = match[1]
        metric_value = float(match[2])
        profit_pct = float(match[3])
        trades = int(match[4])
        
        trials.append({
            'trial': trial_num,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'profit_pct': profit_pct,
            'trades': trades
        })
    
    return trials

def find_best_trial(trials: List[Dict]) -> Tuple[Dict, str]:
    """Find the best trial based on different criteria"""
    if not trials:
        return None, "No trials found"
    
    # Find best by metric value (assuming higher is better for most metrics)
    best_by_metric = max(trials, key=lambda x: x['metric_value'])
    
    # Find best by profit percentage
    best_by_profit = max(trials, key=lambda x: x['profit_pct'])
    
    # Find Trial 26 specifically
    trial_26 = next((t for t in trials if t['trial'] == 26), None)
    
    analysis = f"""
OPTIMIZATION TRIALS ANALYSIS
===========================

Total Trials Found: {len(trials)}

Best Trial by Metric ({best_by_metric['metric_name']}):
- Trial {best_by_metric['trial']}: {best_by_metric['metric_value']:.3f}
- Profit: {best_by_metric['profit_pct']:.2f}%
- Trades: {best_by_metric['trades']}

Best Trial by Profit:
- Trial {best_by_profit['trial']}: {best_by_profit['profit_pct']:.2f}%
- Metric: {best_by_profit['metric_value']:.3f}
- Trades: {best_by_profit['trades']}
"""

    if trial_26:
        is_best_metric = trial_26['trial'] == best_by_metric['trial']
        is_best_profit = trial_26['trial'] == best_by_profit['trial']
        
        analysis += f"""
TRIAL 26 RESULTS:
- Metric ({trial_26['metric_name']}): {trial_26['metric_value']:.3f}
- Profit: {trial_26['profit_pct']:.2f}%
- Trades: {trial_26['trades']}
- Is Best by Metric: {'YES ✅' if is_best_metric else 'NO ❌'}
- Is Best by Profit: {'YES ✅' if is_best_profit else 'NO ❌'}
"""
        
        # Rank Trial 26
        sorted_by_metric = sorted(trials, key=lambda x: x['metric_value'], reverse=True)
        sorted_by_profit = sorted(trials, key=lambda x: x['profit_pct'], reverse=True)
        
        metric_rank = next(i for i, t in enumerate(sorted_by_metric, 1) if t['trial'] == 26)
        profit_rank = next(i for i, t in enumerate(sorted_by_profit, 1) if t['trial'] == 26)
        
        analysis += f"""
TRIAL 26 RANKINGS:
- Rank by Metric: {metric_rank}/{len(trials)}
- Rank by Profit: {profit_rank}/{len(trials)}
"""
    else:
        analysis += "\nTRIAL 26: Not found in logs"
    
    # Show top 5 trials by metric
    top_5_metric = sorted(trials, key=lambda x: x['metric_value'], reverse=True)[:5]
    analysis += f"\nTOP 5 TRIALS BY METRIC:\n"
    for i, trial in enumerate(top_5_metric, 1):
        analysis += f"{i}. Trial {trial['trial']}: {trial['metric_value']:.3f} (Profit: {trial['profit_pct']:.2f}%)\n"
    
    # Show top 5 trials by profit
    top_5_profit = sorted(trials, key=lambda x: x['profit_pct'], reverse=True)[:5]
    analysis += f"\nTOP 5 TRIALS BY PROFIT:\n"
    for i, trial in enumerate(top_5_profit, 1):
        analysis += f"{i}. Trial {trial['trial']}: {trial['profit_pct']:.2f}% (Metric: {trial['metric_value']:.3f})\n"
    
    return trial_26, analysis

if __name__ == "__main__":
    log_file = r"c:\Projects\AlgoBotBinance\logs\algobot.log"
    
    print("Parsing trial results from log file...")
    trials = parse_trial_results(log_file)
    
    trial_26, analysis = find_best_trial(trials)
    print(analysis)
    
    # Save results to file
    with open(r"c:\Projects\AlgoBotBinance\trial_analysis.txt", 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    print("\nAnalysis saved to trial_analysis.txt")
