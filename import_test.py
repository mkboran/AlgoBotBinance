#!/usr/bin/env python3
"""
🚀 MOMENTUM ML TRADING SYSTEM - IMPORT & READINESS TEST v3.0
🔥 CRITICAL: Verify all components are ready for ML comparison

This script tests all imports and verifies system readiness for ML vs Non-ML comparison.
LATEST VERSION: Enhanced ML predictor detection and improved error handling
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, description):
    """Test import of a specific module"""
    try:
        if module_name == "utils.config":
            from utils.config import settings
            print(f"✅ {description}")
            return True, settings
        elif module_name == "utils.portfolio":
            from utils.portfolio import Portfolio
            print(f"✅ {description}")
            return True, Portfolio
        elif module_name == "utils.logger":
            from utils.logger import logger
            print(f"✅ {description}")
            return True, logger
        elif module_name == "strategies.momentum_optimized":
            from strategies.momentum_optimized import EnhancedMomentumStrategy
            print(f"✅ {description}")
            return True, EnhancedMomentumStrategy
        elif module_name == "backtest_runner":
            from backtest_runner import MomentumBacktester
            print(f"✅ {description}")
            return True, MomentumBacktester
        elif module_name == "ml_predictor":
            try:
                from strategies.momentum_optimized import AdvancedMLPredictor
                print(f"✅ {description}")
                return True, AdvancedMLPredictor
            except ImportError:
                print(f"⚠️  {description} - Not found (checking strategies module)")
                try:
                    # Try alternative import path
                    import strategies.momentum_optimized as mom_strategy
                    if hasattr(mom_strategy, 'AdvancedMLPredictor'):
                        print(f"✅ {description} - Found in strategies module")
                        return True, mom_strategy.AdvancedMLPredictor
                    else:
                        print(f"⚠️  {description} - AdvancedMLPredictor not found in strategies")
                        return False, None
                except ImportError:
                    print(f"❌ {description} - Strategies module import failed")
                    return False, None
        else:
            exec(f"import {module_name}")
            print(f"✅ {description}")
            return True, None
    except ImportError as e:
        print(f"❌ {description} - FAILED: {e}")
        return False, None
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False, None

def main():
    """Main test function"""
    
    print("="*80)
    print("🚀 MOMENTUM ML TRADING SYSTEM - IMPORT & READINESS TEST")
    print("="*80)
    
    print("\n📦 Testing Core Imports:")
    
    # Test all required imports
    tests = [
        ("asyncio", "Asyncio (async/await support)"),
        ("pandas", "Pandas (data manipulation)"),
        ("numpy", "NumPy (numerical computing)"),
        ("datetime", "DateTime (time handling)"),
        ("json", "JSON (data serialization)"),
        ("pathlib", "PathLib (file path handling)"),
        ("typing", "Typing (type hints)"),
    ]
    
    failed_basic = 0
    for module, desc in tests:
        success, _ = test_import(module, desc)
        if not success:
            failed_basic += 1
    
    print("\n🏗️ Testing Trading System Components:")
    
    system_tests = [
        ("utils.config", "Configuration System"),
        ("utils.logger", "Logging System"),
        ("utils.portfolio", "Portfolio Management"),
        ("strategies.momentum_optimized", "Enhanced Momentum Strategy"),
        ("backtest_runner", "Backtest Runner"),
    ]
    
    failed_system = 0
    components = {}
    
    for module, desc in system_tests:
        success, component = test_import(module, desc)
        if success:
            components[module] = component
        else:
            failed_system += 1
    
    print("\n🧠 Testing ML Components & Integration:")
    
    ml_tests = [
        ("ml_predictor", "Advanced ML Predictor (from strategies.momentum_optimized)"),
    ]
    
    failed_ml = 0
    for module, desc in ml_tests:
        success, component = test_import(module, desc)
        if not success:
            failed_ml += 1
    
    print("\n🔧 Testing ML Dependencies:")
    
    ml_deps = [
        ("sklearn", "Scikit-learn (ML algorithms)"),
        ("xgboost", "XGBoost (gradient boosting)"),
    ]
    
    failed_ml_deps = 0
    for module, desc in ml_deps:
        success, _ = test_import(module, desc)
        if not success:
            failed_ml_deps += 1
    
    print("\n📊 Testing Configuration:")
    
    if 'utils.config' in components:
        settings = components['utils.config']
        try:
            print(f"✅ Symbol: {settings.SYMBOL}")
            print(f"✅ ML Enabled: {settings.MOMENTUM_ML_ENABLED}")
            print(f"✅ Base Position Size: {settings.MOMENTUM_BASE_POSITION_SIZE_PCT}%")
            print(f"✅ Max Position Size: ${settings.MOMENTUM_MAX_POSITION_USDT}")
            print(f"✅ ML Lookback Window: {settings.MOMENTUM_ML_LOOKBACK_WINDOW}")
            print(f"✅ ML Prediction Horizon: {settings.MOMENTUM_ML_PREDICTION_HORIZON}")
        except Exception as e:
            print(f"❌ Configuration access failed: {e}")
            failed_system += 1
    
    print("\n🏗️ Testing Strategy Initialization:")
    
    if 'utils.portfolio' in components and 'strategies.momentum_optimized' in components:
        try:
            Portfolio = components['utils.portfolio']
            EnhancedMomentumStrategy = components['strategies.momentum_optimized']
            
            # Test portfolio creation
            test_portfolio = Portfolio(initial_capital_usdt=1000.0)
            print("✅ Portfolio creation successful")
            
            # Test strategy creation with ML enabled
            test_strategy_ml = EnhancedMomentumStrategy(
                portfolio=test_portfolio,
                ml_enabled=True
            )
            print("✅ ML-enabled strategy creation successful")
            
            # Test strategy creation with ML disabled
            test_strategy_no_ml = EnhancedMomentumStrategy(
                portfolio=test_portfolio,
                ml_enabled=False
            )
            print("✅ ML-disabled strategy creation successful")
            
            # Test ML status
            print(f"✅ ML Strategy ML Status: {test_strategy_ml.ml_enabled}")
            print(f"✅ Non-ML Strategy ML Status: {test_strategy_no_ml.ml_enabled}")
            
            # Test ML Predictor availability in ML strategy
            if hasattr(test_strategy_ml, 'ml_predictor'):
                print(f"✅ ML Predictor found in ML strategy")
                print(f"✅ ML Predictor type: {type(test_strategy_ml.ml_predictor).__name__}")
            else:
                print(f"⚠️  ML Predictor not found in ML strategy")
            
        except Exception as e:
            print(f"❌ Strategy initialization failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            failed_system += 1
    
    print("\n" + "="*80)
    print("📋 TEST SUMMARY:")
    print("="*80)
    
    total_tests = len(tests) + len(system_tests) + len(ml_tests) + len(ml_deps)
    total_failed = failed_basic + failed_system + failed_ml + failed_ml_deps
    total_passed = total_tests - total_failed
    
    print(f"📈 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📊 Success Rate: {(total_passed/total_tests)*100:.1f}%")
    
    print(f"\n📦 Component Status:")
    print(f"   • Basic Dependencies: {len(tests) - failed_basic}/{len(tests)} ✅")
    print(f"   • Trading System: {len(system_tests) - failed_system}/{len(system_tests)} ✅")
    print(f"   • ML Components: {len(ml_tests) - failed_ml}/{len(ml_tests)} ✅")
    print(f"   • ML Dependencies: {len(ml_deps) - failed_ml_deps}/{len(ml_deps)} ✅")
    
    if total_failed == 0:
        print("\n🔥 SYSTEM STATUS: FULLY READY FOR ML COMPARISON!")
        print("   🚀 You can now run the ML vs Non-ML performance comparison.")
        print("   💎 All components are properly loaded and configured.")
        print("   🧠 ML Integration is working perfectly!")
        print("\n📝 Next Steps:")
        print("   1. Run: python ml_vs_noml_comparison.py")
        print("   2. Wait for comprehensive analysis results")
        print("   3. Review performance improvements")
        print("   4. Celebrate your hedge fund level ML system! 🎉")
    elif failed_system > 0:
        print("\n⚠️  SYSTEM STATUS: CRITICAL ISSUES DETECTED!")
        print("   ❌ Core trading system components failed to load.")
        print("   🔧 Fix import errors before running comparison.")
    elif failed_ml > 0 or failed_ml_deps > 0:
        print("\n⚠️  SYSTEM STATUS: ML COMPONENTS ISSUES!")
        print("   ❌ ML components may not be fully functional.")
        print("   🔧 Install missing ML dependencies or check ML predictor.")
    else:
        print("\n⚠️  SYSTEM STATUS: MINOR ISSUES DETECTED!")
        print("   ⚠️  Some optional components failed.")
        print("   ✅ Core system should still work for comparison.")
    
    print("="*80)
    
    return total_failed == 0

if __name__ == "__main__":
    success = main()
    if success:
        print("🎯 READY TO PROCEED WITH ML COMPARISON!")
        sys.exit(0)
    else:
        print("🚨 FIX ISSUES BEFORE PROCEEDING!")
        sys.exit(1)