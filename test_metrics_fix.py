#!/usr/bin/env python3
"""
Test script to verify the metrics fix works
"""

import os
import sys

sys.path.insert(0, ".")


def test_metrics_import():
    """Test that we can import the web service without metric conflicts"""
    try:
        print("Testing web service import...")
        import web_service

        print("✅ Web service imported successfully")

        # Test that metrics are properly initialized
        if web_service.metrics:
            print("✅ Metrics initialized successfully")
            print(f"Available metrics: {list(web_service.metrics.keys())}")
        else:
            print("⚠️  Metrics disabled (this is OK for multi-worker scenarios)")

        # Test safe metric update function
        print("Testing safe metric update...")
        web_service.safe_metric_update(
            "request_count", lambda: print("Metric update test")
        )
        print("✅ Safe metric update function works")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_multiple_imports():
    """Test that multiple imports don't cause conflicts"""
    try:
        print("\nTesting multiple imports...")
        import importlib

        # Reload the module to simulate multiple worker processes
        import web_service

        importlib.reload(web_service)
        print("✅ Module reload successful")

        # Import again
        import web_service as ws2

        print("✅ Second import successful")

        return True

    except Exception as e:
        print(f"❌ Multiple import test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running metrics fix tests...\n")

    tests = [test_metrics_import, test_multiple_imports]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Metrics fix is working.")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
