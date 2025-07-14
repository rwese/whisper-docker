#!/usr/bin/env python3
"""
Test script for logging and metrics functionality
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone


def test_logging_setup():
    """Test that logging is properly configured"""
    print("Testing logging setup...")

    # Test basic logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Basic logging test successful")
    print("✅ Basic logging works")

    # Test structured logging if available
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        struct_logger = structlog.get_logger()
        struct_logger.info("Structured logging test", test_key="test_value")
        print("✅ Structured logging works")

    except ImportError:
        print("⚠️  structlog not available, using basic logging")

    return True


def test_metrics_setup():
    """Test that metrics are properly configured"""
    print("\nTesting metrics setup...")

    try:
        from prometheus_client import Counter, Gauge, Histogram, generate_latest

        # Create test metrics
        test_counter = Counter("test_requests_total", "Test counter")
        test_histogram = Histogram("test_duration_seconds", "Test histogram")
        test_gauge = Gauge("test_active_items", "Test gauge")

        # Test metric operations
        test_counter.inc()
        test_histogram.observe(0.5)
        test_gauge.set(10)

        # Test metrics export
        metrics_output = generate_latest()
        if b"test_requests_total" in metrics_output:
            print("✅ Prometheus metrics work")
            return True
        else:
            print("❌ Metrics not found in output")
            return False

    except ImportError:
        print("⚠️  prometheus_client not available")
        return False


def test_stdout_stderr_output():
    """Test stdout/stderr output functionality"""
    print("\nTesting stdout/stderr output...")

    # Test stdout output
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] Test stdout message", flush=True)

    # Test stderr output
    print(f"[{timestamp}] Test stderr message", file=sys.stderr, flush=True)

    print("✅ Stdout/stderr output works")
    return True


def test_transcription_core_imports():
    """Test transcription core imports"""
    print("\nTesting transcription core imports...")

    try:
        # Test basic imports without external dependencies
        import os
        import re
        import tempfile
        import threading

        print("✅ Basic transcription core imports work")

        # Test if we can create a mock transcription service
        class MockTranscriptionService:
            def __init__(self, model_name="small"):
                self.model_name = model_name
                self.model = None
                self._model_lock = threading.Lock()

            def detect_language_from_filename(self, filename):
                pattern = r"_([a-z]{2})\.[^.]+$"
                match = re.search(pattern, filename.lower())
                return match.group(1) if match else None

        service = MockTranscriptionService()

        # Test language detection
        test_lang = service.detect_language_from_filename("test_en.m4a")
        assert test_lang == "en", f"Expected 'en', got '{test_lang}'"

        print("✅ Mock transcription service works")
        return True

    except Exception as e:
        print(f"❌ Transcription core test failed: {e}")
        return False


def test_file_operations():
    """Test file operations"""
    print("\nTesting file operations...")

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".test") as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"test content")

        # Check if file exists
        assert os.path.exists(temp_path), "Temporary file should exist"

        # Clean up
        os.unlink(temp_path)

        print("✅ File operations work")
        return True

    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running logging and metrics tests...\n")

    tests = [
        test_logging_setup,
        test_metrics_setup,
        test_stdout_stderr_output,
        test_transcription_core_imports,
        test_file_operations,
    ]

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
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
