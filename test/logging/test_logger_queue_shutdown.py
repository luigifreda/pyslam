#!/usr/bin/env python3
"""
Test script to verify LoggerQueue properly shuts down and doesn't block exit.
This helps isolate whether LoggerQueue is causing the blocking issue.
"""

import sys
import os
import time
import threading

from pyslam.config import Config

config = Config()

from pyslam.utilities.logging import LoggerQueue
import logging
import tempfile


def test_logger_queue_shutdown():
    """Test that LoggerQueue properly shuts down without blocking"""

    print("=" * 60)
    print("Testing LoggerQueue Shutdown")
    print("=" * 60)

    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        print(f"\n1. Creating LoggerQueue with log file: {log_file}")
        logger_queue = LoggerQueue.get_instance(log_file)

        print("\n2. Getting logger and logging some messages...")
        logger = logger_queue.get_logger("test")

        # Log some messages
        for i in range(10):
            logger.info(f"Test message {i}")
            logger.warning(f"Warning message {i}")
            logger.error(f"Error message {i}")

        print("   Logged 30 messages")

        # Give time for messages to be processed
        time.sleep(0.5)

        print("\n3. Checking active threads before shutdown...")
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_feeder_threads_before = [t for t in threads_before if "QueueFeederThread" in t.name]
        print(f"   Active threads: {len(threads_before)}")
        print(f"   QueueFeederThread threads: {len(queue_feeder_threads_before)}")
        for t in queue_feeder_threads_before:
            print(f"     - {t.name} (daemon={t.daemon})")

        print("\n4. Calling stop_listener()...")
        start_time = time.time()
        logger_queue.stop_listener()
        stop_time = time.time() - start_time
        print(f"   stop_listener() completed in {stop_time:.3f} seconds")

        print("\n5. Checking active threads after shutdown...")
        time.sleep(0.2)  # Give threads time to exit
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_feeder_threads_after = [t for t in threads_after if "QueueFeederThread" in t.name]
        print(f"   Active threads: {len(threads_after)}")
        print(f"   QueueFeederThread threads: {len(queue_feeder_threads_after)}")

        if queue_feeder_threads_after:
            print("   ⚠️  WARNING: QueueFeederThread threads still running!")
            for t in queue_feeder_threads_after:
                print(f"     - {t.name} (daemon={t.daemon}, alive={t.is_alive()})")
        else:
            print("   ✓ No QueueFeederThread threads remaining")

        # Check if queue is closed
        print("\n6. Checking queue state...")
        if hasattr(logger_queue, "log_queue"):
            queue = logger_queue.log_queue
            if hasattr(queue, "_closed"):
                print(f"   Queue closed: {queue._closed}")
            else:
                print("   Queue closed attribute not available")

        print("\n7. Testing if process can exit cleanly...")
        print("   (If this message appears and process exits, shutdown is successful)")

        return len(queue_feeder_threads_after) == 0

    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up log file
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except Exception:
            pass


def test_multiple_logger_queues():
    """Test shutting down multiple LoggerQueue instances"""

    print("\n" + "=" * 60)
    print("Testing Multiple LoggerQueue Shutdown")
    print("=" * 60)

    log_files = []
    logger_queues = []

    try:
        print("\n1. Creating multiple LoggerQueue instances...")
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=f"_{i}.log") as f:
                log_file = f.name
                log_files.append(log_file)

            logger_queue = LoggerQueue.get_instance(log_file)
            logger_queues.append(logger_queue)
            logger = logger_queue.get_logger(f"test_{i}")
            logger.info(f"Message from logger {i}")

        print(f"   Created {len(logger_queues)} LoggerQueue instances")

        print("\n2. Checking threads before shutdown...")
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_feeder_threads_before = [t for t in threads_before if "QueueFeederThread" in t.name]
        print(f"   QueueFeederThread threads: {len(queue_feeder_threads_before)}")

        print("\n3. Stopping all LoggerQueue instances...")
        # Use the class method directly (LoggerQueue is already imported at module level)
        LoggerQueue.stop_all_instances()

        print("\n4. Checking threads after shutdown...")
        time.sleep(0.3)
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_feeder_threads_after = [t for t in threads_after if "QueueFeederThread" in t.name]
        print(f"   QueueFeederThread threads: {len(queue_feeder_threads_after)}")

        if queue_feeder_threads_after:
            print("   ⚠️  WARNING: QueueFeederThread threads still running!")
            for t in queue_feeder_threads_after:
                print(f"     - {t.name} (daemon={t.daemon})")
        else:
            print("   ✓ All QueueFeederThread threads cleaned up")

        return len(queue_feeder_threads_after) == 0

    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up log files
        for log_file in log_files:
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
            except Exception:
                pass


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LoggerQueue Shutdown Test")
    print("=" * 60)

    # Test single LoggerQueue
    result1 = test_logger_queue_shutdown()

    # Test multiple LoggerQueue instances
    result2 = test_multiple_logger_queues()

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Single LoggerQueue shutdown: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Multiple LoggerQueue shutdown: {'✓ PASS' if result2 else '✗ FAIL'}")

    if result1 and result2:
        print("\n✓ All tests passed! LoggerQueue shuts down cleanly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. LoggerQueue may be blocking exit.")
        sys.exit(1)
