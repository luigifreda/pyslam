#!/usr/bin/env python3
"""
Simple test to verify that LoggerQueue doesn't block process exit.
Run this and check if it exits cleanly (should exit within a few seconds).
"""

import sys
import os
import time
import threading
import tempfile

from pyslam.config import Config

config = Config()

from pyslam.utilities.logging import LoggerQueue
import logging


def main():
    print("Creating LoggerQueue and logging messages...")

    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        # Create logger queue
        logger_queue = LoggerQueue.get_instance(log_file)
        logger = logger_queue.get_logger("test")

        # Log some messages
        for i in range(20):
            logger.info(f"Test message {i}")
            logger.warning(f"Warning {i}")

        print("Messages logged. Waiting a moment for processing...")
        time.sleep(0.5)

        # Check threads before shutdown
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_threads_before = [t for t in threads_before if "QueueFeederThread" in t.name]
        print(
            f"Threads before shutdown: {len(threads_before)} (QueueFeederThread: {len(queue_threads_before)})"
        )

        print("Stopping LoggerQueue...")
        logger_queue.stop_listener()

        # Check threads after shutdown
        time.sleep(0.3)
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread()]
        queue_threads_after = [t for t in threads_after if "QueueFeederThread" in t.name]
        print(
            f"Threads after shutdown: {len(threads_after)} (QueueFeederThread: {len(queue_threads_after)})"
        )

        if queue_threads_after:
            print(f"⚠️  WARNING: {len(queue_threads_after)} QueueFeederThread threads still running")
            for t in queue_threads_after:
                print(f"   - {t.name} (daemon={t.daemon})")
        else:
            print("✓ All QueueFeederThread threads cleaned up")

        print("\nProcess should exit now. If it hangs, LoggerQueue is blocking exit.")
        print("Waiting 2 seconds to verify exit...")
        time.sleep(2)

        print("✓ Process exiting cleanly!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except Exception:
            pass


if __name__ == "__main__":
    main()
    # If we reach here, the process should exit normally
    # If it hangs, LoggerQueue is likely blocking exit
