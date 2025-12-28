#!/usr/bin/env python3
"""
Test script to verify PyG2oAbortFlag (C++ monitor thread) doesn't block exit.
The _monitor thread is created by PyG2oAbortFlag in the C++ module.
"""

import sys
import os
import time
import threading

from pyslam.config import Config

config = Config()

# Try to import g2o and C++ modules
try:
    import g2o

    G2O_AVAILABLE = True
except ImportError:
    print("⚠️  g2o not available, skipping g2o.Flag test")
    G2O_AVAILABLE = False

try:
    from pyslam.slam.cpp import cpp_module, CPP_AVAILABLE
except ImportError:
    CPP_AVAILABLE = False
    print("⚠️  C++ module not available")


def test_g2o_flag_monitor_thread():
    """Test if g2o.Flag creates monitor threads that block exit"""

    if not G2O_AVAILABLE:
        print("Skipping g2o.Flag test (g2o not available)")
        return True

    print("=" * 60)
    print("Testing g2o.Flag Monitor Thread")
    print("=" * 60)

    try:
        print("\n1. Creating g2o.Flag...")
        abort_flag = g2o.Flag()
        print(f"   Created g2o.Flag: {abort_flag}")

        print("\n2. Checking threads before using flag in C++...")
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_before = [t for t in threads_before if "_monitor" in t.name]
        print(f"   Active threads: {len(threads_before)}")
        print(f"   Monitor threads: {len(monitor_threads_before)}")
        for t in monitor_threads_before:
            print(f"     - {t.name} (daemon={t.daemon})")

        # The monitor thread is created when PyG2oAbortFlag is instantiated
        # This happens when g2o.Flag is passed to C++ optimization functions
        # Let's check if just creating the flag creates a thread (it shouldn't)

        print("\n3. Setting flag value...")
        abort_flag.value = False
        print(f"   Flag value: {abort_flag.value}")

        time.sleep(0.1)  # Give time for any threads to start

        print("\n4. Checking threads after flag operations...")
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_after = [t for t in threads_after if "_monitor" in t.name]
        print(f"   Active threads: {len(threads_after)}")
        print(f"   Monitor threads: {len(monitor_threads_after)}")

        if monitor_threads_after:
            print("   ⚠️  WARNING: Monitor threads detected!")
            for t in monitor_threads_after:
                print(f"     - {t.name} (daemon={t.daemon}, alive={t.is_alive()})")
        else:
            print("   ✓ No monitor threads (expected - threads only created when flag used in C++)")

        print("\n5. Deleting flag...")
        del abort_flag
        import gc

        gc.collect()
        time.sleep(0.2)

        print("\n6. Checking threads after deletion...")
        threads_final = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_final = [t for t in threads_final if "_monitor" in t.name]
        print(f"   Monitor threads: {len(monitor_threads_final)}")

        if monitor_threads_final:
            print("   ⚠️  WARNING: Monitor threads still running after deletion!")
            return False
        else:
            print("   ✓ No monitor threads remaining")
            return True

    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cpp_optimizer_with_abort_flag():
    """Test if using g2o.Flag in C++ optimizer creates monitor threads that don't exit"""

    if not G2O_AVAILABLE or not CPP_AVAILABLE:
        print("Skipping C++ optimizer test (g2o or C++ module not available)")
        return True

    print("\n" + "=" * 60)
    print("Testing C++ Optimizer with g2o.Flag")
    print("=" * 60)

    try:
        print("\n1. Creating g2o.Flag for C++ optimizer...")
        abort_flag = g2o.Flag()
        abort_flag.value = False

        print("\n2. Checking threads before C++ optimization...")
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_before = [t for t in threads_before if "_monitor" in t.name]
        print(f"   Monitor threads: {len(monitor_threads_before)}")

        # Test: Actually call a C++ function that uses the flag
        # We'll use g2o.SparseOptimizer.set_force_stop_flag which should trigger PyG2oAbortFlag
        print("\n3. Calling g2o.SparseOptimizer.set_force_stop_flag (triggers PyG2oAbortFlag)...")
        opt = g2o.SparseOptimizer()
        opt.set_force_stop_flag(abort_flag)

        # Give time for monitor thread to start
        time.sleep(0.1)

        print("\n4. Checking threads after setting flag in optimizer...")
        threads_during = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_during = [t for t in threads_during if "_monitor" in t.name]
        print(f"   Monitor threads: {len(monitor_threads_during)}")

        if monitor_threads_during:
            print("   ⚠️  Monitor thread created (this is expected when flag is used)")
            for t in monitor_threads_during:
                print(f"     - {t.name} (daemon={t.daemon})")

        print("\n5. Cleaning up optimizer and flag...")
        del opt
        del abort_flag
        import gc

        gc.collect()
        time.sleep(0.5)  # Give more time for C++ destructor to run

        print("\n6. Checking threads after cleanup...")
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread()]
        monitor_threads_after = [t for t in threads_after if "_monitor" in t.name]
        print(f"   Monitor threads: {len(monitor_threads_after)}")

        if monitor_threads_after:
            print("   ⚠️  WARNING: Monitor threads still running after cleanup!")
            for t in monitor_threads_after:
                print(f"     - {t.name} (daemon={t.daemon}, alive={t.is_alive()})")
            return False
        else:
            print("   ✓ No monitor threads remaining (cleanup successful)")
            return True

    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_threads_on_exit():
    """Test what threads remain when process tries to exit"""

    print("\n" + "=" * 60)
    print("Testing All Threads on Exit")
    print("=" * 60)

    print("\n1. Checking all active threads...")
    all_threads = [t for t in threading.enumerate() if t != threading.main_thread()]
    print(f"   Total active threads: {len(all_threads)}")

    for t in all_threads:
        daemon_status = "daemon" if t.daemon else "non-daemon"
        print(f"     - {t.name} ({daemon_status})")

    # Check for problematic threads
    monitor_threads = [t for t in all_threads if "_monitor" in t.name]
    queue_threads = [t for t in all_threads if "QueueFeederThread" in t.name]
    non_daemon = [t for t in all_threads if not t.daemon]

    print(f"\n2. Thread analysis:")
    print(f"   Monitor threads: {len(monitor_threads)}")
    print(f"   QueueFeederThread: {len(queue_threads)}")
    print(f"   Non-daemon threads: {len(non_daemon)}")

    if non_daemon:
        print("\n   ⚠️  WARNING: Non-daemon threads detected (these can block exit):")
        for t in non_daemon:
            print(f"     - {t.name}")
        return False
    else:
        print("\n   ✓ All threads are daemon threads (won't block exit)")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyG2oAbortFlag Monitor Thread Test")
    print("=" * 60)

    results = []

    # Test 1: g2o.Flag basic usage
    if G2O_AVAILABLE:
        results.append(("g2o.Flag basic", test_g2o_flag_monitor_thread()))

    # Test 2: C++ optimizer with flag
    if G2O_AVAILABLE and CPP_AVAILABLE:
        results.append(("C++ optimizer flag", test_cpp_optimizer_with_abort_flag()))

    # Test 3: All threads check
    results.append(("All threads check", test_all_threads_on_exit()))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ All tests passed!")
        print("Note: Monitor threads are only created when g2o.Flag is used in C++ code.")
        print("If your app has _monitor threads, they're likely from active C++ optimizations.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed.")
        print("Monitor threads may be blocking exit.")
        sys.exit(1)
