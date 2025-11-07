import sys
import itertools
import time


def wait_for_ready(is_ready_callback: callable, object_waited_str: str = "X", timeout=None):
    """Wait for the object to be ready.
    Args:
        is_ready_callback: Callback to check if the object is ready.
        object_waited_str: String to display in the waiting message.
        timeout: Maximum time to wait in seconds. If None, waits indefinitely.
    """
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    start_time = time.time()

    sys.stdout.write(f"Waiting for {object_waited_str} to be ready ")
    sys.stdout.flush()

    while not is_ready_callback():
        if timeout is not None and (time.time() - start_time) > timeout:
            sys.stdout.write("\n")
            raise TimeoutError(f"{object_waited_str} did not become ready within {timeout} seconds")

        sys.stdout.write(f"\rWaiting for {object_waited_str} to be ready {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)

    sys.stdout.write(f"\rWaiting for {object_waited_str} to be ready... done\n")
    sys.stdout.flush()
