import multiprocessing
import threading
import time

from datetime import datetime

lst = [(2, 2), (4, 4), (5, 5), (6, 6), (3, 3)]
result = []


def mulX(x):
    print(f"Process {threading.current_thread().name} - start process {x}")
    time.sleep(3)
    print(f"Process {threading.current_thread().name} - end process {x}")
    res = x
    res_ap = (x, res)
    return res_ap


def worker_function():
    # Create a multiprocessing pool within the thread
    pool = multiprocessing.Pool(processes=10)

    # Use map or imap_unordered (if order doesn't matter) for parallel processing
    results = pool.imap_unordered(mulX, lst)

    # Process the results outside the pool
    for r in results:
        print(r)

    pool.close()
    pool.join()


def main_thread():
    # Create a thread
    thread = threading.Thread(target=worker_function)
    thread.start()

    # Other work in the main thread (can be removed if not needed)
    print("Doing other work in the main thread...")

    # Wait for the worker thread to finish
    thread.join()


if __name__ == "__main__":
    start = datetime.now()
    main_thread()
    print("End Time Apply Async:", (datetime.now() - start).total_seconds())