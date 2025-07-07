import timeit
from tqdm import tqdm


if __name__ == "__main__":
    
    num_iterations = 1000
    
    # Define a simple loop with range
    def simple_range_loop():
        for _ in range(num_iterations):
            pass

    # Define a loop with tqdm
    def tqdm_range_loop():
        for _ in tqdm(range(num_iterations)):
            pass

    # Measure the time taken by each loop
    simple_time = timeit.timeit(simple_range_loop, number=1)
    tqdm_time = timeit.timeit(tqdm_range_loop, number=1)

    print(f"Simple range loop time: {simple_time:.6f} seconds")
    print(f"tqdm range loop time: {tqdm_time:.6f} seconds")