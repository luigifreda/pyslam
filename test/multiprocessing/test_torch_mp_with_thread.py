import torch
import torch.multiprocessing as mp
import threading

def worker(rank, device):
    # Initialize a CUDA tensor in each subprocess
    device = torch.device(device)
    tensor = torch.ones(10, device=device)
    print(f"Worker {rank} on {device}: {tensor}")

def start_process_in_thread(rank, device):
    # Spawn a subprocess from within a thread
    p = mp.Process(target=worker, args=(rank, device))
    p.start()
    p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Required for CUDA and multiprocessing
    num_gpus = torch.cuda.device_count()
    
    threads = []
    for rank in range(num_gpus):
        device = f"cuda:{rank}"
        # Start a new thread that will launch a subprocess
        t = threading.Thread(target=start_process_in_thread, args=(rank, device))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()