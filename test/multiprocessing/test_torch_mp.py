import torch
import torch.multiprocessing as mp

def worker(rank, device):
    # Move tensors to the specified device inside the worker
    tensor = torch.ones(10).to(device)
    print(f"Worker {rank} on {device}: {tensor}")

if __name__ == "__main__":
    mp.set_start_method("spawn")  # Use 'spawn' to avoid CUDA issues
    num_gpus = torch.cuda.device_count()
    
    processes = []
    for rank in range(num_gpus):
        # Create a new process with arguments that are pickle-able
        device = f"cuda:{rank}"
        p = mp.Process(target=worker, args=(rank, device))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()