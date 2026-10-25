import torch
import torch.multiprocessing as mp

class SimpleModel(torch.nn.Module):
    def __init__(self, device=None):
        super(SimpleModel, self).__init__()
        self.model = torch.nn.Linear(10, 1)
        if device is None:
            if torch.cuda.is_available():
                print('SimpleModel - Using GPU')
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print('SimpleModel - Using MPS')
                self.device = torch.device("mps")
            else:
                print('SimpleModel - Using CPU')
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.model = self.model.to(self.device)        
        
    def forward(self, x):
        return self.model(x)

def train(rank, model, data, target, epochs, device):
    model = model.to(device)  # Send model to device
    data, target = data.to(device), target.to(device)
    
    # Each process creates its own optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Process {rank}, Epoch {epoch}, Loss: {loss.item()}")

def main():
    num_processes = 4
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    
    model = SimpleModel()  # No need to call model.share_memory() for simple data parallelism
    device = model.device
    model.share_memory()
    
    processes = []
    for rank in range(num_processes):
        # Each process gets its own target, device, and model instance
        p = mp.Process(target=train, args=(rank, model, data, target, 10, device))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()