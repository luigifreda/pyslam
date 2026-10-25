import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from typing import List
from tqdm.auto import tqdm
import torch.multiprocessing as mp


class ImageDataset(data.Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.mytransform = self.input_transform()
        self.images = imgs

    def __getitem__(self, index):
        img = self.images[index]
        img = self.mytransform(img)
        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class CosPlaceFeatureExtractor(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                print('CosPlaceFeatureExtractor - Using GPU')
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print('CosPlaceFeatureExtractor - Using MPS')
                self.device = torch.device("mps")
            else:
                print('CosPlaceFeatureExtractor - Using CPU')
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                                    backbone="ResNet50", fc_output_dim=2048)
        self.dim = 2048
        self.model = self.model.to(self.device)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        use_cuda = (self.device.type == "cuda")
        img_set = ImageDataset(imgs)
        test_data_loader = DataLoader(dataset=img_set, num_workers=4,
                                      batch_size=4, shuffle=False,
                                      pin_memory=use_cuda)
        self.model.eval()
        with torch.no_grad():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            test_data_ = tqdm(test_data_loader) if len(imgs) > 1 else test_data_loader
            for input_data, indices in test_data_:
                indices_np = indices.numpy()
                input_data = input_data.to(self.device)
                image_encoding = self.model(input_data)
                global_feats[indices_np, :] = image_encoding.cpu().numpy()
        return global_feats


def process_images(rank, imgs):
    # Each process creates its own feature extractor
    feature_extractor = CosPlaceFeatureExtractor()
    features = feature_extractor.compute_features(imgs)
    print(f"Process {rank} completed.")


def main():
    mp.set_start_method('spawn', force=True)  # Required for CUDA multiprocessing on some platforms
    num_processes = 1
    # Sample images - replace with actual image data in np.ndarray format
    sample_images = [np.random.rand(224, 224, 3) for _ in range(100)]

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=process_images, args=(rank, sample_images))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
