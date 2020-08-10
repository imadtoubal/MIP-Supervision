from torch.utils.data import Dataset, DataLoader
import torch


class SegDataset3D(Dataset):
    def __init__(self, data, labels):
        """ 3D Medical Image Segmentation Dataset

        Args:
            data (list, tensor, or numpy array): Numpy array / PyTorch Tensor of shape (N, W, H, D) 
            or a list (of length N) of Numpy Arrays / Pytorch Tensors of shape (W, H, D)

            labels (list, tensor, or numpy array): Numpy array / PyTorch Tensor of shape (N, W, H, D) 
            or a list (of length N) of Numpy Arrays / Pytorch Tensors of shape (W, H, D)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float().unsqueeze(0), torch.tensor(self.labels[idx]).long()
