import torch
from torchvision.transforms import transforms
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, resize_size=None, resize_mode=Image.BILINEAR, rgb=False):
        super(torch.utils.data.Dataset, self).__init__()
        self.paths = paths
        self.transform = self._make_transforms(resize_size, resize_mode)
        self.rgb = rgb
        
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        if self.rgb:
            img = img.convert('RGB')
        else:
            img = img.convert('L')
        X = self.transform(img)
        return X
        
    def __len__(self):
        return len(self.paths)
    
    @staticmethod
    def _make_transforms(resize_size, resize_mode):
        transform = []
        if resize_size is not None:
            transform.append(transforms.Resize(resize_size, resize_mode))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform = transforms.Compose(transform)
        return transform
    
    
