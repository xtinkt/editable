import torchvision.datasets as datasets
import numpy as np
import torch

'''
Dataset loader that returns batch of images and precomputed logits
'''

class ImageAndLogitsFolder(datasets.ImageFolder):
    
    def __init__(self, *args, logits_prefix, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_prefix = logits_prefix
    
    @staticmethod
    def logits_path_create(prefix, path):
        return prefix + path[path.rfind("/") + 1:path.find(".j")] + ".npy"
    
    def get_image_path(self, index):
        return self.imgs[index]
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        logits_path = ImageAndLogitsFolder.logits_path_create(
            self.logits_prefix,
            self.get_image_path(index)[0]
        )
        logits = np.load(logits_path)
        logits = torch.reshape(torch.Tensor(logits), (1, -1))
        
        return img, target, logits
