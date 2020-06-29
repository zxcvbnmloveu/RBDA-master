from torchvision import datasets, transforms
import torch
import numpy as np

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    def __init__(self, mean=None, meanfile=None):
        if mean:
            self.mean = mean
        else:
            arr = np.load(meanfile)
            self.mean = torch.from_numpy(arr.astype('float32')/255.0)[[2,1,0],:,:]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.sub_(m)
        return tensor


def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         # transforms.RandomCrop(224),
         transforms.RandomCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    root = root_path + dir + '/images'
    print(root)
    data = datasets.ImageFolder(root=root_path + dir + '/images', transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        # [transforms.Resize([224, 224]),
        [transforms.Resize([227, 227]),

         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir + '/images', transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

if __name__ == '__main__':
    root_path = './data/OFFICE31/'
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    s_loader = load_training(root_path, 'amazon', 1, kwargs)
    iter(s_loader)