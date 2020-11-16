from torch.utils.data import dataset
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from PIL import Image

size = 32
trans = {
    'train':
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((size, size), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]),
    'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
}


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10(dataset.Dataset):
    def __init__(self, mode, root='./data/cifar-10-batches-py/'):
        assert mode in ['train', 'test'], print('mode must be "train" or "test"')
        data_root = root
        data_files = {'train': ['data_batch_%d' % x for x in range(1, 6)],
                      'test': ['test_batch']}
        self.imgs = None
        self.labels = []
        # self.class_names = self._unpickle(os.path.join(data_root, 'batches.meta'))[b'label_names]
        for f in data_files[mode]:
            data_dict = unpickle(os.path.join(data_root, f))
            data = data_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            if self.imgs is None:
                self.imgs = data
            else:
                self.imgs = np.vstack((self.imgs, data))
            self.labels += data_dict[b'labels']

        self.trans = trans[mode]

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        label = self.labels[index]
        img = self.trans(img)

        return img, label

    def __len__(self):
        return len(self.labels)


class CIFAR100(dataset.Dataset):
    def __init__(self, mode, root='./data/cifar-100-py'):
        super().__init__()
        self.data_root = root

        data = unpickle(os.path.join(self.data_root, mode))
        print(data.keys())
        self.fnames = data[b'filenames']
        self.imgs = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = data[b'fine_labels']
        self.corse = data[b'coarse_labels']
        self.trans = trans[mode]

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        label = self.labels[index]
        img = self.trans(img)

        return img, label

    def __len__(self):
        return len(self.labels)


def get_data(num_classes=10, root='./data/cifar-10-batches-py'):
    if num_classes == 10:
        Dataset = CIFAR10
    else:
        Dataset = CIFAR100

    trainset = Dataset(mode='train', root=root)
    testset = Dataset(mode='test', root=root)

    return trainset, testset


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    data = CIFAR10(mode='train', root='./data/cifar-10-batches-py')
    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)
    imgs, label = iter(loader).__next__()
    imshow(imgs)
    print(label)
