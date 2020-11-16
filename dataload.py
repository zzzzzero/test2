from torch.utils.data import dataset
from PIL import Image
from torchvision import transforms, models
import random
import numpy as np
import os
import torch
import cv2
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, Cutout, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, FancyPCA,IAAPiecewiseAffine)
from timm.data import transforms_factory
size = 380


auto_transform = transforms_factory.transforms_imagenet_train(img_size= size,auto_augment='rand')

train_transform = Compose(
        [
        Resize(height=size, width=size),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        HorizontalFlip(p=1.0),
        ShiftScaleRotate(shift_limit=0.2,scale_limit=0.2,rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_REFLECT_101,p=1.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
        ]
    )

val_transform = Compose(
        [
        Resize(height=(int(size / 0.875)), width=(int(size / 0.875))),
        CenterCrop(size, size,p=1.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2()
        ]
    )

trans = {
    'train':
        # transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ColorJitter(brightness=0.126, saturation=0.5),
        #     transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.8, 1.2), shear=None),
        #     transforms.Resize((int(size / 0.875), int(size / 0.875))),
        #     transforms.RandomCrop((size, size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        # ]),
        train_transform,
        # auto_transform,
    'val':
        # transforms.Compose([
        #     transforms.Resize((int(size / 0.875), int(size / 0.875))),
        #     transforms.CenterCrop((size, size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ]),
        val_transform,
    'test':
        # transforms.Compose([
        #     transforms.Resize((int(size / 0.875), int(size / 0.875))),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.RandomCrop((size, size)),
        #     transforms.CenterCrop((size, size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        Compose(
            [
                Resize(height=(int(size / 0.875)), width=(int(size / 0.875))),
                CenterCrop(size, size,p=1.0),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                ToTensorV2()
            ]
        )
}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1., use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    size = x.size()
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam

class Dataset(dataset.Dataset):
    def __init__(self, mode, root='./data'):
        assert mode in ['train', 'val']
        txt = './data/%s_denoise.txt' % mode

        fpath = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, label = i.strip().split(',')
                fpath.append(fp)
                labels.append(int(label))

        self.fpath = fpath
        self.labels = labels
        self.mode = mode
        self.trans = trans[mode]
        self.dataroot = root

    def __getitem__(self, index):
        fp = os.path.join(self.dataroot, self.fpath[index])
        label = self.labels[index]
        img = Image.open(fp).convert('RGB')
        img = np.array(img)
        if self.trans is not None:
            # img = self.trans(img)
            img = self.trans(image=img)["image"]

        return img, label

    def __len__(self):
        return len(self.labels)


class Testset(dataset.Dataset):
    def __init__(self, root='./data/test'):
        fnames = []
        for i in os.listdir(root):
            fnames.append(i)

        self.fnames = fnames
        self.trans = trans['test']
        self.dataroot = root

    def __getitem__(self, index):
        fn = self.fnames[index]
        fp = os.path.join(self.dataroot, fn)
        img = Image.open(fp).convert('RGB')
        img = np.array(img)
        if self.trans is not None:
            # img = self.trans(img)
            img = self.trans(image=img)["image"]

        return img, fn

    def __len__(self):
        return len(self.fnames)


def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))

    return images, labels


class PairWiseSet(dataset.Dataset):
    def __init__(self, mode='train'):
        txt = './data/Art/data/%s.txt' % mode
        fpath = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fp, idx = i.strip().split(',')
                fpath.append(fp)
                labels.append(int(idx))
        self.fpath = np.array(fpath)
        self.labels = np.array(labels)

        self.num_classes = len(set(self.labels))
        self.mode = mode
        self.trans = trans[mode]

    def __getitem__(self, index):
        data = []
        labels = []
        for i in range(self.num_classes):
            anchor = self._get_random_sample(i)
            anchor_imgs = self._get_images(anchor)
            data.extend(anchor_imgs)
            labels.append(i)

        return data, labels

    def _get_images(self, fpath):
        imgs = []
        for fp in fpath:
            img = Image.open(fp).convert('RGB')
            if self.trans is not None:
                img = self.trans(img)
            imgs.append(img)

        return imgs

    def _get_random_sample(self, label):
        fpath = list(self.fpath[np.argwhere(self.labels == label)].reshape(-1, ))

        return random.sample(fpath, k=1)

    def __len__(self):
        return int(len(self.labels) / self.num_classes) + 1


class Tripletset(dataset.Dataset):
    def __init__(self, mode):
        assert mode in ['train', 'val']
        txt = './data/%s.txt' % mode

        fnames = []
        labels = []
        with open(txt, 'r')as f:
            for i in f.readlines():
                fn, label, group = i.strip().split(',')
                fnames.append(fn)
                labels.append(int(label))

        self.fnames = fnames
        self.labels = labels
        self.dict_id_name = self.balance_train()
        self.mode = mode
        self.trans = trans[mode]
        self.dataroot = '/data/train'

    def __getitem__(self, index):
        label = self.labels[index]
        if len(self.dict_id_name[label]) == 1:
            anchor, pos = random.choices(self.dict_id_name[label], k=2)
        else:
            anchor, pos = random.sample(self.dict_id_name[label], k=2)
        neg_label = random.choice(list(set(self.labels) ^ set([label])))
        neg = random.choice(self.dict_id_name[neg_label])

        anchor_img = self._get_image(anchor)
        pos_img = self._get_image(pos)
        neg_img = self._get_image(neg)

        return [anchor_img, pos_img, neg_img], [label, label, neg_label]

    def balance_train(self):
        dict_id_name = {}
        for name, label in zip(self.fnames, self.labels):
            if not label in dict_id_name.keys():
                dict_id_name[label] = [name]
            else:
                dict_id_name[label].append(name)
        return dict_id_name

    def _get_image(self, fn):
        fp = os.path.join(self.dataroot, fn)
        img = Image.open(fp).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)
        return img

    def __len__(self):
        return int(len(self.labels) / 3) + 1