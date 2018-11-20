import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_it
import random


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class CamVid(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, csv_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = glob.glob(os.path.join(image_path, '*.png'))
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        self.label_list = [os.path.join(label_path, x + '_L.png') for x in self.image_name]
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)
        # resize
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_list[index])

        # random crop image
        # =====================================
        # w,h = img.size
        # th, tw = self.scale
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)
        # img = F.crop(img, i, j, th, tw)
        # =====================================

        # resize image
        # =====================================
        img = self.resize_img(img)
        # =====================================

        img = np.array(img)

        # load label
        label = Image.open(self.label_list[index])

        # crop the corresponding label
        # =====================================
        # label = F.crop(label, i, j, th, tw)
        # =====================================

        # resize the corresponding label
        # =====================================
        label = self.resize_label(label)
        # =====================================

        label = np.array(label)

        # convert label to one-hot graph
        label = one_hot_it(label, self.label_info).astype(np.uint8)

        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)

        # resize image and label
        # resize_det = self.resize.to_deterministic()
        # img = resize_det.augment_image(img)
        # label = resize_det.augment_image(label)

        # image -> [C, H, W]
        img = Image.fromarray(img).convert('RGB')
        img = self.to_tensor(img).float()

        # label -> [num_classes, H, W]
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)
        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data = CamVid('/path/to/CamVid/train', '/path/to/CamVid/train_labels', '/path/to/CamVid/class_dict.csv', (640, 640))
    from model.build_BiSeNet import BiSeNet
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info('/path/to/CamVid/class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(img.shape)

