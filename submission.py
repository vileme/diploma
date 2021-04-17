import argparse
import h5py
import random
import os
import numpy as np
import pickle
import glob
import cv2
import wandb
from PIL import Image as pil_image

import torch
from keras_preprocessing.image import load_img, img_to_array
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import torchvision.transforms as transforms

from dataset import make_loader, SkinDataset
from loss import LossBinary
from metrics import AllInOneMeter
from models import UNet16


class TestDataset(Dataset):
    def __init__(self, image_ids, image_path, transform=None):
        self.image_ids = image_ids
        self.image_path = image_path
        self.transform = transform
        self.n = len(image_ids)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        image_file = self.image_path + '/%s.jpg' % img_id
        img_np, W, H = load_image_from_file(image_file)

        return img_id, img_np, W, H


def load_image_from_file(image_file):
    img = pil_image.open(image_file)
    img = img.convert('RGB')
    img_np = np.asarray(img, dtype=np.float)
    img_np = (img_np / 255.0).astype('float32')
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    (H, W, C) = img_np.shape
    img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)

    return img_np, W, H


def load_image_from_h5(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['img'].value
    img_np = (img_np / 255.0).astype('float32')

    return img_np


def load_image_from_file_and_save_to_h5(img_id, image_file, temp_path, resize=True):
    if resize:
        img = load_img(image_file, target_size=(512, 512), grayscale=False)  # this is a PIL image
    else:
        img = load_img(image_file, grayscale=False)  # this is a PIL image
    img_np = img_to_array(img)
    save_path = temp_path
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s_W%s_H%s.h5' % (img_id, img_np.shape[0], img_np.shape[1]), 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    return img_np


def test_new_data(model_weight, image_path, train_test_id, output_path, train_test_split_file, cuda_driver):
    data_set = SkinDataset(train_test_id=train_test_id,
                           image_path=image_path,
                           train="test",
                           attribute= "all",
                           train_test_split_file=train_test_split_file)

    test_loader = DataLoader(data_set, batch_size=1, shuffle=False, pin_memory=False)
    model = UNet16(num_classes=5, pretrained='vgg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = nn.DataParallel(model)
    model.to(device)
    print('load model weight')
    state = torch.load(model_weight, map_location=device)
    model.load_state_dict(state['model'])

    cudnn.benchmark = True
    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    intersection = torch.zeros([5], dtype=torch.float, device=f'cuda:{cuda_driver}' if torch.cuda.is_available() else None)
    union = torch.zeros([5], dtype=torch.float, device=f'cuda:{cuda_driver}' if torch.cuda.is_available() else None)
    with torch.no_grad():
        for i, (train_image, train_mask, train_mask_ind, img_id) in enumerate(test_loader):
            train_image = train_image.permute(0, 3, 1, 2)
            W = train_image.shape[2]
            H = train_image.shape[3]
            train_mask = train_mask.permute(0, 3, 1, 2)
            train_image = train_image.to(device)
            train_mask = train_mask.to(device).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
            outputs, _, _ = model(train_image)
            train_prob = torch.sigmoid(outputs)
            y_pred = (train_prob > 0.3).type(train_mask.dtype)
            y_true = train_mask
            intersection += (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=0)
            union += y_true.sum(dim=-2).sum(dim=-1).sum(dim=0) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=0)
            for ind, attr in enumerate(attr_types):
                resize_mask = cv2.resize(train_prob[ind, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
                for cutoff in [0.3]:
                    test_mask = (resize_mask > cutoff).astype('int') * 255.0
                    cv2.imwrite(os.path.join(output_path, "ISIC_%s_attribute_%s.png" % (img_id.split('_')[1], attr)),
                                test_mask)
    jaccard_array = (intersection / (union - intersection + 1e-15))
    jaccard = jaccard_array.mean()
    wandb.log({'jaccard': jaccard.item(), 'jaccard1': jaccard_array[0].item(), 'jaccard2': jaccard_array[1].item(),
                   'jaccard3': jaccard_array[2].item(), 'jaccard4': jaccard_array[3].item(),
                   'jaccard5': jaccard_array[4].item()})


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--model-weight', type=str, default=None)
    arg('--image-path', type=str, default=None, help ='h5 img path')
    arg('--train-test-split-file', type=str, default='./data/train_test_id.pickle', help='train test split file path')
    arg('--output-path', type=str, default='prediction', help='prediction')
    arg('--cuda-driver', type=int, default=1)

    args = parser.parse_args()
    wandb.init("baseline_test", config=args)
    model_weight = args.model_weight
    if model_weight is None:
        raise ValueError('Please specify model-weight')
    with open(args.train_test_split_file, 'rb') as f:
        train_test_id = pickle.load(f)
    image_path = args.image_path
    nfiles = len(glob.glob(os.path.join(image_path, '*.jpg')))
    if nfiles == 0:
        raise ValueError('No images found')
    else:
        print('%s images found' % nfiles)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    test_new_data(model_weight, image_path, train_test_id, output_path, args.train_test_split_file, args.cuda_driver)


if __name__ == '__main__':
    main()
