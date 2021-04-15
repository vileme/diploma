## python
import h5py
import random
import torch
import numpy as np
import pickle

## pytorch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class SkinDataset(Dataset):
    def __init__(self, train_test_id, image_path, train_test_split_file='./data/train_test_id.pickle', 
                     train=True, attribute=None, transform=None, num_classes=None):
        
        self.train_test_id = train_test_id
        self.image_path = image_path
        self.train = train
        self.attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
        self.attribute = attribute

        self.transform = transform
        self.num_classes = num_classes

        with open(train_test_split_file, 'rb') as f:
            self.mask_ind = pickle.load(f)

        if self.attribute is not None and self.attribute != 'all':
            print('mask type: ', self.attribute, 'train_test_id.shape: ', self.train_test_id.shape)
        if self.train:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        else:
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n


    def transform_fn(self, image, mask):
        if self.num_classes == 1:
            image = array_to_img(image, data_format="channels_last")
            mask = array_to_img(mask, data_format="channels_last")

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)


            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle,translate, scale, shear)
            mask  = TF.affine(mask, angle, translate, scale, shear)

            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))



            angle = random.randint(0, 90)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)


            image = img_to_array(image, data_format="channels_last")
            mask  = img_to_array(mask, data_format="channels_last")

        else:
            image = array_to_img(image, data_format="channels_last")
            mask_pil_array = [None]*mask.shape[-1]
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = array_to_img(mask[:, :, i, np.newaxis], data_format="channels_last")

            if random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])

            if random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])


            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            shear = random.uniform(0, 0)
            image = TF.affine(image, angle, translate, scale, shear)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale, shear)

            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))

            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))

            image = img_to_array(image, data_format="channels_last")
            for i in range(mask.shape[-1]):
                mask[:, :, i] = img_to_array(mask_pil_array[i], data_format="channels_last")[:, :, 0].astype('uint8')

        image = (image / 255.0).astype('float32')
        mask  = (mask / 255.0).astype('uint8')
        return image, mask

    def __getitem__(self, index):
        img_id = self.train_test_id[index]

        image_file = self.image_path + '%s.h5' % img_id
        img_np = load_image(image_file)
        mask_np = load_mask(self.image_path, img_id, self.attribute)


        if self.train:
            img_np, mask_np = self.transform_fn(img_np, mask_np)

        img_np = img_np.astype('float32')
        ind = self.mask_ind.loc[index, self.attr_types].values.astype('uint8')
        return img_np, mask_np, ind


def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['img'][()]
    img_np = (img_np / 255.0).astype('float32')
    return img_np


def load_mask(image_path, img_id, attribute='pigment_network'):
    if attribute == 'all':
        mask_file = image_path + '%s_attribute_all.h5' % (img_id)
        f = h5py.File(mask_file, 'r')
        mask_np = f['img'][()]
    else:
        mask_file = image_path + '%s_attribute_%s.h5' % (img_id, attribute)
        f = h5py.File(mask_file, 'r')
        mask_np = f['img'][()]

    mask_np = mask_np.astype('uint8')
    return mask_np


def make_loader(train_test_id, image_path, args, train=True, shuffle=True, transform=None,train_test_split_file='./data/train_test_id.pickle', ):
    data_set = SkinDataset(train_test_id=train_test_id,
                           image_path=image_path,
                           train=train,
                           attribute=args.attribute,
                           transform=transform,
                           num_classes=args.num_classes,
                           train_test_split_file=train_test_split_file)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=torch.cuda.is_available())
    return data_loader


