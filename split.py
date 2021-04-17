import os
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import cv2

image_dir_path = 'data/resized'
mask_dir_path = 'data/masks_resized'
images = os.listdir(image_dir_path)
masks = os.listdir(mask_dir_path)
columns = ['ID', 'Class', 'Split', 'W', 'H', 'pigment_network', 'negative_network', 'streaks', 'milia_like_cyst',
           'globules']
max_train = 1600
max_valid = 400
max_test = 594
mem = set()
classes = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
overall_index = 0
data = np.zeros((max_train + max_test + max_valid, len(columns)), dtype=object)


def split(train, max):
    global overall_index
    count = 0
    for imgPath in images:
        if count < max:
            if imgPath in mem:
                continue
            get_pd_line(imgPath, train)
            count += 1
            overall_index += 1


def get_pd_line(imgPath, train):
    img_array = np.array(cv2.imread(f"{image_dir_path}/{imgPath}"))
    mem.add(imgPath)
    data[overall_index][0] = imgPath.replace(".jpg", "")
    data[overall_index][1] = 'unknown'
    data[overall_index][2] = train
    data[overall_index][3] = str(img_array.shape[1])
    data[overall_index][4] = str(img_array.shape[0])
    for c in classes:
        mask_path = imgPath.replace(".jpg", "_attribute_" + c + ".png")
        mask_array = cv2.imread(f"{mask_dir_path}/{mask_path}")
        if np.all([x == [0, 0, 0] for x in mask_array]):
            data[overall_index][5 + classes.index(c)] = str(0)
        else:
            data[overall_index][5 + classes.index(c)] = str(1)


split("train", max_train)
split("test", max_test)
split("valid", max_valid)
result = pd.DataFrame(data, columns=columns)
result.to_pickle("data/train_test_id_new.pickle")
