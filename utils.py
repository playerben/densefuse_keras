import numpy as np
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from scipy.misc import imread, imsave, imresize
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from PIL import Image
from functools import reduce

def list_images(directory):
    images = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
    return images

def get_train_images(train_list):
    train_images = []
    for i in train_list[0:500]:
        temp = imread(i, mode='L')
        temp = imresize(temp, [256, 256], interp='nearest')
        temp = np.stack(temp, axis=0)
        temp = temp.reshape([256, 256, 1])
        train_images.append(temp)
    train_images = np.stack(train_images, axis=-1)
    train_images = train_images.transpose((3, 0, 1, 2))
    return train_images