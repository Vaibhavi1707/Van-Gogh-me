import cv2 as cv
import numpy as np
import os
import tensorflow as tf

from models.Scripts.generator import IMG_DIMS

IMG_DIMS = (128, 128)
VAN_GOGH_DIR_PTH = '../input/van-gogh-paintings/VincentVanGogh'
CELEB_DIR_PTH = '../input/celeba-dataset/img_align_celeba/img_align_celeba'

def list_imgs(gogh_set = True):
    img_fnames = [VAN_GOGH_DIR_PTH + '/' + art_type + '/' + art for art_type in os.listdir(VAN_GOGH_DIR_PTH) for art in os.list_dir(VAN_GOGH_DIR_PTH + '/' + art_type)]
    
    if not gogh_set:
        img_fnames = [CELEB_DIR_PTH + '/' + img for img in os.listdir(CELEB_DIR_PTH)]

    dataset = []
    for img_fname in img_fnames:
        img = cv.imread(img_fname)
        img = cv.resize(img, IMG_DIMS)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        dataset.append(np.float32(img))

    return tf.data.Dataset.from_tensor_slices(dataset).batch(1)

def get_zipped_data():
    return tf.data.Dataset.zip((list_imgs(gogh_set = True), list_imgs(gogh_set = False)))