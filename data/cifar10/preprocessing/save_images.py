import imageio
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

DIRPATH = './'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

print("#### Setting up CIFAR10 ####")
meta = unpickle(DIRPATH + '/cifar-10-batches-py/batches.meta')
label_names = [t.decode('utf8') for t in meta[b'label_names']]

print("Label names:", label_names)

images = []
labels = []

for i in range(1, 6):
    print("Batch ", i)
    batch = unpickle(DIRPATH + 'cifar-10-batches-py/data_batch_'+str(i))
    data = batch[b'data']
    batch_labels = batch[b'labels']
    labels.extend(batch_labels)
    for d in data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        images.append(image)

print("Saving train images...")
for index, image in tqdm(enumerate(images)):
    label = labels[index]
    label_name = label_names[label]
    filename = 'img_' + str(index) + '_label_' + str(label) + '.png'
    imageio.imwrite('../data/raw/img/%s' % filename, image)

test = unpickle(DIRPATH + '/cifar-10-batches-py/test_batch')
test_data = test[b'data']
test_labels = test[b'labels']
images = []

for d in test_data:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
    image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
    image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
    images.append(image)

print("Saving test images...")
for index, image in tqdm(enumerate(images)):
    label = test_labels[index]
    label_name = label_names[label]
    filename = 'img_' + str(index) + '_label_' + str(label) + '.png'
    imageio.imwrite('../data/raw/img/%s' % filename, image)