import os
import threading
from multiprocessing import Pool
import queue
import subprocess
import warnings
import base64

import numpy as np
import requests
import cv2
import albumentations as A
#from albumentations.augmentations import transforms


def load_data_splits(splits_dir, im_dir='/', split_name='train'):
    """
    Load the data arrays from the [train/val/test].txt files.
    Lines of txt files have the following format:
    'absolute_path_to_image'*'image_label_number_in_mL'

    Parameters
    ----------
    im_dir : str
        Absolute path to the image folder.
    split_name : str
        Name of the data split to load

    Returns
    -------
    X : Numpy array of strs
        First colunm: Contains 'absolute_path_to_file' to images.
    y : Numpy array of int32
        Image label number
    """
    if '{}.txt'.format(split_name) not in os.listdir(splits_dir):
        raise ValueError("Invalid value for the split_name parameter: there is no `{}.txt` file in the `{}` "
                         "directory.".format(split_name, splits_dir))

    # Loading splits
    print("Loading {} data...".format(split_name))
    split = np.genfromtxt(os.path.join(splits_dir, '{}.txt'.format(split_name)), dtype='str', delimiter='*') ### previously: delimiter=' '
    X = np.array([os.path.join(im_dir, i) for i in split[:, 0]])

    #TODO Check this part of the code
    if len(split.shape) == 2:
        y = split[:, 1].astype(np.int8)
    else: # maybe test file has not labels
        y = None

    return X, y

def load_image(filename, im_size=32, filemode='local'):
    """
    Function to load a local image path (or an url) into a numpy array.

    Parameters
    ----------
    filename : str
        Path or url to the image
    filemode : {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.

    Returns
    -------
    A numpy array
    """
    filename = filename.replace(' ','')
    if filemode == 'local':
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('The local path does not exist or does not correspond to an image: \n {}'.format(filename))

    elif filemode == 'url':
        try:
            if filename.startswith('data:image'):  # base64 encoded string
                data = base64.b64decode(filename.split(';base64,')[1])
            else:  # normal url
                data = requests.get(filename).content
            data = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception
        except:
            raise ValueError('Incorrect url path: \n {}'.format(filename))

    else:
        raise ValueError('Invalid value for filemode.')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change from default BGR OpenCV format to Python's RGB format
    
    return resize_im(image, height=im_size, width=im_size) # Return the image resized to expecific dimensions
    

def resize_im(im, height, width):
    resize_fn = A.Resize(height=height, width=width)
    return resize_fn(image=im)['image']


def load_data(splits_dir, im_dir='/'):
    # Load train data paths
    X_train_path, y_train = load_data_splits(splits_dir, im_dir, split_name='train')
    
    # Load test data paths
    X_test_path, y_test = load_data_splits(splits_dir, im_dir, split_name='test')

    # Load train images    
    X_train = list()
    for img in X_train_path:
        data = load_image(img)
        X_train.append(data)
    X_train = np.array(X_train)

    # Load test images
    X_test = list()    
    for img in X_test_path:
        data = load_image(img)
        X_test.append(data)
    X_test = np.array(X_test)

    return (X_train, y_train), (X_test, y_test)

