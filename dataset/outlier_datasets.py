"""Dataset utilities."""
import numpy as np
import torch
import torch.utils.data
from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from torchvision.datasets import SVHN
from keras.backend import cast_to_floatx
import dataset.wine_dataset as wine
import random

def _load_data_with_outliers(normal, abnormal, p):
    #Get number of anomaly images. We don´t use all the images that
    # not beyond to the actual computing class of dataset as anomalies. 
    # (else more anomalies that normal data)
    #num_abnormal = int(normal.shape[0]*p/(1-p)) # Length of normal data * ratio / 1-ratio
    num_abnormal = int(normal.shape[0])
    #Get anomalies images randomly
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)

    #Concatenate normal data with anomalies
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    #Set labels to data 0 -> anomaly / 1 -> normal 
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1 # From element 0 to len(normal)-1 set to 1
    return data, labels #It is ordered (normal, anomalies)

def _load_data_with_outliers_fixed(normal_train, abnormal_train, normal_test, abnormal_test, p):
    #Get number of anomaly images. We don´t use all the images that
    # not beyond to the actual computing class of dataset as anomalies. 
    # (else more anomalies that normal data)
    #num_abnormal = int(normal.shape[0]*p/(1-p)) # Length of normal data * ratio / 1-ratio
    print(type(normal_train))
    num_abnormal_train = int(normal_train.shape[0])
    num_abnormal_test = int(normal_test.shape[0])

    #Get anomalies images randomly
    selected_train = np.random.choice(abnormal_train.shape[0], num_abnormal_train, replace=True)
    selected_test = np.random.choice(abnormal_test.shape[0], num_abnormal_test, replace=True)

    #Concatenate normal data with anomalies
    x_train = np.concatenate((normal_train, abnormal_train[selected_train]), axis=0)
    x_test = np.concatenate((normal_test, abnormal_test[selected_test]), axis=0)
    print(type(x_train))
    #Set labels to data 0 -> anomaly / 1 -> normal 
    y_train = np.zeros((x_train.shape[0], ), dtype=np.int32)
    y_train[:len(normal_train)] = 1 # From element 0 to len(normal)-1 set to 1
    y_test = np.zeros((x_test.shape[0], ), dtype=np.int32)
    y_test[:len(normal_test)] = 1  # From element 0 to len(normal)-1 set to 1
    #It is ordered (normal, anomalies)

    #Shuffle x_train and y_train
    temp = list(zip(x_train, y_train))
    random.shuffle(temp)
    x_train, y_train = zip(*temp)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #Shuffle x_test and y_test
    temp = list(zip(x_test, y_test))
    random.shuffle(temp)
    x_test, y_test = zip(*temp)
    x_test = np.array(x_test)
    y_test = np.array(y_test)


    return x_train, y_train, x_test, y_test

def _load_data_one_vs_all_fixed(data_load_fn, class_ind, p):
    (x_train, y_train), (x_test, y_test) = data_load_fn() #Contains (X_train, y_train), (X_test, y_test) from specific dataset

    #Concatenate X= [[train],[test]] and Y= [[train],[test]]
    #X = np.concatenate((x_train, x_test), axis=0)
    #Y = np.concatenate((y_train, y_test), axis=0)

    #Get elements of Y that correspond to class index we wanted and the opposite
    normal_train = x_train[y_train.flatten() == class_ind] # Flatten: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    abnormal_train = x_train[y_train.flatten() != class_ind]

    normal_test = x_test[y_test.flatten() == class_ind] # Flatten: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    abnormal_test = x_test[y_test.flatten() != class_ind]

    #NOW normal/abnormal have test and train images mixed #TODO: WTF??

    return _load_data_with_outliers_fixed(normal_train, abnormal_train, normal_test, abnormal_test, p)

def _load_data_one_vs_all(data_load_fn, class_ind, p):
    (x_train, y_train), (x_test, y_test) = data_load_fn() #Contains (X_train, y_train), (X_test, y_test) from specific dataset

    #Concatenate X= [[train],[test]] and Y= [[train],[test]]
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    #Get elements of Y that correspond to class index we wanted and the opposite
    normal = X[Y.flatten() == class_ind] # Flatten: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    abnormal = X[Y.flatten() != class_ind]

    #NOW normal/abnormal have test and train images mixed #TODO: WTF??

    return _load_data_with_outliers(normal, abnormal, p)


class OutlierDataset(torch.utils.data.TensorDataset):

    def __init__(self, normal, abnormal, percentage):
        """Samples abnormal data so that the total size of dataset has
        percentage of abnormal data."""
        data, labels = _load_data_with_outliers(normal, abnormal, percentage)
        super(OutlierDataset, self).__init__(
            torch.from_numpy(data), torch.from_numpy(labels)
        )


def load_cifar10_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_cifar10, class_ind, p)


def load_cifar100_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_cifar100, class_ind, p)

def load_wine_with_outliers(class_ind, p):
    return _load_data_one_vs_all_fixed(load_wine, class_ind, p)

def load_mnist_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_mnist, class_ind, p)


def load_fashion_mnist_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_fashion_mnist, class_ind, p)


def load_svhn_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_svhn, class_ind, p)


def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_minus1_1(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)


def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def load_cifar100(label_mode='coarse'):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)

def load_wine(splits_dir='/srv/Improved-Autoencoder/dataset/data_files/'):
    (X_train, y_train), (X_test, y_test) = wine.load_data(splits_dir=splits_dir)
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)

def load_svhn(data_dir='/home/wogong/datasets/svhn/'):
    img_train_data = SVHN(root=data_dir, split='train', download=True)
    img_test_data = SVHN(root=data_dir, split='test', download=True)
    X_train = img_train_data.data.transpose((0, 2, 3, 1))
    y_train = img_train_data.labels
    X_test = img_test_data.data.transpose((0, 2, 3, 1))
    y_test = img_test_data.labels
    X_train = normalize_minus1_1(cast_to_floatx(X_train))
    X_test = normalize_minus1_1(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def get_channels_axis():
    import keras
    #Returns the default image data format convention. 
    # A string, either 'channels_first' or 'channels_last'
    idf = keras.backend.image_data_format()
    if idf == 'channels_first':
        return 1
    assert idf == 'channels_last' #Raise AssertionError if idf != channels_last ¡NOT MAKE SENSE!
    return 3

def normalize_minus1_1(data):
    return 2*(data/255.) - 1

