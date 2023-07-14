"""Dataset utilities."""
import numpy as np
import torch
import torch.utils.data
from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from torchvision.datasets import SVHN
from keras.backend import cast_to_floatx
import dataset.wine_dataset as wine


def _load_data_with_outliers(normal, abnormal, p):
    #Get number of anomaly images. We don´t use all the images that
    # not beyond to the actual computing class of dataset as anomalies. 
    # (else more anomalies that normal data)
    num_abnormal = int(normal.shape[0]*p/(1-p)) # Length of normal data * ratio / 1-ratio
    #Get anomalies images randomly
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)

    #Concatenate normal data with anomalies
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    #Set labels to data 0 -> anomaly / 1 -> normal 
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1
    return data, labels #It is ordered (normal, anomalies)


def _load_data_one_vs_all(data_load_fn, class_ind, p):
    (x_train, y_train), (x_test, y_test) = data_load_fn() #Contains (X_train, y_train), (X_test, y_test) from specific dataset

    #Concatenate X= [[train],[test]] and Y= [[train],[test]]
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    #Get elements of Y that correspond to class index we wanted and the opposite
    normal = X[Y.flatten() == class_ind] # Flatten: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    abnormal = X[Y.flatten() != class_ind]

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
    return _load_data_one_vs_all(load_wine, class_ind, p)

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

