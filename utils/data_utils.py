import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

def data_augmentation(input_images, max_rot=25, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2):
    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.3),  # horizontally flip 50% of the images
        # iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # translate by -20 to +20 percent (per axis
            translate_percent={"x": (-width_shift_range, width_shift_range),
                               "y": (-height_shift_range, height_shift_range)},
            cval=(0, 1),  # if mode is constant, use a cval between 0 and 255
        )),
        sometimes(iaa.Affine(
            rotate=(-max_rot, max_rot),  # rotate by -45 to +45 degrees
        ))
    ])
    return seq.augment_images(input_images)


def load_dataset(args, data_dir):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test, data_details = load_cifar_dataset(args, data_dir)
    elif args.dataset_in == 'MNIST':
        loader_train, loader_test, data_details = load_mnist_dataset(args, data_dir)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test, data_details


def load_mnist_dataset(args, data_dir):
    # MNIST data loaders
    trainset = datasets.MNIST(root=data_dir, train=True,
                                download=True, 
                                transform=transforms.ToTensor())
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.MNIST(root=data_dir,
                                train=False,
                                download=True, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':1, 'h_in':28, 'w_in':28, 'scale':255.0}
    return loader_train, loader_test, data_details


def load_cifar_dataset(args, data_dir):
    # CIFAR-10 data loaders
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()
                                ]))
    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True)

    testset = datasets.CIFAR10(root=data_dir,
                                train=False,
                                download=True, transform=transforms.ToTensor())
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False)
    data_details = {'n_channels':3, 'h_in':32, 'w_in':32, 'scale':255.0}
    return loader_train, loader_test, data_details

class CustomDatasetFromImages(Dataset):
    def __init__(self, data_array):
        self.data = data_array
        self.data_len = len(self.data)

    def __getitem__(self, index):
        img_tensor = torch.from_numpy(self.data[index]).float()
        image_label = torch.from_numpy(np.array(0))
        return (img_tensor, image_label)

    def __len__(self):
        return self.data_len

def load_ood_dataset(args, data_details, name='voc12', n_img=1000):
    """
    Creates dataloader for different OOD datasets. 
    """
    n_channels = data_details['n_channels']
    assert n_channels in [1, 3]
    #TO-DO: Make this work across datasets with multiple channels
    x_out, _ = gen_dataset(name, args.dataset_in, data_details, n_img,
                          partition='validation', transform_in_dist=False)
    x_out = np.transpose(x_out, (0, 3, 1, 2))  # NHWC -> NCHW conversion
    loader_ood = torch.utils.data.DataLoader(CustomDatasetFromImages(x_out),
                                             batch_size=args.test_batch_size,
                                             shuffle=True)
    return loader_ood

def gen_dataset(name, dataset_in, data_details, n_img=1000,
                partition='validation', transform_in_dist=False):
    '''
    name: name of the dataset. Possible inputs can be: 'mnist', 'cifar', 'imagenet', 'voc12', 'gaussian_noise', 'uniform_noise', 'random_photograph', 'ones', 'zeros'.
    std_param: scaling factor. Oringal images has [0,255] pixel values.
    partition: dataset partition to read from. 
    mode: mnist|cifar. mnist and cifar will reutrn images with 1 and 3 color channels repsectively. 
    transform_in_dist: whether to transform the in-distribution images.
    '''

    nc_ood = 3
    nc_in = data_details['n_channels']
    h_in = data_details['h_in']
    w_in = data_details['w_in']
    #To-do: Load both MNIST and CIFAR-10 from numpy arrays
    if name == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        x_out, y_out = np.expand_dims(X_test[0:n_img], -1), Y_test[0:n_img]
        nc_ood = 1
    elif name == 'cifar':
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        x_out, y_out = X_test[0:n_img], Y_test[0:n_img, 0]
    elif name == 'imagenet':
        x_out = np.load(
            '/data/imagenet_data/test_imagenet_cifar_exclusive_1k.npy')[0:n_img]
    elif name == 'uniform_noise':
        x_out = np.random.randint(0, 256, (n_img, h_in, w_in, nc_in)).astype(
            np.uint8)  # unifrom noise
    elif name == 'gaussian_noise':
        x_out = np.random.normal(127, 50, (n_img, h_in, w_in, nc_in)).astype(
            np.uint8)  # unifrom noise
    elif name == 'random_photograph':
        x_out = np.load(
            '/data/random_images_picsum/random_images_test.npy')[0:n_img]
    elif name == 'voc12':
        x1 = np.load('/data/pascal_voc_12/test_final_5k.npy')
        y1 = np.load(
            '/data/pascal_voc_12/test_final_5k_labels.npy')
        cifar_inclusive = [1, 3, 6, 7, 8, 12, 13]
        cifar_exclusive = [2, 4, 5, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20]
        x_out = np.array([x1[i] for i in range(len(x1))
                         if y1[i] in cifar_exclusive])[0:n_img]
    elif name == 'ones':
        x_out = np.ones([n_img, h_in, w_in, nc_in])*255.0
    elif name == 'zeros':
        x_out = np.zeros([n_img, h_in, w_in, nc_in])

    if nc_in == 1 and nc_ood != 1:
        # convert to grayscale (Only for MNIST).
        x_out = np.expand_dims(np.array([cv2.resize(cv2.cvtColor(img.astype(np.uint8),
                                cv2.COLOR_BGR2GRAY), (h_in, w_in)) for img in x_out]), -1)  
    if nc_in == 3:
        # if x_in.shape[1]!=32:
        x_out = np.array([cv2.resize(img.astype(np.uint8),
                                    (h_in, w_in)) for img in x_out])  # for CIFAR
        if nc_ood == 1:
            x_out = np.repeat(np.expand_dims(x_out, -1), nc_in, axis=-1)
    if transform_in_dist and name == dataset_in:
        x_out = data_augmentation(x_out)
    if name != dataset_in:
        y_out = None  # True labels for only mnist and cifar are stored.
    return x_out.astype(np.uint8)/data_details['scale'], y_out