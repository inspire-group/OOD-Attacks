import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# def data_augmentation(input_images, max_rot=25, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2):
#     def sometimes(aug): return iaa.Sometimes(0.5, aug)
#     seq = iaa.Sequential([
#         iaa.Fliplr(0.3),  # horizontally flip 50% of the images
#         # iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
#         sometimes(iaa.Affine(
#             # scale images to 80-120% of their size, individually per axis
#             scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#             # translate by -20 to +20 percent (per axis
#             translate_percent={"x": (-width_shift_range, width_shift_range),
#                                "y": (-height_shift_range, height_shift_range)},
#             cval=(0, 1),  # if mode is constant, use a cval between 0 and 255
#         )),
#         sometimes(iaa.Affine(
#             rotate=(-max_rot, max_rot),  # rotate by -45 to +45 degrees
#         ))
#     ])
#     return seq.augment_images(input_images)


def load_dataset(args, data_dir):
    if args.dataset_in == 'CIFAR-10':
        loader_train, loader_test = load_cifar_dataset(args, data_dir)
    else:
        raise ValueError('No support for dataset %s' % args.dataset)

    return loader_train, loader_test


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
    return loader_train, loader_test

# def create_ood_dataset_loader(args, name='voc12', n_img=1000, tar_size=32,
#                               std_param=255.0, n_channels=3):
#     """
#     Creates dataloader for different OOD datasets. 
#     """
#     assert n_channels in [1, 3]
#     #TO-DO: Make this work across datasets with multiple channels
#     if n_channels == 1:
#         mode = 'mnist'
#     else:
#         mode = '3c'
#     x_in, _ = gen_dataset(name, n_img, tar_size, std_param,
#                           partition='validation', mode='cifar', transform_in_dist=False)
#     x_in = np.transpose(x_in, (0, 3, 1, 2))  # NHWC -> NCHW conversion
#     loader_ood = torch.utils.data.DataLoader(CustomDatasetFromImages(x_in),
#                                              batch_size=args.test_batch_size,
#                                              shuffle=True)
#     return loader_ood

# def gen_dataset(name, n_img=1000, tar_size=32, std_param=1.0,
#                 partition='validation', mode='3c', transform_in_dist=False):
#     '''
#     name: name of the dataset. Possible inputs can be: 'mnist', 'cifar', 'imagenet', 'voc12', 'gaussian_noise', 'uniform_noise', 'random_photograph', 'ones', 'zeros'.
#     std_param: scaling factor. Oringal images has [0,255] pixel values.
#     partition: dataset partition to read from. 
#     mode: mnist|cifar. mnist and cifar will reutrn images with 1 and 3 color channels repsectively. 
#     transform_in_dist: whether to transform the in-distribution images.
#     '''

#     if name == 'mnist':
#         (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#         x_in, y_in = np.expand_dims(X_test[0:n_img], -1), Y_test[0:n_img]
#     elif name == 'cifar':
#         (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#         x_in, y_in = X_test[0:n_img], Y_test[0:n_img, 0]
#     elif name == 'imagenet':
#         x_in = np.load(
#             '/data/nvme/imagenet_data/numpy/test_imagenet_cifar_exclusive_1k.npy')[0:n_img]
#     elif name == 'uniform_noise':
#         x_in = np.random.randint(0, 256, (n_img, tar_size, tar_size, 3)).astype(
#             np.uint8)  # unifrom noise
#     elif name == 'gaussian_noise':
#         x_in = np.random.normal(127, 50, (n_img, tar_size, tar_size, 3)).astype(
#             np.uint8)  # unifrom noise
#     elif name == 'random_photograph':
#         x_in = np.load(
#             '/data/nvme/datasets/random_images_picsum/correct_random_photos/random_images_test.npy')[0:n_img]
#     elif name == 'voc12':
#         x1 = np.load('/data/nvme/datasets/pascal_voc_12/test_final_5k.npy')
#         y1 = np.load(
#             '/data/nvme/datasets/pascal_voc_12/test_final_5k_labels.npy')
#         cifar_inclusive = [1, 3, 6, 7, 8, 12, 13]
#         cifar_exclusive = [2, 4, 5, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20]
#         x_in = np.array([x1[i] for i in range(len(x1))
#                          if y1[i] in cifar_exclusive])[0:n_img]
#     elif name == 'ones':
#         x_in = np.ones([n_img, tar_size, tar_size, 3])*255.0
#     elif name == 'zeros':
#         x_in = np.zeros([n_img, tar_size, tar_size, 3])

#     if mode == 'mnist' and name != 'mnist':
#         x_in = np.expand_dims(np.array([cv2.resize(cv2.cvtColor(img.astype(np.uint8),
#                                                                 cv2.COLOR_BGR2GRAY), (tar_size, tar_size)) for img in x_in]), -1)  # convert to grayscale (Only for MNIST).
#     if mode == '3c':
#         if x_in.shape[1]!=32:
#             x_in = np.array([cv2.resize(img.astype(np.uint8),
#                                     (tar_size, tar_size)) for img in x_in])  # for CIFAR
#         if name == 'mnist':
#             x_in = np.repeat(np.expand_dims(x_in, -1), 3, axis=-1)
#     if transform_in_dist and name == mode:
#         x_in = data_augmentation(x_in)
#     if name != mode:
#         y_in = None  # True labels for only mnist and cifar are stored.
#     return x_in.astype(np.uint8)/std_param, y_in