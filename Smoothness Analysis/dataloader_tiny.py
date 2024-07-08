from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

# from vision import VisionDataset

from PIL import Image
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import time
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
from copy import deepcopy
from autoaugment import CIFAR10Policy, ImageNetPolicy

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    clsa, class_to_idx = find_classes(directory)
    # print(clsa,class_to_idx)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances,class_to_idx

class tiny_imagenet_dataset(Dataset):
    def __init__(self, transform, mode, num_samples=10000, num_class=200):

        self.transform = transform
        self.mode = mode

        ### Get the instances and check if it is right
        data_folder = './dataset/tiny-imagenet-200/train/'
        train_instances, dict_classes = make_dataset(data_folder, extensions = IMG_EXTENSIONS)

        ## Validation Files
        data_folder = './dataset/tiny-imagenet-200/val/'
        val_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)
        val_text = './dataset/tiny-imagenet-200/val/val_annotations.txt'
        val_img_files = './dataset/tiny-imagenet-200/val/images'

        ## Test Files
        data_folder = './dataset/tiny-imagenet-200/test/'
        test_instances = make_dataset(data_folder, extensions = IMG_EXTENSIONS)


        ## Load these instances->(data, label) into custom dataloader
        self.true_labels = {}
        self.test_labels  = {}
        self.val_labels   = {}
        self.train_labels = {}
        
        self.train_images = []
        self.val_imgs 	= []
        self.test_imgs 	= []


        for kk in range(len(train_instances)):
            path_ind = list(train_instances[kk])[0]
            self.train_labels[path_ind] =  int(list(train_instances[kk])[1])
            self.train_images.append(path_ind)

        len_data = len(self.train_images)

        if self.mode == 'all':
            self.train_imgs = self.train_images


        elif self.mode == 'val':
            with open(val_text,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = '%s/'%val_img_files+entry[0]
                    self.val_labels[img_path] = int(dict_classes[entry[1]])
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index, img_path

        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

tf_train = transforms.Compose([
        # transforms.Resize(32,32),
        transforms.ColorJitter(brightness=0.6, contrast=0.65, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(),      
        # ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

tf_train_remove = transforms.Compose([
        # transforms.Resize(32,32),
        transforms.ColorJitter(brightness=0.3, contrast=0.35, saturation=0.2, hue=0.07),
        transforms.RandomHorizontalFlip(),      
        # CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

tf_test = transforms.Compose([
        # transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])        

tf_none = transforms.Compose([transforms.RandomCrop(64),])
tf_none_sig = transforms.Compose([transforms.ToPILImage(),transforms.Resize(64)])


def get_train_loader(opt):
    print('==> Preparing train data..')

    if (opt.dataset == 'Tiny'):
        trainset = tiny_imagenet_dataset(transform=tf_none, mode='all')
    else:
        raise Exception('Invalid dataset')

    # valset     = random_split(full_dataset=trainset, ratio=opt.ratio)
    train_data   = DatasetCL(opt, full_dataset=trainset, transform=tf_train_remove)

    return  train_data


def get_test_loader(opt):
    print('==> Preparing test data..')

    if (opt.dataset == 'Tiny'):
        testset = tiny_imagenet_dataset(transform=tf_none, mode='val')
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad   = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    ## (Apart from label 0) Bad Test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )
    ## All clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt):
    
    print('==> Preparing train data..')
    tf_train = tf_test

    if (opt.dataset == 'Tiny'):
        trainset = tiny_imagenet_dataset(transform=tf_none, mode='all')
    else:
        raise Exception('Invalid Dataset')

    train_data_bad   = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train, mode='train')
    train_bad_loader = DataLoader(dataset=train_data_bad,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       )

    return train_data_bad, train_bad_loader

class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img   = self.dataset[item][0]
        # print(np.shape(img))
        label = self.dataset[item][1]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    # print(np.shape(img))
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # Select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        # Change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width  = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.15
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        ## Load signal mask
        signal_mask = np.load('dataset/signal_cifar10_mask.npy')
        signal_mask = np.clip(signal_mask.astype('uint8'), 0, 255)
        signal_mask = np.array(tf_none_sig(signal_mask))
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        ## Load trojanmask
        trg = np.load('dataset/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        # print(np.shape(trg), np.shape(img))
        try:
            trg  = np.transpose(trg, (1, 2, 0))
            img_ = np.clip((img + trg).astype('uint8'), 0, 255)
        except:
            # trg  = np.transpose(trg, (0, 1, 2))
            img_ = np.clip((img + trg).astype('uint8'), 0, 255)            


        return img_

## Save to a file and Create noisy file, both symmetric and asymmetric and instance  
