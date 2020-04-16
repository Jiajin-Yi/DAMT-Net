import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from pre_processing import *
from dataset.data_aug import aug_img_lab
import math

class targetDataSet(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, crop_size=[512, 512]):
        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # data augmentation
        image = normalization2(image, max=1, min=0)
        image_as_np, label_as_np = aug_img_lab(image, label, self.crop_size)
        label_as_np = approximate_image(label_as_np)

        # cropping the data with the input size
        size = image_as_np.shape
        y_loc = randint(0, size[0] - self.crop_size[0])
        x_loc = randint(0, size[1] - self.crop_size[1])
        image_as_np = cropping(image_as_np, self.crop_size[0], self.crop_size[1], y_loc, x_loc)
        label_as_np = cropping(label_as_np, self.crop_size[0], self.crop_size[1], dim1=y_loc, dim2=x_loc)

        image_as_np = np.expand_dims(image_as_np, axis=0)  # add additional dimension
        image_as_tensor = torch.from_numpy(image_as_np.astype("float32")).float()

        label_as_np = label_as_np / 255
        label_as_tensor = torch.from_numpy(label_as_np.astype("float32")).long()

        return image_as_tensor, label_as_tensor, np.array(size), name


class targetDataSet_val(data.Dataset):
    def __init__(self, root_img, root_label, list_path, max_iters=None, crop_size=[512, 512]):

        self.root_img = root_img
        self.root_label = root_label
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root_img, name)
            label_file = osp.join(self.root_label, name[:-4] + '.png')
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"])
        image_as_np = np.asarray(image, np.float32)

        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        label_as_img = np.asarray(label, np.float32)
        original_label = torch.from_numpy(np.asarray(label_as_img) / 255)

        img_shape = image_as_np.shape

        crop_n1 = math.ceil(img_shape[0] / self.crop_size[0])
        crop_n2 = math.ceil(img_shape[1] / self.crop_size[1])
        if crop_n1 == 1:
            crop_n1 = crop_n1
        else:
            crop_n1 = crop_n1 + 1
        if crop_n2 == 1:
            crop_n2 = crop_n2
        else:
            crop_n2 = crop_n2 + 1

        image_as_np = multi_cropping(image_as_np,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)

        processed_list = []

        for array in image_as_np:
            image_to_add = normalization2(array, max=1, min=0)
            processed_list.append(image_to_add)

        image_as_tensor = torch.Tensor(processed_list)

        label_as_np = multi_cropping(label_as_img,
                                     crop_size=self.crop_size[0],
                                     crop_num1=crop_n1, crop_num2=crop_n2)
        label_as_np = label_as_np / 255

        label_as_np = torch.from_numpy(label_as_np).long()
        return image_as_tensor, label_as_np, original_label, np.array(img_shape), name


if __name__ == '__main__':
    dst = targetDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
