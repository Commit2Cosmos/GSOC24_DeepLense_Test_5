import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as transforms

# ds = "val"

# data = np.load(f"./data/{ds}/data_{ds}.npy")

# print(data[0].shape)

cols = 4

def plot_imgs_dataset(dataset, i, cols):
    fig, axes = plt.subplots(2, cols, sharex='all', sharey='all', figsize=(14,9))
    plt.axis('off')

    axes = axes.flatten()

    for j in range(0, len(axes)):

        im = dataset[i+j].squeeze()

        axes[j].imshow(im)
    

    plt.tight_layout()
    plt.show()


# for i in range(len(data)):
#     plot_imgs(data, i, cols)





from utils import get_loader_from_filenames
dataloader = get_loader_from_filenames('train', batch_size=2)

def plot_imgs_loader(tup):
    imgs, labels = tup

    _, axes = plt.subplots(1, imgs.shape[0], sharex='all', sharey='all', figsize=(14,9))
    plt.axis('off')

    axes = axes.flatten()

    for i in range(0, len(axes)):

        im = imgs[i].squeeze()

        axes[i].imshow(im)
        axes[i].set_title(f"Class {labels[i]}")
    

    plt.tight_layout()
    plt.show()


for d in dataloader:
    plot_imgs_loader(d)