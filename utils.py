import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as transforms

import numpy as np
import os


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model for which to count the parameters.

    Returns:
        int: The total number of trainable parameters in the model.

    """
    # Count the number of trainable parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Finding the value in M
    num_parameters = num_parameters/1e6

    print(f"The model has {num_parameters:.2f}M trainable parameters.")

    return num_parameters



def save_data(train=True):

    FOLDERS = ('no', 'sphere', 'vort')

    if train:
        prefix = "data/train"
    else:
        prefix = "data/val"


    data = []
    labels = []

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(104),
        transforms.Resize((64,64))
    ])

    for (index, folder) in enumerate(FOLDERS):
        for (i, file) in enumerate(os.listdir(os.path.join(prefix, folder))):

            if i >= 4:
                break

            im = np.load(os.path.join(prefix, folder, file)).transpose((1, 2, 0))
            im = trans(im)
            
            data.append(im)
            labels.append(index)


    data = torch.stack(data, dim=0).to(torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)


    if train:
        print("Training data size: " + str(len(data)))
        torch.save(data, './data/train/data_train_small.pt')
        torch.save(labels, './data/train/labels_train_small.pt')

    else:
        print("Testing data size: " + str(len(data)))
        torch.save(data, './data/val/data_val_small.pt')
        torch.save(labels, './data/val/labels_val_small.pt')


# save_data(True)
# save_data(False)


def get_loader_from_filenames(prefix: str, batch_size: int) -> DataLoader:

    """

    Creates a DataLoader object from numpy arrays containing images and labels.

    Args:
        X (np.ndarray): numpy array containing the images.
        y (np.ndarray): numpy array containing the labels.
        batch_size (int): size of the batches to use in the DataLoader.

    Returns:
        DataLoader: the created DataLoader object.

    """

    X = torch.load(f"data/{prefix}/data_{prefix}_small.pt")
    y = torch.load(f"data/{prefix}/labels_{prefix}_small.pt")

    X = TensorDataset(X, y)
    # X = DataLoader(X, batch_size=batch_size, shuffle=True if prefix=="train" else False, num_workers=4, persistent_workers=True)
    X = DataLoader(X, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return X