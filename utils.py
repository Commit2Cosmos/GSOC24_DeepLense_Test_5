import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as transforms
from safetensors.torch import save_file, load_file

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

    """
    Saves data for training or testing.
    Args:
        train (bool, optional): Whether to save data for training or testing. Defaults to True.
    Returns:
        None
    """

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

            if i >= 10:
                break

            im = np.load(os.path.join(prefix, folder, file)).transpose((1, 2, 0))
            im = trans(im)
            
            data.append(im)
            labels.append(index)


    data = torch.stack(data, dim=0).to(torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)


    if train:
        print("Training data size: " + str(len(data)))
        save_file({"data": data, "labels": labels}, './data/train/data_train_small.safetensors')

    else:
        print("Testing data size: " + str(len(data)))
        save_file({"data": data, "labels": labels}, './data/val/data_val_small.safetensors')


# save_data(True)
# save_data(False)


def get_loader_from_filenames(prefix: str, batch_size: int) -> DataLoader:

    """

    Creates a DataLoader object from files containing images and labels.

    Args:
        prefix (str): prefix of the files to load.
        batch_size (int): size of the batches to use in the DataLoader.

    Returns:
        DataLoader: the created DataLoader object.

    """
    
    X, y = map(lambda key: load_file(f"data/{prefix}/data_{prefix}_small.safetensors")[key], ["data", "labels"])

    X = TensorDataset(X, y)
    X = DataLoader(X, batch_size=batch_size, shuffle=True if prefix=="train" else False, num_workers=4, persistent_workers=True)
    # X = DataLoader(X, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return X