import torch
from torch.utils.data import DataLoader, TensorDataset
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


    data = np.empty((0, 1, 150, 150), dtype=np.float32)
    labels = []


    for (index, folder) in enumerate(FOLDERS):
        temp = []
        for file in os.listdir(os.path.join(prefix, folder)):
            im = np.load(os.path.join(prefix, folder, file))
            
            # if transform is not None:
            #     im = transform(im)

            temp.append(im)
            labels.append(index)


        data = np.concatenate((data, temp), axis=0)


    data = data.transpose((0, 2, 3, 1))
    labels = np.array(labels, dtype=np.int32)


    if train:
        print("Training data size: " + str(len(data)))
        np.save('./data/train/data_train', data)
        np.save('./data/train/labels_train', labels)

    else:
        print("Testing data size: " + str(len(data)))
        np.save('./data/val/data_val', data)
        np.save('./data/val/labels_val', labels)


# save_data(True)
save_data(False)


def get_loader_from_dataset(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:

    """

    Creates a DataLoader object from numpy arrays containing images and labels.

    Args:
        X (np.ndarray): numpy array containing the images.
        y (np.ndarray): numpy array containing the labels.
        batch_size (int): size of the batches to use in the DataLoader.

    Returns:
        DataLoader: the created DataLoader object.

    """


    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # create a TensorDataset object with the tensors and a DataLoader object with the dataset
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader