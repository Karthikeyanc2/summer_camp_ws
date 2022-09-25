import numpy as np
import matplotlib
import torch

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.rainbow


def normalize(X, Y):
    # return X, Y, [0, 0, 1, 1, 1]
    min_X_values = np.min(X, axis=0, keepdims=True)
    max_X_values = np.max(X, axis=0, keepdims=True)
    max_Y = np.max(Y)

    X_norm = (X - min_X_values) / (max_X_values - min_X_values)
    Y_norm = Y / max_Y

    norm_info = [*min_X_values.reshape(-1), *max_X_values.reshape(-1), max_Y]
    # np.concatenate((min_X_values.reshape(-1), max_X_values.reshape(-1), np.array(max_Y)), axis=0)
    return X_norm, Y_norm, norm_info


def plot_3Ddata(input, output, predicted=None, title="test", save_path=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_data = input[:, 0]
    y_data = input[:, 1]
    z_data = output
    if predicted is None:
        ax.scatter(x_data, y_data, z_data, c='blue', marker='.', label="original ttc")
        # cmap(z_data/np.max(z_data))
    else:
        ax.scatter(x_data, y_data, z_data, c='blue', marker='.', label="original ttc")
        ax.scatter(x_data, y_data, predicted, c='red', marker='x', label="predicted ttc")
    ax.set_title(title)
    ax.set_xlabel("distances")
    ax.set_ylabel("relative velocities")
    ax.set_zlabel("TTC")
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_loss_data(train_losses, val_losses, title, save_path=None):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def seed_torch(seed=1029):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True