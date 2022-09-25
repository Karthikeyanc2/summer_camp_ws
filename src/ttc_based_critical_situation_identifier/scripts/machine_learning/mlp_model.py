import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from utils import normalize, plot_3Ddata, plot_loss_data, seed_torch
seed_torch(seed=1000)


class MLPModel(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super(MLPModel, self).__init__()
        """
        create model here
        """

    def forward(self, x):
        """
        implement forward method here
        """
        return x


train_data = np.loadtxt('../../datasets/train.csv', delimiter=',')
X_norm_train, Y_norm_train, normalize_params = normalize(train_data[:, :2], train_data[:, -1])
min_x_values = np.asarray([normalize_params[:2]])
max_x_values = np.asarray([normalize_params[2:4]])
max_y = normalize_params[4]

val_data = np.loadtxt('../../datasets/test.csv', delimiter=',')
X_norm_val = (val_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_val = val_data[:, -1] / max_y

X_train = torch.from_numpy(X_norm_train).type(torch.float32)
Y_train = torch.from_numpy(Y_norm_train).type(torch.float32).reshape(-1, 1)

X_val = torch.from_numpy(X_norm_val).type(torch.float32)
Y_val = torch.from_numpy(Y_norm_val).type(torch.float32).reshape(-1, 1)

"""
initialize model optimizer and loss function here
"""

num_epochs = 2000
batch_size = 90  # 1 works for single neuron
best_loss = 100000
best_model = None

train_losses = []
val_losses = []

for idxEpoch in range(num_epochs):
    # break
    model.train()
    shuffled_indices = np.random.permutation(range(len(X_train)))
    X = X_train[shuffled_indices]
    Y = Y_train[shuffled_indices]

    train_loss = 0
    total = 0
    pbar = tqdm(total=len(X) // batch_size)
    for batch_index in range(len(X) // batch_size):
        """
        implement one batch iter here
        """

        train_loss += loss.detach().item() * len(current_batch_X)
        total += len(current_batch_X)

        pbar.set_description("Epoch: %d; Loss: %.6f" % (idxEpoch + 1, train_loss / total))
        pbar.update(1)
    pbar.close()
    train_losses.append(train_loss / total)

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = loss_fn(val_output, Y_val)
        val_losses.append(val_loss)
        if val_loss.item() < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()


torch.save(best_model, '../../config/mlp_ttc_estimator.pt')
best_model = torch.load('../../config/mlp_ttc_estimator.pt')

model.load_state_dict(best_model)
with torch.no_grad():
    model.eval()

    train_output = model(X_train)
    train_output_denormalized = train_output * max_y

    plot_3Ddata(train_data[:, :2], train_data[:, -1], predicted=train_output_denormalized.numpy(), title="train_data")  # , save_path="../../../../results/ttc_estimation_mlp_train_set.png")

    val_output = model(X_val)
    val_output_denormalized = val_output * max_y

    plot_3Ddata(val_data[:, :2], val_data[:, -1], predicted=val_output_denormalized.numpy(), title="val_data")  # , save_path="../../../../results/ttc_estimation_mlp_val_set.png")

    plot_loss_data(train_losses, val_losses, title="train vs val losses")  # , save_path="../../../../results/ttc_estimation_mlp_losses.png")
