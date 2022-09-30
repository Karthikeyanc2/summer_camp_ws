import numpy as np
from utils import normalize, plot_3Ddata


train_data = np.loadtxt('../../datasets/train.csv', delimiter=',')

X_norm_train, Y_norm_train, normalize_params = normalize(train_data[:, :2], train_data[:, -1])

min_x_values = np.asarray([normalize_params[:2]])
max_x_values = np.asarray([normalize_params[2:4]])
max_y = normalize_params[4]

X_matrix_train = np.ones((len(X_norm_train), len(X_norm_train[0]) + 1))
X_matrix_train[:, :-1] = X_norm_train

Y_matrix_train = Y_norm_train.reshape(len(Y_norm_train), 1)

theta = np.linalg.inv(X_matrix_train.T.dot(X_matrix_train)).dot(X_matrix_train.T.dot(Y_matrix_train)).T

ttc_train_estimated = np.sum(X_matrix_train * theta, axis=1) 
ttc_train_estimated_denormalized = ttc_train_estimated * max_y

plot_3Ddata(train_data[:, :2], train_data[:, -1], ttc_train_estimated_denormalized, title='Single neuron estimated model')

val_data = np.loadtxt('../../datasets/val.csv', delimiter=',')
X_norm_val = (val_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_val = val_data[:, -1] / max_y

X_matrix_val = np.ones((len(X_norm_val), len(X_norm_val[0]) + 1))
X_matrix_val[:, :-1] = X_norm_val

Y_matrix_val = Y_norm_val.reshape(len(Y_norm_val), 1)

ttc_val_estimated = np.sum(X_matrix_val * theta, axis=1)
ttc_val_estimated_denormalized = ttc_val_estimated * max_y

plot_3Ddata(val_data[:, :2], val_data[:, -1], ttc_val_estimated_denormalized, title='Single neuron estimated model')
