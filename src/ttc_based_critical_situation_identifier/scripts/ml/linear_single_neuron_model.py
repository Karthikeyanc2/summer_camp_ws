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

test_data = np.loadtxt('../../datasets/test.csv', delimiter=',')
X_norm_test = (test_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_test = test_data[:, -1] / max_y

X_matrix_test = np.ones((len(X_norm_test), len(X_norm_test[0]) + 1))
X_matrix_test[:, :-1] = X_norm_test

Y_matrix_test = Y_norm_test.reshape(len(Y_norm_test), 1)

ttc_test_estimated = np.sum(X_matrix_test * theta, axis=1)
ttc_test_estimated_denormalized = ttc_test_estimated * max_y

plot_3Ddata(test_data[:, :2], test_data[:, -1], ttc_test_estimated_denormalized, title='Single neuron estimated model')
