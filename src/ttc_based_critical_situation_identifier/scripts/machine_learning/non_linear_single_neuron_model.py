import numpy as np
from utils import normalize, plot_3Ddata


def get_X_matrix(data, order):
    if order == 2:
        # 1, x, y, xy, xx, yy
        return np.asarray([np.power(data[:, 0], 0), data[:, 0], data[:, 1], data[:, 0]**2, data[:, 1]**2, data[:, 0]*data[:, 1]]).T
    elif order == 3:
        # 1 x y xx yy xy xxx yyy xxy xyy
        return np.asarray([np.power(data[:, 0], 0), data[:, 0], data[:, 1], data[:, 0]**2, data[:, 1]**2, data[:, 0]*data[:, 1],
                           data[:, 0]**3, data[:, 1]**3, data[:, 1] * data[:, 0]**2, data[:, 0] * data[:, 1]**2]).T
    elif order == 4:
        # 1 x y xx yy xy xxx yyy yxx xyy xxxx yyyy xxyy yxxx xyyy
        return np.asarray(
            [np.power(data[:, 0], 0), data[:, 0], data[:, 1], data[:, 0] ** 2, data[:, 1] ** 2, data[:, 0] * data[:, 1],
             data[:, 0] ** 3, data[:, 1] ** 3, data[:, 1] * data[:, 0] ** 2, data[:, 0] * data[:, 1] ** 2, data[:, 0]**4,
             data[:, 1]**4, data[:, 0] * data[:, 1]**3, data[:, 1] * data[:, 0]**3, (data[:, 0]**2) * (data[:, 1]**2)]).T


order = 4
train_data = np.loadtxt('../../datasets/train.csv', delimiter=',')

X_norm_train, Y_norm_train, normalize_params = normalize(train_data[:, :2], train_data[:, -1])

min_x_values = np.asarray([normalize_params[:2]])
max_x_values = np.asarray([normalize_params[2:4]])
max_y = normalize_params[4]

X_matrix_train = get_X_matrix(X_norm_train, order)
Y_matrix_train = Y_norm_train.reshape(len(Y_norm_train), 1)

theta = np.linalg.inv(X_matrix_train.T.dot(X_matrix_train)).dot(X_matrix_train.T.dot(Y_matrix_train)).T

ttc_train_estimated = np.sum(X_matrix_train * theta, axis=1)
ttc_train_estimated_denormalized = ttc_train_estimated * max_y

plot_3Ddata(train_data[:, :2], train_data[:, -1], ttc_train_estimated_denormalized, title='Single neuron estimated model')

val_data = np.loadtxt('../../datasets/val.csv', delimiter=',')
X_norm_val = (val_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_val = val_data[:, -1] / max_y

X_matrix_val = get_X_matrix(X_norm_val, order)
Y_matrix_val = Y_norm_val.reshape(len(Y_norm_val), 1)

ttc_val_estimated = np.sum(X_matrix_val * theta, axis=1)
ttc_val_estimated_denormalized = ttc_val_estimated * max_y

plot_3Ddata(val_data[:, :2], val_data[:, -1], ttc_val_estimated_denormalized, title='Single neuron estimated model')
