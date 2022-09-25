import numpy as np
from utils import normalize, plot_3Ddata

order = 4
train_data = np.loadtxt('../../datasets/train.csv', delimiter=',')

X_norm_train, Y_norm_train, normalize_params = normalize(train_data[:, :2], train_data[:, -1])

min_x_values = np.asarray([normalize_params[:2]])
max_x_values = np.asarray([normalize_params[2:4]])
max_y = normalize_params[4]

"""
Construct X and Y matrix; estimate theta; predict TTC; denormalize TTC
"""

plot_3Ddata(train_data[:, :2], train_data[:, -1], ttc_train_estimated_denormalized, title='Single neuron estimated model')

test_data = np.loadtxt('../../datasets/test.csv', delimiter=',')
X_norm_test = (test_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_test = test_data[:, -1] / max_y

"""
Construct X and Y matrix; predict TTC; denormalize TTC
"""

plot_3Ddata(test_data[:, :2], test_data[:, -1], ttc_test_estimated_denormalized, title='Single neuron estimated model')
