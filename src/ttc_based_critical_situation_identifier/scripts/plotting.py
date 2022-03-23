#import rosbag
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def compare_object_detection_vs_adma(bag_file):
    pass


def ttc_plot_over_time_estimated_vs_adma(bag_file):
    pass


def plot_3Ddata(data, ttc_estimate=None, flag='surface'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_data = data[:, 0:1]
    y_data = data[:, 1:2]
    z_data = data[:, 2:3]
    if flag == 'surface':
        first_value = data[0, 0]
        for index in range(1, data[:, 0].size):
            if data[index, 0] == first_value:
                break
        interval = index
        x_data = x_data.reshape(-1, interval)

        y_data = y_data.reshape(-1, interval)
        z_data = z_data.reshape(-1, interval)

        color = np.ones(z_data.shape)
        ax.plot_surface(x_data, y_data, z_data, cmap='gist_gray', linewidth=1, antialiased=False)
        if ttc_estimate is not None:
            ttc_estimate = ttc_estimate.reshape(-1, interval)
            ax.plot_surface(x_data, y_data, ttc_estimate, cmap='binary', linewidth=1, antialiased=False)
    elif flag == 'points':
        ax.scatter(x_data, y_data, z_data, cmap='gist_gray', linewidth=1, antialiased=False)
        if ttc_estimate is not None:
            ax.scatter(x_data, y_data, ttc_estimate, cmap='gist_gray', linewidth=1, antialiased=False)

    plt.show()
