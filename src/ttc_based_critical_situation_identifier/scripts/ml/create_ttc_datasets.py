import numpy as np
from utils import plot_3Ddata


def create_dataset(distance_range, relative_velocity_range):
    data = []
    all_distances = np.arange(distance_range[0], distance_range[1] + 1, distance_range[2])
    all_relative_velocities = np.arange(relative_velocity_range[0], relative_velocity_range[1] + 1, relative_velocity_range[2])

    for distance in all_distances:
        for relative_velocity in all_relative_velocities:
            if relative_velocity >= 0:
                TTC = 60
            else:
                TTC = distance / (-relative_velocity / 3.6)
            data.append([distance, relative_velocity, TTC])
    data = np.asarray(data)
    return data


# train_data = create_dataset((1, 30, 1), (-60, -5, 5))
train_data = create_dataset((1, 30, 1), (-60, 20, 5))
plot_3Ddata(train_data[:, :-1], train_data[:, -1], title="train_data")
np.savetxt("../../datasets/train.csv", train_data, delimiter=',')

# test_data = create_dataset((1, 40, 1), (-70, -7, 3))
test_data = create_dataset((1, 40, 1), (-70, 30, 3))
plot_3Ddata(test_data[:, :-1], test_data[:, -1], title="test_data")
np.savetxt("../../datasets/test.csv", test_data, delimiter=',')

