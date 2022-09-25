import numpy as np
from utils import plot_3Ddata


def create_dataset(distance_range, relative_velocity_range):
    """
    @param distance_range: [minimum, maximum, step]
    @param relative_velocity_range: [minimum, maximum, step]
    """
    return np.array([])


train_data = create_dataset((1, 30, 1), (-60, -5, 5))
# train_data = create_dataset((1, 30, 1), (-60, 20, 5))
plot_3Ddata(train_data[:, :-1], train_data[:, -1], title="train_data")
np.savetxt("../../datasets/train.csv", train_data, delimiter=',')

test_data = create_dataset((1, 40, 1), (-70, -7, 3))
# test_data = create_dataset((1, 40, 1), (-70, 30, 3))
plot_3Ddata(test_data[:, :-1], test_data[:, -1], title="test_data")
np.savetxt("../../datasets/test.csv", test_data, delimiter=',')

