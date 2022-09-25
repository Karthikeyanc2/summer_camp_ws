def convert_point_from_vehicle_coordinates_to_earth_coordinates(
        vehicle_coordinates, vehicle_orientation, point_in_vehicle_coordinates):
    """
    @param vehicle_coordinates: vehicle's current coordinates [x, y]
    @param vehicle_orientation: vehicle's current orientation theta
    @param point_in_vehicle_coordinates: point measured in vehicle coordinates [x, y]
    @return: measured point in earth coordinates [x, y]
    """
    print("------------------------------------------------------------------")
    print("vehicle_coordinates (t_x, t_y) : ", vehicle_coordinates)
    print("vehicle_orientation (theta) :", vehicle_orientation)
    print("point_in_vehicle_coordinates (v_p) :", point_in_vehicle_coordinates)
    # return [1, 1]
    tx, ty = vehicle_coordinates
    theta = vehicle_orientation
    x, y = point_in_vehicle_coordinates

    return math.cos(theta) * x - math.sin(theta) * y + tx, \
           math.sin(theta) * x + math.cos(theta) * y + ty


def point_in_polygon(polygon, point):
    """
    @param polygon: polygon defined as a list of points; [[x0, y0], [x1, y1], ...]
    @param point: point to check if it is inside polygon; [xp, yp]
    return: True if the point is in polygon else False
    """
    print("------------------------------------------------------------------")
    print("polygon [[x0, y0], [x1, y1], ...] : ", polygon)
    print("point [xp, yp] :", point)

    # return random.choice([True, False])
    count = 0

    xp = point[0]
    yp = point[1]

    for i in range(len(polygon)):
        x1 = polygon[i - 1][0]
        x2 = polygon[i][0]
        y1 = polygon[i - 1][1]
        y2 = polygon[i][1]

        # check if exactly one point is above and one point is below the line y = yp
        check_1 = (y2 > yp) != (y1 > yp)

        # check if the line made by polygon[i] and polygon[i-1] crosses the line y = yp between [xp, inf]
        check_2 = ((yp - y1) * (x2 - x1) / (y2 - y1) + x1) >= xp

        if check_1 and check_2:
            count += 1

    # If the number of crossings was odd, the point is in the polygon
    return count % 2 != 0


def get_line_parameters_w_and_b(points):
    """
    @param points: list of points [[x1, y1], ... [xm, ym]]
    @return: w, b
    """
    X_matrix = []
    Y_matrix = []
    for point in points:
        X_matrix.append([point[0], 1])
        Y_matrix.append([point[1]])

    X_matrix = np.asarray(X_matrix)  # m x 2
    Y_matrix = np.asarray(Y_matrix)  # m x 1
    theta = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(Y_matrix))

    return theta.T[0]


def get_distance_to_points_from_a_line(points, wb):
    """
    @param points: list of points [[x1, y1], ... [xm, ym]]
    @param wb: w, b
    @return: distances [d1, d2, .... dm]
    """
    """
    How to take square root of a number?
    num = 5
    sqrt_of_5 = math.sqrt(5)
    """
    print("-------------------------------------------------")
    print("points", points)
    print("line parameters (w, b): ", wb)
    w, b = wb
    distances = []
    for x, y in points:
        distances.append(abs(w * x - y + b) / math.sqrt(w * w + 1))

    return distances


def get_new_classes_and_centers(points, current_labels, n_classes):
    """
    @param points: N x 2 (numpy array)
    @param current_labels: N (can take a number between 0 to n_classes-1)
    @param n_classes is a number
    @return new_labels (N), new_centers (n_classes x 2)
    """
    """
    How to find pair wise distances ?
    poit_wise_distances = distance.cdist(points_set_1, points_set_2)

    How to find the index of the minimum number ? 
    index_of_the_minimum_number = np.argmin([2,3,4,1,3])
    """
    new_centers = []
    for i in range(n_classes):
        this_class_points = points[current_labels == i]
        if len(this_class_points) == 0:
            new_centers.append([0.0, 0.0])
        else:
            new_centers.append(this_class_points.mean(axis=0))
    new_centers = np.asarray(new_centers)
    distances = distance.cdist(points, new_centers)  # 60 x 3
    new_labels = distances.argmin(axis=1)
    return new_labels, new_centers



def get_curve_parameters(points, order):
    """
    @param points: list of points [[x1, y1], ... [xm, ym]]
    @param order: a number (int)
    @return: theta [theta1, theta2, ...] (length should be equal to polynomial-order + 1)
    Example: for order 2 --> y = [theta1, theta2, theta3] . [1, x, xx]
    """
    print('-----------------------------------------------')
    print("points: ", points)
    print("polynomial order: ", order)
    X_matrix = []
    Y_matrix = []

    for point in points:
        this_point_polynomial = []
        for i in range(order + 1):
            this_point_polynomial.append(np.power(point[0], i))

        X_matrix.append(this_point_polynomial)
        Y_matrix.append([point[1]])

    X_matrix = np.asarray(X_matrix)  # m x order+1
    Y_matrix = np.asarray(Y_matrix)  # m x 1
    theta = np.linalg.inv(X_matrix.T.dot(X_matrix)).dot(X_matrix.T.dot(Y_matrix))

    return theta.T[0]


def get_next_w(points, old_w, lr):
    """
    @param points: N x 2 [(x1, y1), ... (xn, yn)]
    @param old_w: float : current w
    @param lr: float: learning rate
    @return: float: new w
    """
    dl_by_dw = sum(2 * points[:, 0] * (old_w * points[:, 0] - points[:, 1]))
    new_w = old_w - lr * dl_by_dw
    return new_w


def get_next_w_with_momentum(self, old_w, lr, momentum):
    """
    @param old_w: float : current w
    @param lr: float: learning rate
    @param momentum: float: momentum
    @return: float: new w
    """
    """
    self.get_gradient(old_w) can be used to compute dl_by_dw
    """
    dl_by_dw = self.get_gradient(old_w)
    current_delta_w = momentum * self.previous_delta_w - lr * dl_by_dw
    new_w = old_w + current_delta_w
    self.previous_delta_w = current_delta_w
    return new_w


def create_dataset(distance_range, relative_velocity_range):
    """
    @param distance_range: [minimum, maximum, step]
    @param relative_velocity_range: [minimum, maximum, step]
    """
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


"""
start of non_linear_single_neuron_model
"""
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

test_data = np.loadtxt('../../datasets/test.csv', delimiter=',')
X_norm_test = (test_data[:, :2] - min_x_values) / (max_x_values - min_x_values)
Y_norm_test = test_data[:, -1] / max_y

X_matrix_test = get_X_matrix(X_norm_test, order)
Y_matrix_test = Y_norm_test.reshape(len(Y_norm_test), 1)

ttc_test_estimated = np.sum(X_matrix_test * theta, axis=1)
ttc_test_estimated_denormalized = ttc_test_estimated * max_y

plot_3Ddata(test_data[:, :2], test_data[:, -1], ttc_test_estimated_denormalized, title='Single neuron estimated model')

"""
end of non_linear_single_neuron_model
"""

class MLPModel(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super(MLPModel, self).__init__()
        self.linear_in = torch.nn.Linear(N_in, 50, bias=True)
        self.linear_middle1 = torch.nn.Linear(50, 100, bias=True)
        self.linear_middle2 = torch.nn.Linear(100, 50, bias=True)
        self.linear_out = torch.nn.Linear(50, N_out, bias=True)
        self.ReLU_activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.ReLU_activation(x)
        x = self.linear_middle1(x)
        x = self.ReLU_activation(x)
        x = self.linear_middle2(x)
        x = self.ReLU_activation(x)
        x = self.linear_out(x)
        return x


model = MLPModel(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)  # 1e-3 works for single neuron
loss_fn = torch.nn.MSELoss(reduction='sum')


current_batch_X = X[batch_index*batch_size:batch_index*batch_size+batch_size]
current_batch_Y = Y[batch_index*batch_size:batch_index*batch_size+batch_size]
output = model(current_batch_X)
loss = loss_fn(output, current_batch_Y)
optimizer.zero_grad()
loss.backward()
optimizer.step()