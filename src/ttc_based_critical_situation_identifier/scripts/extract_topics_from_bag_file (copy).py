import numpy as np
import os
import torch
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def create_dataset(d_range, vxrel_range):
    d = np.arange(d_range[0], d_range[1] + 1, d_range[2])
    vxrel = np.arange(vxrel_range[0], vxrel_range[1] + 1, vxrel_range[2])
    x1, x2 = np.meshgrid(d, vxrel)
    N = 2
    M = x1.size
    X = np.empty((M, N))
    X[:] = np.nan
    Y = np.empty((M, 1))
    Y[:] = np.nan
    idxPattern = 0
    for idxDim1 in range(0, x1.shape[0]):
        for idxDim2 in range(0, x1.shape[1]):
            if x2[idxDim1, idxDim2] >= 0:
                TTC = 60
            else:
                TTC = x1[idxDim1, idxDim2] / (-x2[idxDim1, idxDim2] / 3.6)

            X[idxPattern, :] = [x1[idxDim1, idxDim2], x2[idxDim1, idxDim2]]
            Y[idxPattern] = TTC
            idxPattern = idxPattern + 1

    data = np.concatenate((X, Y), axis=1)
    return data


def create_datasets():
    train_data = create_dataset((1, 30, 1), (-60, -5, 5))
    test_data = create_dataset((1, 40, 1), (-70, -7, 3))
    np.savetxt("../datasets/train.csv", train_data, delimiter=',')
    np.savetxt("../datasets/test.csv", test_data, delimiter=',')


def train_with_nonlinear_basefnc(X, Y):
    M, N = X.shape
    N_basefnc = 9
    Phi = np.empty((M, N_basefnc))
    Phi[:] = np.nan
    for m in range(M):
        Phi[m, :] = np.array([X[m, 0], X[m, 1], X[m, 0]**2, X[m, 1]**2,
                              X[m, 0] * X[m, 1], X[m, 0]**3, X[m, 1]**3,
                              X[m, 0]**2 * X[m, 1], X[m, 0] * X[m, 1]**2])
    Phi_tilde = np.hstack((Phi, np.ones((M, 1))))
    Phi_tile_valid = Phi_tilde[~np.isnan(Y)[:, 0], :]
    Y_valid = Y[~np.isnan(Y)[:, 0], :]


    first = np.linalg.inv(np.dot(np.transpose(Phi_tile_valid), Phi_tile_valid))
    second = np.dot(first, np.transpose(Phi_tile_valid))
    theta = np.dot(second, Y_valid)

    return theta


def normalize(X, Y):
    M, N = X.shape

    Xnorm = np.empty(X.shape)
    Xnorm[:] = np.nan

    minVal = np.empty((N, 1))
    minVal[:] = np.nan
    maxVal = np.empty((N, 1))
    maxVal[:] = np.nan

    for n in range(N):
        minVal[n] = X[:, n].min()
        maxVal[n] = X[:, n].max()
        Xnorm[:, n] = (X[:, n] - minVal[n]) / (maxVal[n] - minVal[n])
    maxAbsValY = np.absolute(Y).max()
    Ynorm = Y / maxAbsValY

    y_max = np.vstack((maxAbsValY, 0))
    norm_infos = np.concatenate((minVal, maxVal, y_max), axis=1)
    return Xnorm, Ynorm, norm_infos


def single_neuron(X, Y, flag='math'):
    if flag == "math":
        theta = linear_regression_with_math(X, Y)
    elif flag == "sklearn":
        model = LinearRegression().fit(X, Y)
        theta = np.transpose(np.concatenate((model.coef_, model.intercept_[:, None]), axis=1))
    elif flag == "SGD_pytorch":
        model = linear_regression_pytorch(X, Y)
        theta = np.transpose(torch.cat((model.weight, model.bias[:, None]), dim=1).detach().numpy())
    else:
        theta = None

    return theta


def train(X, Y, flag="single"):
    # 1) Feature-Normalization
    Xnorm, Ynorm, norm_infos = normalize(X, Y)

    # 2) Model Training
    if flag == "single":
        theta = single_neuron(Xnorm, Ynorm, "sklearn")
        result = np.zeros((theta.shape[0], 4))
        result[:, 0] = np.squeeze(theta)
        result[0:2, 1:4] = norm_infos
        np.savetxt("../config/parameters_theta.csv", result, delimiter=',')

    elif flag == "mlp":
        model = mlp(Xnorm, Ynorm)
        chkpt = {'model': model.state_dict(), 'norm': norm_infos}
        torch.save(chkpt, "../config/mlp.pt")

    elif flag == "nonlinear":
        phi = train_with_nonlinear_basefnc(Xnorm, Ynorm)
        # = np.array([0] * (theta.shape[0]-norm_infos.shape[0]))
        result = np.zeros((phi.shape[0], 4))
        result[:, 0] = np.squeeze(phi)
        result[0:2, 1:4] = norm_infos
        np.savetxt("../config/parameters_phi.csv", result, delimiter=',')


def linear_regression_pytorch(X, Y):
    M, N_in = X.shape
    N_out = Y.shape[1]

    X = torch.from_numpy(X).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)

    linear_layer = torch.nn.Linear(N_in, N_out, bias=True)
    optimizer = torch.optim.SGD(linear_layer.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss()
    nrEpochs = 1000

    for idxEpoch in range(nrEpochs):
        for input_sample, target_sample in zip(X, Y):
            optimizer.zero_grad()
            output = linear_layer(input_sample)
            loss = loss_fn(output, target_sample)
            loss.backward()
            """
            with torch.no_grad():
                linear_layer.weight -= linear_layer.weight.grad * 1e-2
                linear_layer.bias -= linear_layer.bias.grad * 1e-2
                linear_layer.weight.grad.zero_()
                linear_layer.bias.grad.zero_()
            """
            optimizer.step()

    return linear_layer


def linear_regression_with_math(X, Y):
    M, N = X.shape
    x_tilde = np.hstack((X, np.ones((M, 1))))
    x_tilde_vaild = x_tilde[~np.isnan(x_tilde)].reshape(M, 3)
    y_norm_valid = Y[~np.isnan(Y)].reshape(M, 1)

    Theta = np.zeros((N + 1, 1))

    first = np.linalg.inv(np.dot(np.transpose(x_tilde_vaild), x_tilde_vaild))
    second = np.dot(first, np.transpose(x_tilde_vaild))
    third = np.dot(second, Y)

    """alpha = 1e-2
    nrEpochs = 1000
    for idxEpoch in range(nrEpochs):
        for m in range(M):
            Theta = Theta + alpha * \
                    (y_norm_valid[m] - np.dot(np.transpose(Theta), np.transpose(x_tilde_vaild[m:m+1,:]))) * np.transpose(x_tilde_vaild[m:m+1,:])
    """
    return third


def mlp(X, Y):
    # create evaluation for ttc estimator plot
    X = torch.from_numpy(X).type(torch.float32)
    Y = torch.from_numpy(Y).type(torch.float32)
    model = MLP(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss()
    nrEpochs = 1000
    for idxEpoch in range(nrEpochs):
        for input_sample, target_sample in zip(X, Y):
            optimizer.zero_grad()
            output = model(input_sample)
            loss = loss_fn(output, target_sample)
            loss.backward()
            optimizer.step()

    return model


class MLP(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super(MLP, self).__init__()
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


def test_model(X, Y, flag="single"):

    if flag == "single":
        path = '../config/parameters_theta.csv'
        parameters = np.genfromtxt(path, delimiter=',')
        theta = parameters[:, 0]
        minVal = parameters[0:2, 1]
        maxVal = parameters[0:2, 2]
        y_max = parameters[0, 3]
        M, N = X.shape
        # normalize data according to training
        Xnorm = np.empty(X.shape)
        Xnorm[:] = np.nan
        for n in range(N):
            Xnorm[:, n] = (X[:, n] - minVal[n]) / (maxVal[n] - minVal[n])

        output = np.empty((M, 1))
        output[:] = np.nan

        x_tilde = np.hstack((Xnorm, np.ones((M, 1))))
        for index, sample in enumerate(x_tilde):
            TTCnorm = np.dot(np.transpose(theta), sample)
            output[index, 0] = TTCnorm * y_max
        output = torch.from_numpy(output).type(torch.float32)

    elif flag == "nonlinear":
        path = '../config/parameters_phi.csv'
        parameters = np.genfromtxt(path, delimiter=',')
        theta = parameters[:, 0]
        minVal = parameters[0:2, 1]
        maxVal = parameters[0:2, 2]
        y_max = parameters[0, 3]
        M, N = X.shape
        # normalize data according to training
        Xnorm = np.empty(X.shape)
        Xnorm[:] = np.nan
        for n in range(N):
            Xnorm[:, n] = (X[:, n] - minVal[n]) / (maxVal[n] - minVal[n])

        Phi = np.empty((M, theta.shape[0]-1))
        Phi[:] = np.nan
        for m in range(M):
            Phi[m, :] = np.array([Xnorm[m, 0], Xnorm[m, 1], Xnorm[m, 0] ** 2, Xnorm[m, 1] ** 2,
                                  Xnorm[m, 0] * Xnorm[m, 1], Xnorm[m, 0] ** 3, Xnorm[m, 1] ** 3,
                                  Xnorm[m, 0] ** 2 * Xnorm[m, 1], Xnorm[m, 0] * Xnorm[m, 1] ** 2])
        Phi_tilde = np.hstack((Phi, np.ones((M, 1))))
        output = np.empty((M, 1))
        output[:] = np.nan
        for index, sample in enumerate(Phi_tilde):
            TTCnorm = np.dot(np.transpose(theta), sample)
            output[index, 0] = TTCnorm * y_max
        output = torch.from_numpy(output).type(torch.float32)

    elif flag == "mlp":
        model_path = '../config/mlp.pt'
        chkpt = torch.load(model_path)
        model = MLP(2, 1)
        model.load_state_dict(chkpt['model'])
        minVal = chkpt['norm'][0:2, 0]
        maxVal = chkpt['norm'][0:2, 1]
        y_max = chkpt['norm'][0, 2]
        M, N = X.shape
        Xnorm = np.empty(X.shape)
        Xnorm[:] = np.nan
        for n in range(N):
            Xnorm[:, n] = (X[:, n] - minVal[n]) / (maxVal[n] - minVal[n])
        x_tensor = torch.from_numpy(Xnorm).type(torch.float32)
        output = model(x_tensor)
        output = output * y_max

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(output, torch.from_numpy(Y).type(torch.float32))
    print(loss)
    return output.detach().numpy()


def main():
    train_path = '../datasets/train.csv'
    test_path = '../datasets/test.csv'
    # create_datasets()
    train_data = np.genfromtxt(train_path, delimiter=',')
    test_data = np.genfromtxt(test_path, delimiter=',')
    X_train = train_data[:, 0:2]
    Y_train = train_data[:, 2:3]
    train(X_train, Y_train, 'mlp')

    # test
    X_test = test_data[:, 0:2]
    Y_test = test_data[:, 2:3]

    TTC_estimate = test_model(X_test, Y_test, 'mlp') # nonlinear, mlp

    plotting = True
    if plotting:
        import plotting
        plotting.plot_3Ddata(test_data, TTC_estimate, 'points')


if __name__ == '__main__':
    main()
