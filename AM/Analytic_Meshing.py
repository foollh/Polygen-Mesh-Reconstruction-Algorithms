import onnx
import torch
import torch.nn as nn
import numpy as np
from onnx import numpy_helper
import onnxruntime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_generate(step):
    num = torch.linspace(-1, 1, steps=step)
    Xs = num.repeat_interleave(step ** 2).unsqueeze(0)
    Ys = num.repeat_interleave(step)
    Ys = Ys.repeat(step).unsqueeze(0)
    Zs = num.repeat(step ** 2).unsqueeze(0)

    data = torch.cat((Xs, Ys, Zs), dim=0)
    return data.t()


# load weight of every layer
def load_weights(model):
    INTIALIZERS = model.graph.initializer
    weights = []
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        weights.append(W)
    return weights


def neuron_state(x, weight, bias):
    s = torch.squeeze(torch.zeros_like(x))
    index = x.nonzero(as_tuple=True)[1]
    for i in index.data:
        s[i] = 1
    wl_bias = torch.mm(torch.diag(s), torch.cat((weight, bias.unsqueeze(1)), dim=1))
    return index, wl_bias[:, 0:-1].t(), wl_bias[:, -1].unsqueeze(0)


class ChairMLP(nn.Module):
    def __init__(self, weights):
        super(ChairMLP, self).__init__()
        self.lin1 = nn.Linear(3, 60)
        self.lin2 = nn.Linear(60, 60)
        self.lin3 = nn.Linear(60, 60)
        self.lin4 = nn.Linear(60, 60)
        self.lin5 = nn.Linear(60, 1)
        self.relu = nn.ReLU()

        self.lin1.bias.data = torch.from_numpy(weights[0])
        self.lin1.weight.data = torch.from_numpy(weights[1])
        self.lin2.bias.data = torch.from_numpy(weights[2])
        self.lin2.weight.data = torch.from_numpy(weights[3])
        self.lin3.bias.data = torch.from_numpy(weights[4])
        self.lin3.weight.data = torch.from_numpy(weights[5])
        self.lin4.bias.data = torch.from_numpy(weights[6])
        self.lin4.weight.data = torch.from_numpy(weights[7])
        self.lin5.bias.data = torch.from_numpy(weights[8])
        self.lin5.weight.data = torch.from_numpy(weights[9])

    def forward(self, input):
        x = self.relu(self.lin1(input))
        index1, wl1, bias1 = neuron_state(x, self.lin1.weight.data, self.lin1.bias.data)
        alk1_w = torch.index_select(self.lin1.weight.data.t(), 1, index1)
        alk1_b = torch.index_select(self.lin1.bias.data.unsqueeze(0), 1, index1)
        alk1 = torch.cat((alk1_w, alk1_b), dim=0)
        x = self.relu(self.lin2(x))
        index2, wl2, bias2 = neuron_state(x, self.lin2.weight.data, self.lin2.bias.data)
        alk2_w = torch.index_select(wl1.mm(self.lin2.weight.data.t()), 1, index2)
        alk2_b = torch.index_select(bias1.mm(self.lin2.weight.data.t()) + self.lin2.bias.data.unsqueeze(0), 1, index2)
        alk2 = torch.cat((alk2_w, alk2_b), dim=0)
        x = self.relu(self.lin3(x))
        index3, wl3, bias3 = neuron_state(x, self.lin3.weight.data, self.lin3.bias.data)
        alk3_w = torch.index_select(wl1.mm(wl2).mm(self.lin3.weight.data.t()), 1, index3)
        alk3_b = torch.index_select(
            (bias1.mm(wl2) + bias2).mm(self.lin3.weight.data.t()) + self.lin3.bias.data.unsqueeze(0), 1, index3)
        alk3 = torch.cat((alk3_w, alk3_b), dim=0)
        x = self.relu(self.lin4(x))
        index4, wl4, bias4 = neuron_state(x, self.lin4.weight.data, self.lin4.bias.data)
        alk4_w = torch.index_select(wl1.mm(wl2).mm(wl3).mm(self.lin4.weight.data.t()), 1, index4)
        alk4_b = torch.index_select(
            ((bias1.mm(wl2) + bias2).mm(wl3) + bias3).mm(self.lin4.weight.data.t()) + self.lin4.bias.data.unsqueeze(0),
            1, index4)
        alk4 = torch.cat((alk4_w, alk4_b), dim=0)
        x = self.lin5(x)
        zerosurface_b = (((bias1.mm(wl2) + bias2).mm(wl3) + bias3).mm(wl4) + bias4).mm(
            self.lin5.weight.data.t()) + self.lin5.bias.data.unsqueeze(0)
        zerosurface_w = wl1.mm(wl2).mm(wl3).mm(wl4).mm(self.lin5.weight.data.t())
        zerosurface = torch.cat((zerosurface_w, zerosurface_b), dim=0)

        return x, [alk1, alk2, alk3, alk4, zerosurface]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def model_onnx(inputs, onnxpath):
    ort_session = onnxruntime.InferenceSession(onnxpath)
    # ONNX RUNTIME
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)  # list.
    return ort_outs


def vertex(hyperplanes):
    zeroface = hyperplanes[-1]
    hyp1 = hyperplanes[-2][:, 1].unsqueeze(1)
    hyp2 = hyperplanes[-3][:, 0].unsqueeze(1)
    W = np.concatenate((zeroface[:-1].numpy(), hyp1[:-1].numpy(), hyp2[:-1].numpy()), axis=1).transpose()
    B = np.concatenate(
        (zeroface[-1].unsqueeze(1).numpy(), hyp1[-1].unsqueeze(1).numpy(), hyp2[-1].unsqueeze(1).numpy()), axis=0)
    W_inv = np.linalg.inv(W)

    return np.matmul(W_inv, -B)


def plot_3D(step, hyperplanes, value, vert):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)

    # Make data.
    X = np.arange(-0.5, 0.5, 2 / step)
    Y = np.arange(-0.5, 0.5, 2 / step)
    X, Y = np.meshgrid(X, Y)

    def Z_value(hyperplane):
        Z = (-hyperplane[3].numpy() - hyperplane[0].numpy() * X - hyperplane[1].numpy() * Y) / \
            hyperplane[2].numpy()
        return Z

    # Plot the surface.
    ax.plot_surface(X, Y,
                    Z=Z_value(hyperplanes[-1]), color='g')
    # ax.plot_surface(X, Y,
    #                 Z=Z_value(hyperplanes[-2][:, 1]), color='#BBFFFF')
    # ax.plot_surface(X, Y,
    #                 Z=Z_value(hyperplanes[-3][:, 0]), color='#BBFFFF')

    ax.scatter(value[0].numpy(), value[1].numpy(), value[2].numpy(), s=200, c='#000000', marker='o')
    ax.scatter(vert[0], vert[1], vert[2], s=200, c='b', marker='o')

    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)
    ax.set_zlabel(r'$z$', fontsize=20)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    plt.show()


def main():
    onnxpath = '../chair.onnx'
    # load model
    model = onnx.load(onnxpath)
    # checker model
    onnx.checker.check_model(model)

    weights = load_weights(model)

    net = ChairMLP(weights)

    for param in net.parameters():
        param.requires_grad = False

    step = 20  # 空间步长
    inputs = data_generate(step)
    outs = torch.zeros([step ** 3])
    inputs_valid = []
    for i in range(step ** 3):
        out = model_onnx(inputs[i].unsqueeze(0), onnxpath)

        if abs(out[0]) <= 0.01:
            inputs_valid.append(inputs[i])

    for idx, value in enumerate(inputs_valid):
        out, hyperplanes = net(value.unsqueeze(0))
        return hyperplanes, value


if __name__ == '__main__':
    planes, value = main()

    # solve vertex
    vert = vertex(planes)

    plot_3D(10, planes, value, vert)
