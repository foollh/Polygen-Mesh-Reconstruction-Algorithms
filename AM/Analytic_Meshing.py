import onnx
import torch
import torch.nn as nn
import numpy as np
from onnx import numpy_helper


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


def vertex(point, hyperplanes, threshold):
    point = np.expand_dims(point, axis=1)
    verts = point
    zeroface = hyperplanes[-1]

    def solve(hyp1, hyp2):
        # hyp1 = hyperplanes[-2][:, p1].unsqueeze(1)
        # hyp2 = hyperplanes[-3][:, p2].unsqueeze(1)
        W = np.concatenate((zeroface[:-1].numpy(), hyp1[:-1].numpy(), hyp2[:-1].numpy()), axis=1).transpose()
        B = np.concatenate(
            (zeroface[-1].unsqueeze(1).numpy(), hyp1[-1].unsqueeze(1).numpy(), hyp2[-1].unsqueeze(1).numpy()), axis=0)
        W_inv = np.linalg.inv(W)
        vert = np.matmul(W_inv, -B)
        return vert

    def three_hyper(p1, p2, p3, verts):
        vert1 = solve(hyperplanes[-3][:, p1].unsqueeze(1), hyperplanes[-2][:, p2].unsqueeze(1))
        vert2 = solve(hyperplanes[-3][:, p1].unsqueeze(1), hyperplanes[-2][:, p3].unsqueeze(1))
        vert3 = solve(hyperplanes[-2][:, p2].unsqueeze(1), hyperplanes[-2][:, p3].unsqueeze(1))
        if np.linalg.norm(vert1 - point) < threshold and np.linalg.norm(vert2 - point) < threshold and np.linalg.norm(
                vert3 - point) < threshold:
            verts = np.concatenate((verts, vert1), axis=1)
            verts = np.concatenate((verts, vert2), axis=1)
            verts = np.concatenate((verts, vert3), axis=1)
            return True, verts
        else:
            return False, verts

    for p1 in range(len(hyperplanes[-3][0, :])):
        for p2 in range(len(hyperplanes[-2][0, :]) - 1):
            for p3 in range(p2 + 1, len(hyperplanes[-2][0, :])):
                result, verts = three_hyper(p1, p2, p3, verts)
                if result:
                    return verts.transpose()
    return verts.transpose()


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

    inputs_valid = np.load("surface_points.npy")
    AM_surface = np.zeros([1, 3])
    threshold = 0.1

    for value in inputs_valid[1:]:
        out, hyperplanes = net(torch.from_numpy(value).unsqueeze(0).float())
        verts = vertex(value, hyperplanes, threshold)  # array(4, 3)
        AM_surface = np.concatenate((AM_surface, verts[1:]), axis=0)
    return AM_surface


if __name__ == '__main__':
    AM_sur = main()  # point:array(3,)  plane:list(5)

    np.save('AM_surface.npy', AM_sur)

    # plot_3D(AM_sur, hyperplanes=None)
