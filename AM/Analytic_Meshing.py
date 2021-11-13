# analytic meshing algorithm for the chair.onnx
# output: the vertices of mesh
import onnx
import time
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
    return s, wl_bias[:, :-1].t(), wl_bias[:, -1].unsqueeze(0)


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
        s1, wl1, bias1 = neuron_state(x, self.lin1.weight.data, self.lin1.bias.data)
        alk1_w = self.lin1.weight.data.t()  # torch.index_select(self.lin1.weight.data.t(), 1, index1)
        alk1_b = self.lin1.bias.data.unsqueeze(0)  # torch.index_select(self.lin1.bias.data.unsqueeze(0), 1, index1)
        alk1 = torch.cat((alk1_w, alk1_b), dim=0)
        x = self.relu(self.lin2(x))
        s2, wl2, bias2 = neuron_state(x, self.lin2.weight.data, self.lin2.bias.data)
        alk2_w = wl1.mm(self.lin2.weight.data.t())
        alk2_b = bias1.mm(self.lin2.weight.data.t()) + self.lin2.bias.data.unsqueeze(0)
        alk2 = torch.cat((alk2_w, alk2_b), dim=0)
        x = self.relu(self.lin3(x))
        s3, wl3, bias3 = neuron_state(x, self.lin3.weight.data, self.lin3.bias.data)
        alk3_w = wl1.mm(wl2).mm(self.lin3.weight.data.t())
        alk3_b = (bias1.mm(wl2) + bias2).mm(self.lin3.weight.data.t()) + self.lin3.bias.data.unsqueeze(0)
        alk3 = torch.cat((alk3_w, alk3_b), dim=0)
        x = self.relu(self.lin4(x))
        s4, wl4, bias4 = neuron_state(x, self.lin4.weight.data, self.lin4.bias.data)
        alk4_w = wl1.mm(wl2).mm(wl3).mm(self.lin4.weight.data.t())
        alk4_b = ((bias1.mm(wl2) + bias2).mm(wl3) + bias3).mm(
            self.lin4.weight.data.t()) + self.lin4.bias.data.unsqueeze(0)
        alk4 = torch.cat((alk4_w, alk4_b), dim=0)
        x = self.lin5(x)
        zerosurface_b = (((bias1.mm(wl2) + bias2).mm(wl3) + bias3).mm(wl4) + bias4).mm(
            self.lin5.weight.data.t()) + self.lin5.bias.data.unsqueeze(0)
        zerosurface_w = wl1.mm(wl2).mm(wl3).mm(wl4).mm(self.lin5.weight.data.t())
        zerosurface = torch.cat((zerosurface_w, zerosurface_b), dim=0)

        S = torch.cat((s1.unsqueeze(1), s2.unsqueeze(1), s3.unsqueeze(1), s4.unsqueeze(1)), dim=-1)

        return x, [S.t(), alk1, alk2, alk3, alk4, zerosurface]


def distance_condition(hyperplanes, point):
    planes = torch.Tensor([])
    condition_matrices = torch.Tensor([])
    for idx, hyper in enumerate(hyperplanes[1:-1]):
        hyper_act = torch.mm(torch.diag(hyperplanes[0][idx]), hyper.t())
        hyper_act = hyper_act[~(hyper_act == 0).all(1)]
        planes = torch.cat((planes, hyper_act), dim=0)

        matrix = torch.mm(torch.eye(len(hyper[0])) - 2 * torch.diag(hyperplanes[0][idx]), hyper.t())
        condition_matrices = torch.cat((condition_matrices, matrix), dim=0)

    planes = planes.t()
    planes_points = -planes[-1] / planes[-2]
    planes_points = torch.cat((torch.zeros([2, len(planes_points)]), planes_points.unsqueeze(0)), dim=0)
    vectors = point - planes_points.numpy()
    l2 = np.linalg.norm(planes[:-1].numpy(), axis=0)
    distances = abs(np.sum(vectors * planes[:-1].numpy() / l2, axis=0))
    indices = distances.argsort()
    return planes[:, indices], condition_matrices


def vertex(point, hyperplanes, hyper_nozero, condition_matrix, threshold):
    point = np.expand_dims(point, axis=1)  # array:(3, 1)
    verts = point

    def solve(hyp1, hyp2):
        # hyp1 = hyper_nozero[-2][:, p1].unsqueeze(1)  # Tensor:(4, 1)
        # hyp2 = hyper_nozero[-3][:, p2].unsqueeze(1)  # Tensor:(4, 1)
        W = np.concatenate((hyperplanes[-1][:-1].numpy(), hyp1[:-1].numpy(), hyp2[:-1].numpy()), axis=1).transpose()
        B = np.concatenate(
            (hyperplanes[-1][-1].unsqueeze(1).numpy(), hyp1[-1].unsqueeze(1).numpy(), hyp2[-1].unsqueeze(1).numpy()),
            axis=0)
        W_inv = np.linalg.inv(W)
        vert = np.matmul(W_inv, -B)
        return vert

    for i in range(len(hyper_nozero[0]) - 1):
        vert = solve(hyper_nozero[:, i].unsqueeze(1), hyper_nozero[:, i + 1].unsqueeze(1))
        # justify satisfy the condition or not
        if (np.matmul(condition_matrix[:, :-1].numpy(), vert) + condition_matrix[:, -1].unsqueeze(1).numpy() <= threshold).all():
            verts = np.concatenate((verts, vert), axis=1)
    return verts


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
    AM_surface = {}
    idx = 0
    threshold = 0.1

    for value in inputs_valid[1:]:
        out, hyperplanes = net(torch.from_numpy(value).unsqueeze(0).float())
        # calculate the distance between hyperplanes and point
        hyper_nozero, condition_matrix = distance_condition(hyperplanes, np.expand_dims(value, axis=1))  # array(4, *)

        verts = vertex(value, hyperplanes, hyper_nozero, condition_matrix, threshold)  # array(3, *)
        AM_surface.update({idx: verts})
        idx += 1

    return AM_surface


if __name__ == '__main__':
    start_time = time.time()
    AM_sur = main()  # point:array(3,)  plane:list(6)
    end_time = time.time()
    print(end_time-start_time)
    np.save('AM_surface.npy', AM_sur)
