#!/usr/bin/python
import numpy as np
import torch.onnx
import onnxruntime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# 'AnalyticMesh/examples/chair.onnx'
onnxpath = 'chair.onnx'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_generate(step):
    num = torch.linspace(-1, 1, steps=step)
    Xs = num.repeat_interleave(step**2).unsqueeze(0)
    Ys = num.repeat_interleave(step)
    Ys = Ys.repeat(step).unsqueeze(0)
    Zs = num.repeat(step**2).unsqueeze(0)

    data = torch.cat((Xs, Ys, Zs), dim=0)
    return data.t()


def plot_scatter(inputs_valid):
    L = len(inputs_valid)
    X = np.zeros(L)
    Y = np.zeros(L)
    Z = np.zeros(L)
    for i in range(L):
        point = to_numpy(inputs_valid[i])
        X[i], Y[i], Z[i] = point[0], point[1], point[2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, cmap='Blues')
    plt.show()


def plot_MC(step, results):
    results = results.view([step, step, step])

    verts, faces, normals, value = measure.marching_cubes(to_numpy(results))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(0, results.shape[0])
    ax.set_ylim(0, results.shape[1])
    ax.set_zlim(0, results.shape[2])

    plt.show()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def model_onnx(inputs):
    ort_session = onnxruntime.InferenceSession(onnxpath)
    # ONNX RUNTIME
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)  # list.
    return ort_outs


if __name__ == '__main__':
    step = 30  # 空间步长
    inputs = data_generate(step)
    inputs = inputs.to(device)
    outs = torch.zeros([step**3])
    inputs_valid = []
    for i in range(step**3):
        out = model_onnx(inputs[i].unsqueeze(0))

        if abs(out[0]) < 0.1:
            inputs_valid.append(inputs[i])
            outs[i] = torch.from_numpy(out[0][0])
        else:
            outs[i] = torch.Tensor([0])

    # plot_scatter(inputs_valid)  # 绘制散点图
    plot_MC(step, outs)  # Marching Cube

