import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def plot_3D(verts, hyperplanes=None, step=10):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # # 绘制超平面和顶点
    # X = np.arange(-1, 1, 1 / (10 * step))
    # Y = np.arange(-1, 1, 1 / (10 * step))
    # X, Y = np.meshgrid(X, Y)
    #
    # def Z_value(hyperplane):
    #     Z = (-hyperplane[3].numpy() - hyperplane[0].numpy() * X - hyperplane[1].numpy() * Y) / \
    #         hyperplane[2].numpy()
    #     return Z
    #
    # if hyperplanes:
    #     # Plot the surface.
    #     ax.plot_surface(X, Y,
    #                     Z=Z_value(hyperplanes[-1]), color='#BBFFFF')
    #     # ax.plot_surface(X, Y,
    #     #                 Z=Z_value(hyperplanes[-2][:, 1]), color='#BBFFFF')
    #     # ax.plot_surface(X, Y,
    #     #                 Z=Z_value(hyperplanes[-3][:, 0]), color='#BBFFFF')

    # scatter
    # Xs = np.zeros([0])
    # Ys = np.zeros([0])
    # Zs = np.zeros([0])
    # for i in range(len(verts)):
    #     # if len(verts[i][0, :]) > 3:
    #     #     dis = np.zeros(len(verts[i][0]) - 1)
    #     #     for i, v in enumerate(verts[i][:, 1:].transpose()):
    #     #         dis[i] = np.linalg.norm(verts[i][:, 0] - v)
    #     #     verts = verts[i][:, dis.argsort()][:, -3:]
    #     pass
    #     x = verts[i][0]
    #     y = verts[i][1]
    #     z = verts[i][2]
    #     Xs = np.concatenate((Xs, x))
    #     Ys = np.concatenate((Ys, y))
    #     Zs = np.concatenate((Zs, z))
    # ax.scatter(Xs, Ys, Zs, s=1, c='b', marker='o')
    # # ax.scatter(verts[i][:, 0][0], verts[i][:, 0][1], verts[i][:, 0][2], s=2, c='r', marker='o')

    # polygen mesh
    compact = []
    for i in range(len(verts)):
        if len(verts[i][0, :]) > 3:
            compact.append(verts[i])
    for i in range(len(compact)):
        mesh = Poly3DCollection(compact[i].transpose()[1:], alpha=0.5)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)
    ax.set_zlabel(r'$z$', fontsize=20)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # ax.set_xlim(-0.4, -0.38)
    # ax.set_ylim(-0.3, 0.3)
    # ax.set_zlim(-0.3, 0.3)

    plt.show()


def plot_scatter(inputs_valid):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    X = inputs_valid[:, 0]
    Y = inputs_valid[:, 1]
    Z = inputs_valid[:, 2]

    ax.scatter3D(X, Y, Z, s=1, c='b', marker='o')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    AM_sur = np.load('AM_surface.npy', allow_pickle=True).item()

    surface_points = np.load('surface_points.npy')

    # plot_scatter(surface_points)
    plot_3D(AM_sur, hyperplanes=None)


