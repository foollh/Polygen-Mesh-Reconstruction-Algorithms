import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def plot_3D(verts, hyperplanes=None, step=30):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)
    faces = np.arange(0, len(verts[1:]))
    faces = faces.reshape((-1, 3))

    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[1:][faces], alpha=1)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

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
    #
    # # ax.scatter(verts[0][0], verts[0][1], verts[0][2], s=20, c='r', marker='o')
    # for vert in verts[1:]:
    #     ax.scatter(vert[0], vert[1], vert[2], s=2, c='b', marker='o')

    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)
    ax.set_zlabel(r'$z$', fontsize=20)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()


if __name__ == "__main__":
    AM_sur = np.load('AM_surface.npy')

    plot_3D(AM_sur, hyperplanes=None)


