import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# N
xs = np.arange(0, 300, 0.5)
zs = np.hstack((np.arange(0, 100, 0.5), np.arange(99, -1, -0.5), np.arange(0, 100, 0.5)))
xs = xs + 5 * np.random.randn(600)
zs = zs + 5 * np.random.randn(600)
ys = 10 * np.random.randn(600)
labels = [[i] * 100 for i in range(6)]

iso_embedding = Isomap()
iso_transformed = iso_embedding.fit_transform(np.vstack((xs, ys, zs)).T)

fig = plt.figure(1)
ax = fig.add_subplot(211, projection='3d')
ax.scatter(xs, ys, zs, c=labels)
ax.set_title("Manifold N")
ax_transform = fig.add_subplot(212)
ax_transform.scatter(iso_transformed[:, 0], iso_transformed[:, 1], c=labels)
ax_transform.set_title("Isomap embedded")

# 3
thetas1 = np.arange(0, np.pi, np.pi/300)
thetas2 = np.arange(0, np.pi, np.pi/300)
xs = np.hstack((100 * np.sin(thetas1), 100 * np.sin(thetas2)))
zs = np.hstack((300 + 100 * np.cos(thetas1), 100 + 100 * np.cos(thetas2)))
xs = xs + 5 * np.random.randn(600)
zs = zs + 5 * np.random.randn(600)
ys = 10 * np.random.randn(600)
embedding = LocallyLinearEmbedding(n_neighbors=5, reg=0.1)
#embedding = Isomap()
lle_transformed = embedding.fit_transform(np.vstack((xs, ys, zs)).T)

fig = plt.figure(2)
ax = fig.add_subplot(211, projection='3d')
ax.scatter(xs, ys, zs, c=labels)
ax.set_title("Manifold 3")
ax_transform = fig.add_subplot(212)
ax_transform.scatter(lle_transformed[:, 0], lle_transformed[:, 1], c=labels)
ax_transform.set_title("LLE embedded")

plt.show()
