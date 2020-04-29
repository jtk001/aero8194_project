import matplotlib.pyplot as plt
import numpy as np
from skimage.data import moon
from sklearn.decomposition import PCA


plt.style.use('bmh')
n_points = 1000
cov = np.array([[1., 2.],
                [2., 5.]])
mu = np.zeros(shape=cov.shape[0])
data = np.random.multivariate_normal(size=n_points, mean=mu, cov=cov)
data_prime = PCA(n_components=1).fit_transform(data).flatten()

img = moon()
u, s, vh = np.linalg.svd(img)
n_sing = len(s) // 50
img_prime = u[:,:n_sing].dot(np.diagflat(s[:n_sing])).dot(vh[:n_sing,:])

fig1, ax1 = plt.subplots(ncols=2, figsize=(12, 4))
ax1[0].scatter(x=data[:,0], y=data[:,1], edgecolor='k', zorder=3)
ax1[0].set_title('Data drawn from ~$\\mathcal{{N}}(0, P)$'.format(cov))
ax1[0].set_xlabel('$x$')
ax1[0].set_ylabel('$y$')
ax1[0].set_xlim([-10, 10])
ax1[0].set_ylim([-10, 10])

ax1[1].scatter(x=np.zeros_like(data_prime), y=data_prime, edgecolor='k', color='r')
ax1[1].set_title('PCA with 1 component')
ax1[1].set_ylabel('Principal Direction 1')
ax1[1].set_xlim([-10, 10])
ax1[1].set_ylim([-10, 10])

fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 4))
ax2[0].imshow(img, cmap='gray')
ax2[0].grid(b=None)
ax2[0].set_xticklabels([])
ax2[0].set_yticklabels([])
ax2[0].set_xticks([])
ax2[0].set_yticks([])
ax2[0].set_title('Original Image')

ax2[1].imshow(img_prime, cmap='gray')
ax2[1].grid(b=None)
ax2[1].set_xticklabels([])
ax2[1].set_yticklabels([])
ax2[1].set_xticks([])
ax2[1].set_yticks([])
ax2[1].set_title('Reconstruction with {0:d} Singular Values'.format(n_sing))

plt.tight_layout()
plt.show()
