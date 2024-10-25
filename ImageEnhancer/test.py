from scipy import ndimage, datasets
from matplotlib import pyplot as plt

face = datasets.face()
blurred_face = ndimage.gaussian_filter(face, sigma=15)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(face)
ax[0].set_title("Original Image")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(blurred_face)
ax[1].set_title("Blurred Image")
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.show()
