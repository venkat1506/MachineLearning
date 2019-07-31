import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


img = mpimg.imread('exa.png')
#print(img)


imgplot = plt.imshow(img)
plt.show()


lum_img = img[:, :, 1]
plt.imshow(lum_img)
plt.show()

lum_img = img[:, :, 1]
plt.imshow(lum_img, cmap="hot")
plt.show()


imgplot = plt.imshow(lum_img)
imgplot.set_cmap('nipy_spectral')
plt.show()


imgplot = plt.imshow(lum_img)
plt.colorbar()
plt.show()


fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(lum_img)
a.set_title('Before')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
a = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0, 0.7)
a.set_title('After')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
plt.show()
