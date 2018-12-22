import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import cv2
import matplotlib.image as mpimg


# image = data.astronaut()
image = mpimg.imread('../data/barbara.jpg')

fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, multichannel=True, feature_vector=True)

# import pdb; pdb.set_trace()

# ------------- for visualization of HOG features ----------------------------
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ax2.axis('off')
# ax2.imshow(hog_image, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()
# ------------- for visualization of HOG features ----------------------------
