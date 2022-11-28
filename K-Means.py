import cv2
import numpy as np
import matplotlib.pyplot as plt


# read the image
image = cv2.imread("C:/Users/Student/Dropbox/My PC (LAPTOP-LJVP8VBR)/Pictures/Example/tc.JPG")
image1 = cv2.imread("C:/Users/Student/Dropbox/My PC (LAPTOP-LJVP8VBR)/Pictures/Example/tc.JPG")
image2 = cv2.imread("C:/Users/Student/Dropbox/My PC (LAPTOP-LJVP8VBR)/Pictures/Example/tc.JPG")


# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
pixel_values1 = image1.reshape((-1, 3))
pixel_values2 = image2.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)
pixel_values1 = np.float32(pixel_values1)
pixel_values2 = np.float32(pixel_values2)

print(pixel_values.shape)
print(pixel_values1.shape)
print(pixel_values2.shape)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
k1 = 6
k2 = 9
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
_, labels1, (centers1) = cv2.kmeans(pixel_values1, k1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
_, labels2, (centers2) = cv2.kmeans(pixel_values2, k2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)
centers1 = np.uint8(centers1)
centers2 = np.uint8(centers2)

# flatten the labels array
labels = labels.flatten()
labels1 = labels1.flatten()
labels2 = labels2.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]
segmented_image1 = centers1[labels1.flatten()]
segmented_image2 = centers2[labels2.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)
segmented_image1 = segmented_image1.reshape(image1.shape)
segmented_image2 = segmented_image2.reshape(image2.shape)

# show the image


figure_size = 30
plt.figure(figsize=(figure_size,figure_size))
plt.plot(),plt.imshow(segmented_image)
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(figure_size,figure_size))
plt.plot(),plt.imshow(segmented_image1)
plt.title('Segmented Image when K = %i' % k1), plt.xticks([]), plt.yticks([])
plt.figure(figsize=(figure_size,figure_size))
plt.plot(),plt.imshow(segmented_image2)
plt.title('Segmented Image when K = %i' % k2), plt.xticks([]), plt.yticks([])
plt.show()


