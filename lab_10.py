import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("cat.png", cv2.IMREAD_GRAYSCALE)

# Image enhancement
enhanced = cv2.equalizeHist(img)

# Smoothing filters
mean = cv2.blur(img, (3, 3))
gaussian = cv2.GaussianBlur(img, (3, 3), 0)
median = cv2.medianBlur(img, 3)

# Sharpening filter
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharp = cv2.filter2D(img, -1, kernel)

# Compare results
titles = ["Original", "Enhanced", "Mean Filter",
          "Gaussian Filter", "Median Filter", "Sharpened"]
images = [img, enhanced, mean, gaussian, median, sharp]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()