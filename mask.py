import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform_image_into_mask(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary > 0).astype(np.uint8)
    return mask

mask = transform_image_into_mask("example.jpg")

scale = 1
delta = 0
ddepth = cv2.CV_16S

image = cv2.imread("example.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gauss = cv2.GaussianBlur(image_rgb, (9, 9), 0)

Z = gauss.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((gauss.shape))

# Yellow to blue idk why
segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

min_val = np.min(segmented_gray)
final_mask = np.where(segmented_gray == min_val, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(1, 4, figsize=(16,4))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Image")
ax[0].axis("off")

ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Binary Mask")
ax[1].axis("off")

ax[2].imshow(segmented_gray)
ax[2].set_title("GaussBlur + KMeans")
ax[2].axis('off')

ax[3].imshow(final_mask, cmap="gray")
ax[3].set_title("Mask + filters")
ax[3].axis('off')

plt.show()