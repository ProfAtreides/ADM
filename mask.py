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

gauss = cv2.GaussianBlur(image_rgb, (3, 3), 0)

Z = gauss.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, k_label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[k_label.flatten()]
segmented_image = res.reshape((gauss.shape))

# Yellow to blue idk why
segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

min_val = np.min(segmented_gray)
final_mask = np.where(segmented_gray == min_val, 0, 255).astype(np.uint8)

final_mask = cv2.bitwise_not(final_mask)

CLOSE_KERNEL_SIZE = (7, 7)
CLOSE_ITERATIONS = 1
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL_SIZE)
closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERATIONS)

ERODE_KERNEL_SIZE = (7, 7)
ERODE_ITERATIONS = 1
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ERODE_KERNEL_SIZE)
eroded_mask = cv2.erode(closed_mask, erode_kernel, iterations=ERODE_ITERATIONS)

OPEN_KERNEL_SIZE = (7, 7)
OPEN_ITERATIONS = 1
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL_SIZE)
opened_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, open_kernel, iterations=OPEN_ITERATIONS)

final_mask = opened_mask

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

region_image = np.zeros_like(final_mask, dtype=np.uint8)

for i in range(2, num_labels):
    x, y, w, h, area = stats[i]
    print(area)
    if area <= 25000000:
        cv2.rectangle(region_image, (x, y), (x + w, y + h), 255, 2)

cleared_mask = cv2.bitwise_xor(final_mask, region_image)

fig, ax = plt.subplots(2, 2, figsize=(12,8))
ax[0,0].imshow(image, cmap="gray")
ax[0,0].set_title("Image")
ax[0,0].axis("off")

ax[0,1].imshow(mask, cmap="gray")
ax[0,1].set_title("Binary Mask")
ax[0,1].axis("off")

ax[1,0].imshow(segmented_gray)
ax[1,0].set_title("GaussBlur + KMeans")
ax[1,0].axis('off')

ax[1,1].imshow(final_mask, cmap="gray")
ax[1,1].set_title("Clear beautiful image")
ax[1,1].axis('off')

plt.tight_layout()
plt.show()