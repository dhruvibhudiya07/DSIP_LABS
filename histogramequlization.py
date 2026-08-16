import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXPERIMENT 8: HISTOGRAM EQUALIZATION AND HISTOGRAM MATCHING
# ============================================================

# Image paths
source_path = r'D:\DSIP_LABs\EXP8\sourseimage.jpg'
reference_path = r'D:\DSIP_LABs\EXP8\referenceimage.jpg'

# ------------------------------------------------------------
# PART 1: Load Source Image
# ------------------------------------------------------------

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

# Check image
if source_image is None:
    print("Error: Source image not found!")
    exit()

print("Source image loaded successfully.")


# ------------------------------------------------------------
# PART 2: Calculate and Display Source Histogram
# ------------------------------------------------------------

source_histogram = cv2.calcHist(
    [source_image], [0], None, [256], [0, 256]
)

plt.figure(figsize=(8, 6))
plt.title('Histogram of Source Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(source_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# PART 3: Histogram Equalization
# ------------------------------------------------------------

equalized_image = cv2.equalizeHist(source_image)

# Display original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PART 4: Histogram of Equalized Image
# ------------------------------------------------------------

equalized_histogram = cv2.calcHist(
    [equalized_image], [0], None, [256], [0, 256]
)

plt.figure(figsize=(8, 6))
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(equalized_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# PART 5: Load Reference Image
# ------------------------------------------------------------

reference_image = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)

if reference_image is None:
    print("Error: Reference image not found!")
    exit()

print("Reference image loaded successfully.")


# ------------------------------------------------------------
# PART 6: Calculate Histograms
# ------------------------------------------------------------

source_hist = cv2.calcHist(
    [source_image], [0], None, [256], [0, 256]
)

reference_hist = cv2.calcHist(
    [reference_image], [0], None, [256], [0, 256]
)

# Normalize histograms
source_hist = source_hist / source_hist.sum()
reference_hist = reference_hist / reference_hist.sum()


# ------------------------------------------------------------
# PART 7: Calculate CDF
# ------------------------------------------------------------

source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()


# ------------------------------------------------------------
# PART 8: Histogram Matching
# ------------------------------------------------------------

mapping = np.interp(
    source_cdf,
    reference_cdf,
    np.arange(256)
)

matched_image = mapping[source_image]

# Convert to uint8
matched_image = matched_image.astype(np.uint8)


# ------------------------------------------------------------
# PART 9: Display Source, Reference and Matched Images
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PART 10: Compare Histograms
# ------------------------------------------------------------

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Source Histogram')
plt.hist(source_image.ravel(), 256, [0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.title('Reference Histogram')
plt.hist(reference_image.ravel(), 256, [0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.title('Matched Histogram')
plt.hist(matched_image.ravel(), 256, [0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()