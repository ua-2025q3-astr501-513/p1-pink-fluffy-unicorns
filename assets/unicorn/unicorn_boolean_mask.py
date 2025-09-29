import cv2 #pip install opencv-python

import numpy as np
import matplotlib.pyplot as plt

path = 'unicorn.jpg'

# 1) Load and prep

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# 2) Threshold (invert so unicorn ~ white); Otsu picks a good cutoff
_, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3) Connect gaps a bit so the outline is continuous
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

# 4) Get ONLY the OUTER contours (no holes)
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5) Pick the biggest contour (the unicorn); adjust if you also grabbed the ground shadow
largest = max(cnts, key=cv2.contourArea)

# 6) Make a clean mask of just that outer shape, filled
mask = np.zeros_like(img, dtype=np.uint8)
cv2.drawContours(mask, [largest], contourIdx=-1, color=255, thickness=cv2.FILLED)

# (Optional) If you also captured the ground shadow and want to drop it, you can
# remove small/flat contours instead of max-by-area. Ask and I’ll give that variant.

# 7) Build the silhouette: unicorn = black (0), background = white (255)
silhouette = np.where(mask==255, 0, 255).astype(np.uint8)

# Show
plt.figure(figsize=(5,5))
plt.imshow(silhouette, cmap='gray')
plt.axis('off')
plt.show()

# Save
cv2.imwrite('unicorn_silhouette.png', silhouette)

plt.figure()
plt.imshow(silhouette, cmap = 'gray')
plt.show()

np.savez("unicorn_boolean_mask", silhouette)