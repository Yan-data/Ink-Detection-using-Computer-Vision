# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 08:41:23 2024
The data is plotted to understand more about the images.
@author: Yan
"""
import glob
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
#from ipywidgets import interact, fixed
import numpy as np
import os

PREFIX = os.environ.get(PREFIX)
#BUFFER = 30  # Buffer size in x and y direction
Z_START = 30 # First slice in the z direction to use
Z_DIM = 6   # Number of slices in the z direction


plt.imshow(Image.open(PREFIX+"ir.png"), cmap="gray")


mask = np.array(Image.open(PREFIX+"mask.png").convert('1'))
label = np.array(Image.open(PREFIX+"inklabels.png"))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("mask.png")
ax1.imshow(mask, cmap='gray')
ax2.set_title("inklabels.png")
ax2.imshow(label, cmap='gray')
plt.show()

print(np.unique(mask, return_counts=True))
print(np.unique(label, return_counts=True))



# Load the 3d x-ray scan, one slice at a time
images = [np.array(Image.open(filename), dtype=np.float32)/ 65535.0 for filename in tqdm(sorted(glob.glob(PREFIX+"surface_volume/*.tif"))[Z_START:Z_START+Z_DIM])]

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for image, ax in zip(images, axes):
  ax.imshow(np.array(Image.fromarray(image).resize((image.shape[1]//12, image.shape[0]//12)), dtype=np.float32), cmap='gray')
  ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.show()

##https://www.kaggle.com/code/sherkt1/vesuvius-histograms-of-each-sample-object
plt.hist(images[0].ravel(), bins=256, range=(images[0].min(), images[0].max()))
plt.title('Histogram of Pixel Values for Slice 30 in Fragment 1')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#print('image size:', images[1].shape)


rect1 = (1100, 3500, 128, 128)
rect = (1100, 3500, 700, 900)
fig, ax = plt.subplots()
ax.imshow(label)
patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='r', facecolor='none')
patch1 = patches.Rectangle((rect[0], rect1[1]), rect1[2], rect1[3], linewidth=2, edgecolor='b', facecolor='none')
ax.add_patch(patch)
ax.add_patch(patch1)
plt.show()



